"""Snapshot dataclass for frozen model state capture."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from semantic_sentry.exceptions import SnapshotCorruptionError


@dataclass(frozen=True, eq=False)
class Snapshot:
    """Frozen snapshot of model embedding state.

    Attributes:
        model_id: Identifier for the model
        checkpoint_hash: Hash of model weights for integrity
        timestamp: When the snapshot was captured
        anchor_set_version: Hash of the anchor set used
        tower_count: Number of towers (1 for single-tower, >1 for multi-tower)
        tower_names: Ordered list of tower names
        embeddings: Dict mapping tower name to (n, d) embedding matrix
        cross_tower_alignment: Dict of mean pairwise cosine similarities for tower pairs
        metadata: Additional metadata dict
    """
    model_id: str
    checkpoint_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    anchor_set_version: str = ""
    tower_count: int = 1
    tower_names: tuple[str, ...] = field(default_factory=lambda: ("encoder",))
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    cross_tower_alignment: dict | None = None
    metadata: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        # `frozen=True` disables setattr but the default `__hash__` is dropped
        # whenever `__eq__` is auto-generated. We disable __eq__ (eq=False)
        # because `embeddings: dict` and `metadata: dict` are unhashable, then
        # restore a content hash from the integrity fields.
        return hash((self.model_id, self.checkpoint_hash, self.timestamp,
                     self.anchor_set_version, self.tower_count, self.tower_names))

    def __post_init__(self):
        """Validate snapshot consistency."""
        # Validate tower_count matches tower_names length
        if self.tower_count != len(self.tower_names):
            raise ValueError(
                f"tower_count ({self.tower_count}) must match len(tower_names) "
                f"({len(self.tower_names)})"
            )

        # Validate tower_count matches embeddings keys
        if self.tower_count != len(self.embeddings):
            raise ValueError(
                f"tower_count ({self.tower_count}) must match len(embeddings) "
                f"({len(self.embeddings)})"
            )

        # Validate all embeddings have same n dimension
        if self.embeddings:
            n_dims = [emb.shape[0] for emb in self.embeddings.values()]
            if len(set(n_dims)) > 1:
                raise ValueError(
                    f"All embedding matrices must have same n dimension, got: {n_dims}"
                )

            # Validate tower names match embedding keys
            emb_keys = set(self.embeddings.keys())
            tower_keys = set(self.tower_names)
            if emb_keys != tower_keys:
                raise ValueError(
                    f"Embeddings keys {emb_keys} must match tower_names {tower_keys}"
                )

    @property
    def is_multi_tower(self) -> bool:
        """Return True if this is a multi-tower snapshot."""
        return self.tower_count > 1

    def get_tower(self, name: str) -> np.ndarray:
        """Get embedding matrix for a named tower.

        Args:
            name: Tower name

        Returns:
            Embedding matrix of shape (n, d)

        Raises:
            KeyError: If tower name not found
        """
        if name not in self.embeddings:
            raise KeyError(f"Tower '{name}' not found. Available: {list(self.embeddings.keys())}")
        return self.embeddings[name]

    def save(self, path: str | Path) -> None:
        """Save snapshot to directory.

        Creates directory structure:
            path/
                metadata.json
                {tower_name}.safetensors  (one per tower)

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each tower's embeddings
        for tower_name, emb_matrix in self.embeddings.items():
            save_file({tower_name: emb_matrix}, path / f"{tower_name}.safetensors")

        # Save metadata
        embeddings_hash = self._compute_embeddings_hash(self.embeddings)
        metadata = {
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "embeddings_hash": embeddings_hash,
            "timestamp": self.timestamp,
            "anchor_set_version": self.anchor_set_version,
            "tower_count": self.tower_count,
            "tower_names": list(self.tower_names),
            # Store as a list of [tower_a, tower_b, value] triples to avoid the
            # "tower-name contains __" ambiguity in the legacy `a__b` key form.
            # The legacy form is still accepted on load.
            "cross_tower_alignment": [
                [k[0], k[1], v]
                for k, v in (self.cross_tower_alignment or {}).items()
            ],
            "metadata": self.metadata,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        """Load snapshot from directory.

        Args:
            path: Directory path to load from

        Returns:
            Loaded Snapshot

        Raises:
            SnapshotCorruptionError: If integrity check fails
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load embeddings from safetensors files
        embeddings = {}
        for tower_name in metadata["tower_names"]:
            tensor_path = path / f"{tower_name}.safetensors"
            with safe_open(tensor_path, framework="np") as f:
                embeddings[tower_name] = f.get_tensor(tower_name)

        # Reconstruct cross_tower_alignment. The on-disk shape may be either
        # the new triple-list form (preferred) or the legacy "a__b" dict form.
        cta_raw = metadata.get("cross_tower_alignment")
        cross_tower_alignment: dict[tuple[str, str], float] | None = None
        if cta_raw:
            cross_tower_alignment = {}
            if isinstance(cta_raw, list):
                for entry in cta_raw:
                    if len(entry) != 3:
                        raise SnapshotCorruptionError(
                            f"Malformed cross_tower_alignment entry: {entry!r}"
                        )
                    a, b, v = entry
                    cross_tower_alignment[(a, b)] = v
            elif isinstance(cta_raw, dict):
                # Legacy "a__b" form. Best-effort split; ambiguous when names
                # contain "__" themselves.
                for k, v in cta_raw.items():
                    parts = k.split("__", 1)
                    if len(parts) != 2:
                        raise SnapshotCorruptionError(
                            f"Malformed cross_tower_alignment key: {k!r}"
                        )
                    cross_tower_alignment[(parts[0], parts[1])] = v
            else:
                raise SnapshotCorruptionError(
                    f"cross_tower_alignment must be list or dict, got "
                    f"{type(cta_raw).__name__}"
                )

        # Integrity check (if embeddings_hash is present)
        if "embeddings_hash" in metadata:
            computed_hash = cls._compute_embeddings_hash(embeddings)
            if computed_hash != metadata["embeddings_hash"]:
                raise SnapshotCorruptionError(
                    f"Snapshot integrity check failed. Expected {metadata['embeddings_hash']}, "
                    f"got {computed_hash}"
                )

        # Integrity check for checkpoint_hash (if present in metadata)
        if "checkpoint_hash" in metadata:
            stored_checkpoint_hash = metadata["checkpoint_hash"]
            # The checkpoint_hash cannot be recomputed from the snapshot alone
            # (it requires the original model), so we just verify it's present
            # and not obviously tampered with (non-empty)
            if not stored_checkpoint_hash:
                raise SnapshotCorruptionError(
                    f"Invalid checkpoint_hash in metadata: {stored_checkpoint_hash}"
                )

        snapshot = cls(
            model_id=metadata["model_id"],
            checkpoint_hash=metadata["checkpoint_hash"],
            timestamp=metadata["timestamp"],
            anchor_set_version=metadata["anchor_set_version"],
            tower_count=metadata["tower_count"],
            tower_names=tuple(metadata["tower_names"]),
            embeddings=embeddings,
            cross_tower_alignment=cross_tower_alignment,
            metadata=metadata.get("metadata", {}),
        )

        return snapshot

    @staticmethod
    def _compute_embeddings_hash(embeddings: dict[str, np.ndarray]) -> str:
        """Compute hash of embeddings for integrity checking."""
        h = hashlib.sha256()
        for name in sorted(embeddings.keys()):
            h.update(name.encode())
            h.update(embeddings[name].tobytes())
        return h.hexdigest()[:16]
