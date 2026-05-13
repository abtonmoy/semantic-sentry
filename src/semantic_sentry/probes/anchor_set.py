"""Anchor set dataclass for probe management."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchorSet:
    """Immutable anchor set for drift detection probes.

    Attributes:
        inputs: The input data (text, images, etc.) for the anchor set
        labels: Optional labels for the anchor points
        version_hash: Deterministic hash computed from serialized inputs
        modality: The modality of the data (e.g., 'text', 'image', 'multimodal')
        n_samples: Number of samples in the anchor set
        distribution_tag: Optional provenance label (e.g. ``"training-dist"``,
            ``"OOD"``, ``"deployment-prod"``). Threaded into snapshot and
            comparison metadata so cross-experiment comparisons can flag
            anchor-distribution mismatches (see lib_enhancement H2).
        parent_hash: For partitioned anchor sets, the ``version_hash`` of the
            parent set. Empty for root sets. Combined with ``role`` to derive
            composite version hashes that let paired Q/D snapshots cross-verify
            without colliding on the parent's hash (see lib_enhancement H1).
        role: Partition role — ``""`` for root sets, ``"Q"`` for the query
            partition, ``"D"`` for the document partition.
        partition_seed: Seed used to produce this partition; recorded so the
            composite version_hash is stable across reruns with the same seed.
    """
    inputs: Any
    labels: tuple[Any, ...] = field(default_factory=tuple)
    modality: str = "text"
    version_hash: str = field(default="", repr=False)
    n_samples: int = field(default=0)
    distribution_tag: str = ""
    parent_hash: str = ""
    role: str = ""
    partition_seed: int = 0

    def __post_init__(self):
        """Compute version_hash and n_samples if not provided."""
        if not self.version_hash:
            if self.parent_hash:
                # Composite hash for partitions — paired Q and D snapshots
                # from the same parent + seed cross-verify; mixing roles or
                # seeds raises in DriftMonitor.compare.
                composite = f"{self.parent_hash}:{self.role}:{self.partition_seed}"
                object.__setattr__(self, 'version_hash', composite)
            else:
                hash_val = self._compute_hash(self.inputs)
                object.__setattr__(self, 'version_hash', hash_val)

        if self.n_samples == 0:
            n = self._infer_n_samples(self.inputs)
            object.__setattr__(self, 'n_samples', n)

    def __len__(self) -> int:
        return self.n_samples

    def partition(
        self,
        ratio: float = 0.5,
        seed: int = 0,
    ) -> tuple["AnchorSet", "AnchorSet"]:
        """Split into a query and document subset.

        Both subsets inherit ``modality``, ``distribution_tag``, and the parent's
        ``version_hash`` (so paired Q/D snapshots can cross-verify in
        ``DriftMonitor.compare``). The partition is reproducible under the
        same seed.

        Args:
            ratio: Fraction of samples assigned to the query subset. Default 0.5.
            seed: RNG seed for the permutation.

        Returns:
            ``(anchor_q, anchor_d)`` tuple.

        Raises:
            ValueError: If ``ratio`` is not in (0, 1) or there are fewer than
                2 samples.
        """
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if self.n_samples < 2:
            raise ValueError(
                f"AnchorSet must have >=2 samples to partition, got {self.n_samples}"
            )

        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_samples)
        q_size = max(1, min(self.n_samples - 1, int(self.n_samples * ratio)))
        q_idx = perm[:q_size]
        d_idx = perm[q_size:]

        q_inputs = _index_inputs(self.inputs, q_idx)
        d_inputs = _index_inputs(self.inputs, d_idx)
        q_labels = _index_labels(self.labels, q_idx)
        d_labels = _index_labels(self.labels, d_idx)

        return (
            AnchorSet(
                inputs=q_inputs,
                labels=q_labels,
                modality=self.modality,
                distribution_tag=self.distribution_tag,
                parent_hash=self.version_hash,
                role="Q",
                partition_seed=seed,
            ),
            AnchorSet(
                inputs=d_inputs,
                labels=d_labels,
                modality=self.modality,
                distribution_tag=self.distribution_tag,
                parent_hash=self.version_hash,
                role="D",
                partition_seed=seed,
            ),
        )

    @staticmethod
    def _compute_hash(inputs: Any) -> str:
        """Compute deterministic hash from inputs."""
        try:
            if isinstance(inputs, (list, tuple)):
                serialized = json.dumps(inputs, sort_keys=True)
            elif isinstance(inputs, str):
                serialized = inputs
            elif hasattr(inputs, 'tolist'):
                serialized = json.dumps(inputs.tolist())
            else:
                serialized = str(inputs)
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]
        except (TypeError, ValueError):
            return hashlib.sha256(str(inputs).encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def _infer_n_samples(inputs: Any) -> int:
        """Infer number of samples from inputs."""
        if hasattr(inputs, '__len__'):
            return len(inputs)
        if hasattr(inputs, 'shape'):
            return inputs.shape[0]
        return 0


def _index_inputs(inputs: Any, idx: np.ndarray) -> Any:
    """Select rows of inputs by integer index, preserving container type where possible."""
    if isinstance(inputs, list):
        return [inputs[int(i)] for i in idx]
    if isinstance(inputs, tuple):
        return tuple(inputs[int(i)] for i in idx)
    if hasattr(inputs, 'shape'):  # numpy / torch-like
        return inputs[idx]
    return [inputs[int(i)] for i in idx]


def _index_labels(labels: tuple[Any, ...], idx: np.ndarray) -> tuple[Any, ...]:
    if not labels:
        return ()
    seq = list(labels) if not hasattr(labels, '__getitem__') else labels
    return tuple(seq[int(i)] for i in idx)
