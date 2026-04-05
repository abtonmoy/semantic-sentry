"""DriftMonitor orchestrator for drift detection."""

import hashlib
from typing import Any

import numpy as np

from semantic_sentry.adapters import detect_adapter
from semantic_sentry.adapters.base import EncoderAdapter
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.exceptions import (
    AdapterDetectionError,
    AnchorSetMismatchError,
    EmbeddingDimError,
    TowerMismatchError,
)
from semantic_sentry.metrics.registry import get_metric_registry
from semantic_sentry.probes.anchor_set import AnchorSet


class DriftMonitor:
    """Main orchestrator for semantic drift detection.
    
    The DriftMonitor captures snapshots of model embeddings and compares
    them to detect drift over time. It supports both single-tower and
    multi-tower models.
    
    Example:
        monitor = DriftMonitor()
        snapshot_v0 = monitor.snapshot(model_v0, anchor_set)
        snapshot_v1 = monitor.snapshot(model_v1, anchor_set)
        comparison = monitor.compare(snapshot_v0, snapshot_v1)
    """

    def __init__(self):
        """Initialize the drift monitor."""
        self._metric_registry = get_metric_registry()
        self._last_comparison: Comparison | None = None

    def snapshot(
        self,
        model: Any,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None
    ) -> Snapshot:
        """Capture a snapshot of model embeddings.
        
        Args:
            model: Model to snapshot (any supported model type)
            anchor_set: Anchor set to use for encoding
            adapter: Optional adapter (auto-detected if not provided)
            
        Returns:
            Frozen Snapshot of the model state
            
        Raises:
            AdapterDetectionError: If adapter cannot be auto-detected
            TowerMismatchError: If model and adapter tower counts disagree
            EmbeddingDimError: If embeddings have unexpected dimensions
        """
        # Auto-detect adapter if not provided
        if adapter is None:
            try:
                adapter = detect_adapter(model)
            except AdapterDetectionError:
                raise

        # Encode anchor set
        embeddings_dict = adapter.encode_numpy(anchor_set.inputs)

        # Validate tower count
        if adapter.tower_count != len(embeddings_dict):
            raise TowerMismatchError(
                f"Adapter reports {adapter.tower_count} towers but "
                f"encode returned {len(embeddings_dict)} towers"
            )

        # Validate embedding dimensions
        n_samples = anchor_set.n_samples
        for tower_name, emb in embeddings_dict.items():
            if emb.shape[0] != n_samples:
                raise EmbeddingDimError(
                    f"Tower '{tower_name}' has {emb.shape[0]} samples, "
                    f"expected {n_samples}"
                )

        # Compute cross-tower alignment for multi-tower models
        cross_tower_alignment = None
        tower_names = adapter.list_towers()
        if adapter.tower_count > 1:
            cross_tower_alignment = self._compute_cross_tower_alignment(embeddings_dict)

        # Compute checkpoint hash from model state
        checkpoint_hash = self._compute_checkpoint_hash(model)

        # Create snapshot
        snapshot = Snapshot(
            model_id=self._get_model_id(model),
            checkpoint_hash=checkpoint_hash,
            anchor_set_version=anchor_set.version_hash,
            tower_count=adapter.tower_count,
            tower_names=tuple(tower_names),
            embeddings=embeddings_dict,
            cross_tower_alignment=cross_tower_alignment,
        )

        return snapshot

    def compare(
        self,
        snapshot_v0: Snapshot,
        snapshot_v1: Snapshot,
        per_tower: bool = True
    ) -> Comparison:
        """Compare two snapshots and compute drift metrics.
        
        Args:
            snapshot_v0: Base snapshot
            snapshot_v1: Updated snapshot
            per_tower: Whether to compute per-tower metrics for multi-tower models
            
        Returns:
            Comparison result with drift metrics
            
        Raises:
            AnchorSetMismatchError: If snapshots use different anchor sets
            TowerMismatchError: If snapshots have different tower counts/names
            EmbeddingDimError: If embeddings have different dimensions
        """
        # Validate anchor set version
        if snapshot_v0.anchor_set_version != snapshot_v1.anchor_set_version:
            raise AnchorSetMismatchError(
                f"Snapshots captured with different anchor sets: "
                f"{snapshot_v0.anchor_set_version} vs {snapshot_v1.anchor_set_version}"
            )

        # Validate tower structure
        if snapshot_v0.tower_count != snapshot_v1.tower_count:
            raise TowerMismatchError(
                f"Tower count mismatch: {snapshot_v0.tower_count} vs {snapshot_v1.tower_count}"
            )

        if snapshot_v0.tower_names != snapshot_v1.tower_names:
            raise TowerMismatchError(
                f"Tower names mismatch: {snapshot_v0.tower_names} vs {snapshot_v1.tower_names}"
            )

        # Validate embedding dimensions per tower
        for tower_name in snapshot_v0.tower_names:
            emb_v0 = snapshot_v0.get_tower(tower_name)
            emb_v1 = snapshot_v1.get_tower(tower_name)

            if emb_v0.shape != emb_v1.shape:
                raise EmbeddingDimError(
                    f"Tower '{tower_name}' dimension mismatch: "
                    f"{emb_v0.shape} vs {emb_v1.shape}"
                )

        # Compute global metrics (concatenate all towers)
        Z0_global = np.concatenate([snapshot_v0.get_tower(name) for name in snapshot_v0.tower_names], axis=1)
        Z1_global = np.concatenate([snapshot_v1.get_tower(name) for name in snapshot_v1.tower_names], axis=1)

        global_metrics = self._metric_registry.compute_all(Z0_global, Z1_global)

        # Compute per-tower metrics if requested and multi-tower
        per_tower_metrics = None
        if per_tower and snapshot_v0.is_multi_tower:
            per_tower_metrics = {}
            for tower_name in snapshot_v0.tower_names:
                Z0_tower = snapshot_v0.get_tower(tower_name)
                Z1_tower = snapshot_v1.get_tower(tower_name)
                per_tower_metrics[tower_name] = self._metric_registry.compute_all(
                    Z0_tower, Z1_tower
                )

        # Compute alignment deltas for multi-tower
        alignment_deltas = None
        if snapshot_v0.is_multi_tower and snapshot_v0.cross_tower_alignment is not None:
            alignment_deltas = {}
            for pair, v0_value in snapshot_v0.cross_tower_alignment.items():
                v1_value = snapshot_v1.cross_tower_alignment.get(pair, 0.0)
                alignment_deltas[pair] = v1_value - v0_value

        # Create comparison
        comparison = Comparison(
            snapshot_v0_hash=snapshot_v0.checkpoint_hash,
            snapshot_v1_hash=snapshot_v1.checkpoint_hash,
            global_metrics=global_metrics,
            per_tower_metrics=per_tower_metrics,
            alignment_deltas=alignment_deltas,
        )

        # Store for later use
        self._last_comparison = comparison

        return comparison

    @property
    def last_comparison(self) -> Comparison | None:
        """Get the most recent comparison result."""
        return self._last_comparison

    def _compute_cross_tower_alignment(
        self,
        embeddings: dict[str, np.ndarray]
    ) -> dict[tuple[str, str], float]:
        """Compute mean pairwise cosine similarity between towers.
        
        Args:
            embeddings: Dict mapping tower name to embeddings
            
        Returns:
            Dict mapping (tower1, tower2) to mean cosine similarity
        """
        alignment = {}
        tower_names = list(embeddings.keys())

        for i, name1 in enumerate(tower_names):
            for name2 in tower_names[i+1:]:
                emb1 = embeddings[name1]
                emb2 = embeddings[name2]

                # Normalize
                emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-9)
                emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-9)

                # Compute cosine similarities
                similarities = np.sum(emb1_norm * emb2_norm, axis=1)

                # Store mean
                alignment[(name1, name2)] = float(np.mean(similarities))

        return alignment

    def _compute_checkpoint_hash(self, model: Any) -> str:
        """Compute hash of model weights.
        
        Args:
            model: Model to hash
            
        Returns:
            Hash string
        """
        # Try to get state dict
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        elif hasattr(model, 'parameters'):
            # Fallback: hash parameter values
            params = list(model.parameters())
            h = hashlib.sha256()
            for p in params:
                h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()[:16]
        else:
            # Cannot hash, use type name
            return hashlib.sha256(type(model).__name__.encode()).hexdigest()[:16]

        # Hash sorted state dict
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            h.update(key.encode())
            h.update(state_dict[key].detach().cpu().numpy().tobytes())

        return h.hexdigest()[:16]

    def _get_model_id(self, model: Any) -> str:
        """Get a model identifier.
        
        Args:
            model: Model instance
            
        Returns:
            Model ID string
        """
        if hasattr(model, 'name_or_path'):
            return str(model.name_or_path)
        elif hasattr(model, '__class__'):
            return model.__class__.__name__
        else:
            return "unknown_model"
