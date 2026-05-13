"""DriftMonitor orchestrator for drift detection."""

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_sentry.adapters import detect_adapter
from semantic_sentry.adapters.base import EncoderAdapter
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.core.classification import ClassificationResult, ConfidenceLevel
from semantic_sentry.exceptions import (
    AdapterDetectionError,
    AnchorSetMismatchError,
    EmbeddingDimError,
    NoComparisonError,
    TowerMismatchError,
)
from semantic_sentry.metrics.nps import nps_per_point
from semantic_sentry.metrics.registry import get_metric_registry
from semantic_sentry.probes.anchor_set import AnchorSet


@dataclass(frozen=True)
class ClassificationContext:
    """Frozen bundle of everything `classify*()` needs from a comparison.

    Returned from `DriftMonitor.make_classification_context()` and accepted
    by the new `classify(*, context=..., ...)` / `classify_batch(*,
    context=..., ...)` keyword-only signature. Removes the dependence on
    `DriftMonitor._last_*` mutable state that the v0.1.0 audit flagged
    (improvement.md item 1) and lets a single monitor be reused across
    interleaved compare() / classify() calls.
    """
    comparison: Comparison
    v1_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    per_anchor_nps: dict[str, np.ndarray] = field(default_factory=dict)


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
        self._last_anchor_set: AnchorSet | None = None
        self._last_v0_embeddings: dict[str, np.ndarray] | None = None
        self._last_v1_embeddings: dict[str, np.ndarray] | None = None
        # Per-anchor NPS, computed once at compare() time and reused by every
        # subsequent classify() call. Eliminates the O(n²)-per-input cost
        # that the v0.1.0 audit flagged (improvement.md item 2).
        self._last_anchor_per_point_nps: dict[str, np.ndarray] | None = None

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
        self._last_v0_embeddings = {name: snapshot_v0.get_tower(name).copy()
                                     for name in snapshot_v0.tower_names}
        self._last_v1_embeddings = {name: snapshot_v1.get_tower(name).copy()
                                     for name in snapshot_v1.tower_names}
        # Precompute per-anchor NPS (v0 vs v1) once per tower; classify()
        # later looks up the input's nearest anchors and averages their
        # per-anchor NPS values to get a local-drift estimate.
        per_point: dict[str, np.ndarray] = {}
        anchor_n = next(iter(self._last_v0_embeddings.values())).shape[0]
        if anchor_n >= 12:  # need >= k+1 with default k=10
            for name in snapshot_v0.tower_names:
                per_point[name] = nps_per_point(
                    self._last_v0_embeddings[name],
                    self._last_v1_embeddings[name],
                    k=10,
                )
        self._last_anchor_per_point_nps = per_point or None

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
            16-hex-char SHA-256 prefix.
        """
        # Try to get state dict
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            h = hashlib.sha256()
            for key in sorted(state_dict.keys()):
                h.update(key.encode())
                h.update(state_dict[key].detach().cpu().numpy().tobytes())
            return h.hexdigest()[:16]
        elif hasattr(model, 'parameters'):
            params = list(model.parameters())
            h = hashlib.sha256()
            for p in params:
                h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()[:16]
        else:
            # Cannot inspect weights. Hashing just `type(model).__name__`
            # would collide every model of the same class — clearly wrong
            # for a checkpoint hash. Add `id(model)` so different instances
            # do not collide, and warn so the caller knows the hash is
            # process-scoped.
            import warnings
            warnings.warn(
                f"DriftMonitor: model of type {type(model).__name__!r} has no "
                "state_dict() or parameters(); falling back to a process-"
                "scoped identity hash. Two different runs will not see the "
                "same checkpoint_hash for this model.",
                stacklevel=3,
            )
            h = hashlib.sha256()
            h.update(type(model).__name__.encode())
            h.update(str(id(model)).encode())
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

    def make_classification_context(self) -> ClassificationContext:
        """Bundle the data needed by `classify*()` into an explicit context.

        Call after `compare()` to obtain a stateless context object you can
        pass to `classify(*, context=...)` / `classify_batch(*, context=...)`.
        Preferred over the implicit-state form (which is now deprecated).
        """
        if (self._last_comparison is None
                or self._last_v1_embeddings is None):
            raise NoComparisonError("Must call compare() before make_classification_context()")
        return ClassificationContext(
            comparison=self._last_comparison,
            v1_embeddings={k: v.copy() for k, v in self._last_v1_embeddings.items()},
            per_anchor_nps={k: v.copy()
                             for k, v in (self._last_anchor_per_point_nps or {}).items()},
        )

    def classify(
        self,
        input_data: list,
        model: Any,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None,
        k: int = 10,
        *,
        context: ClassificationContext | None = None,
    ) -> ClassificationResult:
        """Classify input data with drift-aware confidence.

        Args:
            input_data: List of inputs to classify
            model: Model to use for encoding
            anchor_set: Anchor set with known labels
            adapter: Optional adapter (auto-detected if not provided)
            k: Number of nearest neighbors to consider
            context: Optional `ClassificationContext` (preferred). When
                provided, the call is fully stateless and does not read from
                `DriftMonitor` instance state. When `None`, the monitor falls
                back to its `_last_*` state with a `DeprecationWarning`.

        Returns:
            Classification result with confidence level

        Raises:
            NoComparisonError: If no comparison has been run yet
        """
        if context is None:
            if self._last_comparison is None:
                raise NoComparisonError("Must call compare() before classify()")
            warnings.warn(
                "DriftMonitor.classify() without an explicit `context=` kwarg "
                "reads mutable instance state from the most recent compare() "
                "call. Prefer `monitor.classify(..., context="
                "monitor.make_classification_context())`. The implicit-state "
                "form will be removed in v0.3.0.",
                DeprecationWarning, stacklevel=2,
            )
            ctx_v1_embeddings = self._last_v1_embeddings or {}
            ctx_per_anchor = self._last_anchor_per_point_nps or {}
        else:
            ctx_v1_embeddings = context.v1_embeddings
            ctx_per_anchor = context.per_anchor_nps

        # Auto-detect adapter if not provided
        if adapter is None:
            adapter = detect_adapter(model)

        # Encode input data
        input_embeddings = adapter.encode_numpy(input_data)
        tower_name = adapter.list_towers()[0]
        Z_input = input_embeddings[tower_name]

        # Get anchor embeddings from v1 (the "new" model)
        Z_anchor = ctx_v1_embeddings[tower_name]

        # Find k nearest anchor points to the input under v1.
        similarities = Z_input[0] @ Z_anchor.T  # cosine sim (anchors already normalised)
        nearest_indices = np.argsort(-similarities)[:k]
        nearest_distances = 1 - similarities[nearest_indices]

        # Local drift estimate: average per-anchor NPS over the input's
        # k nearest anchors. Per-anchor NPS was precomputed once at compare()
        # time, so this is O(k) per input (vs the previous O(n_anchor²) per
        # input, which also returned a trivially-1.0 value because
        # `nps_per_point(M, M)` is identically 1).
        per_anchor_nps = ctx_per_anchor.get(tower_name)
        if per_anchor_nps is None:
            # Anchor set too small for kNN — treat as no-drift signal.
            local_nps = 1.0
        else:
            local_nps = float(np.mean(per_anchor_nps[nearest_indices]))

        # Get labels of nearest anchors
        nearest_labels = [anchor_set.labels[int(i)] for i in nearest_indices]

        # Majority vote for classification
        from collections import Counter
        label_counts = Counter(nearest_labels)
        predicted_label = label_counts.most_common(1)[0][0]

        # Determine confidence level
        if local_nps > 0.90:
            confidence = ConfidenceLevel.HIGH
        elif local_nps > 0.80:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        # Generate drift warning if needed
        drift_warning = ""
        if confidence != ConfidenceLevel.HIGH:
            drift_warning = f"Drift detected: local NPS = {local_nps:.3f}"

        return ClassificationResult(
            label=predicted_label,
            confidence=confidence,
            local_nps=local_nps,
            drift_warning=drift_warning,
            nearest_anchor_indices=tuple(int(i) for i in nearest_indices),
            nearest_anchor_distances=tuple(float(d) for d in nearest_distances),
        )

    def classify_batch(
        self,
        input_data: list,
        model: Any,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None,
        k: int = 10,
        *,
        context: ClassificationContext | None = None,
    ) -> list[ClassificationResult]:
        """Classify a batch of inputs with drift-aware confidence.

        Args:
            input_data: List of inputs to classify
            model: Model to use for encoding
            anchor_set: Anchor set with known labels
            adapter: Optional adapter (auto-detected if not provided)
            k: Number of nearest neighbors to consider
            context: Optional `ClassificationContext` (preferred). When
                `None`, falls back to monitor state with a
                `DeprecationWarning`.

        Returns:
            List of classification results

        Raises:
            NoComparisonError: If no comparison has been run yet
        """
        if context is None:
            if self._last_comparison is None:
                raise NoComparisonError("Must call compare() before classify_batch()")
            warnings.warn(
                "DriftMonitor.classify_batch() without an explicit `context=` "
                "kwarg reads mutable instance state from the most recent "
                "compare() call. Prefer "
                "`monitor.classify_batch(..., context="
                "monitor.make_classification_context())`. The implicit-state "
                "form will be removed in v0.3.0.",
                DeprecationWarning, stacklevel=2,
            )
            ctx_v1_embeddings = self._last_v1_embeddings or {}
            ctx_per_anchor = self._last_anchor_per_point_nps or {}
        else:
            ctx_v1_embeddings = context.v1_embeddings
            ctx_per_anchor = context.per_anchor_nps

        # Auto-detect adapter if not provided
        if adapter is None:
            adapter = detect_adapter(model)

        # Encode all input data at once
        input_embeddings = adapter.encode_numpy(input_data)
        tower_name = adapter.list_towers()[0]
        Z_inputs = input_embeddings[tower_name]

        # Get anchor embeddings from v1
        Z_anchor = ctx_v1_embeddings[tower_name]

        # Compute similarities for all inputs at once
        similarities = Z_inputs @ Z_anchor.T  # (n_inputs, n_anchors)
        # Top-k anchor indices per input in one vectorised step.
        topk_idx = np.argpartition(-similarities, kth=min(k, similarities.shape[1] - 1),
                                    axis=1)[:, :k]
        # Resort within the slice so they're in similarity order.
        for i in range(topk_idx.shape[0]):
            sims_i = similarities[i, topk_idx[i]]
            order = np.argsort(-sims_i)
            topk_idx[i] = topk_idx[i, order]

        per_anchor_nps = ctx_per_anchor.get(tower_name)

        results = []
        for i in range(len(Z_inputs)):
            nearest_indices = topk_idx[i]
            nearest_distances = 1 - similarities[i, nearest_indices]

            nearest_labels = [anchor_set.labels[int(idx)] for idx in nearest_indices]
            from collections import Counter
            predicted_label = Counter(nearest_labels).most_common(1)[0][0]

            # Local NPS = average per-anchor NPS over the input's k nearest
            # anchors. Per-anchor NPS was precomputed at compare() time.
            if per_anchor_nps is None:
                local_nps = 1.0
            else:
                local_nps = float(np.mean(per_anchor_nps[nearest_indices]))

            # Determine confidence
            if local_nps > 0.90:
                confidence = ConfidenceLevel.HIGH
            elif local_nps > 0.80:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW

            drift_warning = ""
            if confidence != ConfidenceLevel.HIGH:
                drift_warning = f"Drift detected: local NPS = {local_nps:.3f}"

            results.append(ClassificationResult(
                label=predicted_label,
                confidence=confidence,
                local_nps=local_nps,
                drift_warning=drift_warning,
                nearest_anchor_indices=tuple(int(idx) for idx in nearest_indices),
                nearest_anchor_distances=tuple(float(d) for d in nearest_distances),
            ))

        return results
