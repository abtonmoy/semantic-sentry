"""DriftMonitor orchestrator for drift detection."""

import hashlib
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from semantic_sentry.adapters import detect_adapter
from semantic_sentry.adapters.base import EncoderAdapter
from semantic_sentry.core.classification import ClassificationResult, ConfidenceLevel
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.core.snapshot import Snapshot

if TYPE_CHECKING:
    from semantic_sentry.core.calibration import SeverityCalibration
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

    def __init__(
        self,
        *,
        baseline_mode: str = "fixed",
        track_temporal: bool = False,
        temporal_metric: str = "cka",
        plateau_eps: float = 0.005,
        plateau_delta: float = 0.001,
        plateau_k: int = 3,
        history_limit: int = 64,
        async_mode: bool = False,
    ):
        """Initialize the drift monitor.

        The keyword-only args configure `track()`'s live-monitoring behaviour;
        all default to the original behaviour, so ``DriftMonitor()`` is
        unchanged.

        Args:
            baseline_mode: ``"fixed"`` (default) compares every checkpoint to
                the first one seen — "drift since monitoring started".
                ``"previous"`` compares each checkpoint to the immediately
                preceding one — a sliding-window / step-to-step view.
            track_temporal: When True, `track()` accumulates a bounded window
                of snapshots and reports velocity / acceleration / plateau of
                ``temporal_metric`` under ``Comparison.metadata["temporal"]``.
            temporal_metric: Which registered metric the temporal signals wrap
                (default ``"cka"``).
            plateau_eps: Velocity threshold for the plateau detector.
            plateau_delta: Acceleration threshold for the plateau detector.
            plateau_k: Consecutive checkpoints required to declare a plateau.
            history_limit: Max snapshots retained for temporal signals (the
                pinned baseline is kept separately and is never trimmed).
            async_mode: When True, `track()` captures the current embeddings
                synchronously (it must, before training mutates the weights)
                but runs the metric computation + logging on a background
                worker and returns a `concurrent.futures.Future`. Call
                `drain()` / `close()` to flush pending work.
        """
        if baseline_mode not in ("fixed", "previous"):
            raise ValueError(
                f"baseline_mode must be 'fixed' or 'previous', got {baseline_mode!r}"
            )
        self._metric_registry = get_metric_registry()
        self._last_comparison: Comparison | None = None
        self._last_anchor_set: AnchorSet | None = None
        self._last_v0_embeddings: dict[str, np.ndarray] | None = None
        self._last_v1_embeddings: dict[str, np.ndarray] | None = None
        # Rolling baseline for `track()`. The first `track()` call (or an
        # explicit `set_baseline()`) records the reference snapshot; every
        # subsequent call compares the current model against it. Kept separate
        # from the `_last_*` compare()/classify() state so interleaving the two
        # APIs on one monitor doesn't cross-contaminate.
        self._baseline_snapshot: Snapshot | None = None
        self._baseline_device_embeddings: dict[str, Any] | None = None
        self._baseline_anchor_version: str = ""
        # --- live-tracking config (A: baseline mode, B: temporal) ---
        self._baseline_mode = baseline_mode
        self._track_temporal = track_temporal
        self._temporal_metric = temporal_metric
        self._plateau_eps = plateau_eps
        self._plateau_delta = plateau_delta
        self._plateau_k = plateau_k
        self._history_limit = max(2, history_limit)
        self._history: list[Snapshot] = []
        self._history_times: list[float] = []
        # --- async plumbing (C) ---
        self._async_mode = async_mode
        self._executor: ThreadPoolExecutor | None = None
        self._pending: list[Future] = []
        self._last_result: Comparison | None = None
        import threading
        self._async_lock = threading.Lock()
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

        # H2: thread anchor-set provenance into snapshot metadata so it
        # survives save/load and is available at compare() time.
        snapshot_metadata: dict[str, Any] = {}
        if anchor_set.distribution_tag:
            snapshot_metadata["distribution_tag"] = anchor_set.distribution_tag

        # Create snapshot
        snapshot = Snapshot(
            model_id=self._get_model_id(model),
            checkpoint_hash=checkpoint_hash,
            anchor_set_version=anchor_set.version_hash,
            tower_count=adapter.tower_count,
            tower_names=tuple(tower_names),
            embeddings=embeddings_dict,
            cross_tower_alignment=cross_tower_alignment,
            metadata=snapshot_metadata,
        )

        return snapshot

    def compare(
        self,
        snapshot_v0: Snapshot,
        snapshot_v1: Snapshot,
        per_tower: bool = True,
        d_snapshot_v0: Snapshot | None = None,
        d_snapshot_v1: Snapshot | None = None,
        calibration: "SeverityCalibration | None" = None,
        anchor_set: AnchorSet | None = None,
        evaluators: "list[Any] | dict[str, Any] | None" = None,
    ) -> Comparison:
        """Compare two snapshots and compute drift metrics.

        Args:
            snapshot_v0: Base snapshot (Q-side, when paired with d_snapshot_*)
            snapshot_v1: Updated snapshot (Q-side, when paired with d_snapshot_*)
            per_tower: Whether to compute per-tower metrics for multi-tower models
            d_snapshot_v0: Optional D-side baseline snapshot. When provided
                together with ``d_snapshot_v1``, the behavioral registry
                (lib_enhancement Family A) is evaluated and its results
                merged into ``global_metrics`` under their registered names.
                Both Q-snapshots must have role ``"Q"`` and both D-snapshots
                must have role ``"D"`` from the same partition (enforced via
                composite anchor_set_version hashes from ``AnchorSet.partition``).
            d_snapshot_v1: Optional D-side updated snapshot, paired with
                ``d_snapshot_v0``.

        Returns:
            Comparison result with drift metrics

        Raises:
            AnchorSetMismatchError: If snapshots use different anchor sets
            TowerMismatchError: If snapshots have different tower counts/names
            EmbeddingDimError: If embeddings have different dimensions
            ValueError: If only one of ``d_snapshot_v0``/``d_snapshot_v1`` provided
        """
        if (d_snapshot_v0 is None) ^ (d_snapshot_v1 is None):
            raise ValueError(
                "d_snapshot_v0 and d_snapshot_v1 must both be provided or both omitted"
            )
        # Validate anchor set version
        if snapshot_v0.anchor_set_version != snapshot_v1.anchor_set_version:
            raise AnchorSetMismatchError(
                f"Snapshots captured with different anchor sets: "
                f"{snapshot_v0.anchor_set_version} vs {snapshot_v1.anchor_set_version}"
            )

        # H2: validate anchor-distribution provenance matches. Snapshots from
        # different anchor distributions are categorically not comparable —
        # §5.2 showed 2.3x NPS magnitude differences between training-dist and
        # OOD anchors. Allow comparison if neither side declared a tag (legacy
        # behavior) but reject any mismatch where at least one side has one.
        v0_tag = snapshot_v0.metadata.get("distribution_tag", "")
        v1_tag = snapshot_v1.metadata.get("distribution_tag", "")
        if v0_tag != v1_tag:
            raise AnchorSetMismatchError(
                f"Anchor distribution tag mismatch: "
                f"v0='{v0_tag}' vs v1='{v1_tag}'"
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
        Z0_global = np.concatenate(
            [snapshot_v0.get_tower(name) for name in snapshot_v0.tower_names], axis=1
        )
        Z1_global = np.concatenate(
            [snapshot_v1.get_tower(name) for name in snapshot_v1.tower_names], axis=1
        )

        global_metrics = self._metric_registry.compute_all(Z0_global, Z1_global)

        # Behavioral metrics (lib_enhancement A1-A5): require D-side snapshots
        # to compute (q, d) score-distribution / ranking drift. They live in a
        # separate registry because their signature is (Z0_Q, Z1_Q, D0, D1, ...).
        if d_snapshot_v0 is not None and d_snapshot_v1 is not None:
            # Anchor-set version coupling: both D snapshots must come from the
            # paired partition of the same parent set.
            if d_snapshot_v0.anchor_set_version != d_snapshot_v1.anchor_set_version:
                raise AnchorSetMismatchError(
                    f"D-side anchor set version mismatch: "
                    f"{d_snapshot_v0.anchor_set_version} vs "
                    f"{d_snapshot_v1.anchor_set_version}"
                )
            from semantic_sentry.metrics.behavioral import get_behavioral_registry
            behavioral_registry = get_behavioral_registry()
            if behavioral_registry.list_metrics():
                D0_global = np.concatenate(
                    [d_snapshot_v0.get_tower(name) for name in d_snapshot_v0.tower_names],
                    axis=1,
                )
                D1_global = np.concatenate(
                    [d_snapshot_v1.get_tower(name) for name in d_snapshot_v1.tower_names],
                    axis=1,
                )
                behavioral_metrics = behavioral_registry.compute_all(
                    Z0_global, Z1_global, D0_global, D1_global
                )
                global_metrics.update(behavioral_metrics)

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

        # H2: thread distribution_tag through to Comparison.metadata so
        # downstream consumers (calibration, dashboards) can see anchor
        # provenance.
        comparison_metadata: dict[str, Any] = {}
        if v0_tag:
            comparison_metadata["distribution_tag"] = v0_tag

        # Downstream proxy deltas: when labelled anchors + evaluators are
        # supplied, run each evaluator's v0->v1 delta and stash it under
        # comparison.metadata["downstream"]. This surfaces a *measured* task
        # signal (e.g. retrieval MRR change) alongside the purely geometric
        # metrics, which the README is explicit about not predicting.
        if evaluators is not None and anchor_set is not None:
            downstream = self._compute_downstream(
                snapshot_v0, snapshot_v1, anchor_set, evaluators
            )
            if downstream:
                comparison_metadata["downstream"] = downstream

        # I2: pass calibration-derived thresholds (if any) into Comparison so
        # severity is computed against the noise-floor bands instead of the
        # hard-coded defaults.
        thresholds_arg = dict(calibration.thresholds) if calibration is not None else {}

        # Create comparison
        comparison = Comparison(
            snapshot_v0_hash=snapshot_v0.checkpoint_hash,
            snapshot_v1_hash=snapshot_v1.checkpoint_hash,
            global_metrics=global_metrics,
            per_tower_metrics=per_tower_metrics,
            alignment_deltas=alignment_deltas,
            thresholds=thresholds_arg,
            metadata=comparison_metadata,
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

    def _compute_downstream(
        self,
        snapshot_v0: Snapshot,
        snapshot_v1: Snapshot,
        anchor_set: AnchorSet,
        evaluators: "list[Any] | dict[str, Any]",
    ) -> dict[str, float]:
        """Run each evaluator's v0->v1 delta. Skips silently if unlabelled.

        Accepts either a list of `Evaluator` instances (keyed by class name)
        or a ``{name: Evaluator}`` dict. Evaluators that raise (e.g. because
        the anchor set has no labels) are skipped rather than failing the
        whole comparison.
        """
        if isinstance(evaluators, dict):
            items = list(evaluators.items())
        else:
            items = [(type(ev).__name__, ev) for ev in evaluators]

        deltas: dict[str, float] = {}
        for name, evaluator in items:
            try:
                deltas[name] = float(
                    evaluator.evaluate_delta(snapshot_v0, snapshot_v1, anchor_set)
                )
            except Exception as exc:  # noqa: BLE001 - one bad evaluator shouldn't abort
                warnings.warn(
                    f"Evaluator {name!r} failed and was skipped: {exc}",
                    stacklevel=3,
                )
        return deltas

    def set_baseline(
        self,
        model: Any,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None,
    ) -> Snapshot:
        """Record the reference snapshot that `track()` compares against.

        Usually you don't need to call this — the first `track()` call sets
        the baseline automatically. Use it to pin a specific known-good
        checkpoint as the reference up front.

        Returns:
            The captured baseline `Snapshot`.
        """
        baseline = self.snapshot(model, anchor_set, adapter=adapter)
        self._baseline_snapshot = baseline
        self._baseline_anchor_version = baseline.anchor_set_version
        self._baseline_device_embeddings = None
        self._history = [baseline]
        self._history_times = [0.0]
        return baseline

    # --- live-tracking internals (A/B/C) ------------------------------------

    def _select_reference(self, current: Snapshot) -> Snapshot | None:
        """Pick what ``current`` is compared against, honouring baseline_mode.

        Returns ``None`` on the establishing call (nothing to compare yet).
        Must be called *before* ``current`` is appended to the history window
        so ``"previous"`` mode sees the prior snapshot, not ``current`` itself.
        """
        if self._baseline_mode == "fixed":
            if self._baseline_snapshot is None:
                self._baseline_snapshot = current
                return None
            return self._baseline_snapshot
        # "previous": sliding-window — compare against the immediately prior.
        return self._history[-1] if self._history else None

    def _append_history(self, snap: Snapshot, t: float) -> None:
        """Append to the bounded temporal-history window (drops the oldest)."""
        self._history.append(snap)
        self._history_times.append(t)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
            self._history_times.pop(0)

    def _compute_temporal(
        self,
        history: list[Snapshot],
        times: list[float],
    ) -> dict[str, Any]:
        """Velocity / acceleration / plateau of the latest checkpoint.

        Returns ``{}`` when temporal tracking is off or there isn't enough
        history yet. Operates on the supplied (copied) lists so it is safe to
        run on the async worker thread.
        """
        if not self._track_temporal or len(history) < 2:
            return {}
        from semantic_sentry.metrics.temporal import (
            acceleration,
            plateau,
            velocity,
        )
        try:
            v = velocity(history, times, self._temporal_metric)
            a = acceleration(history, times, self._temporal_metric)
            p = plateau(
                history, times, self._temporal_metric,
                eps=self._plateau_eps, delta=self._plateau_delta, k=self._plateau_k,
            )
        except Exception as exc:  # noqa: BLE001 - temporal is best-effort
            warnings.warn(f"temporal signals skipped: {exc}", stacklevel=3)
            return {}
        return {
            "metric": self._temporal_metric,
            "velocity": float(v[-1]),
            "acceleration": float(a[-1]),
            "plateau": bool(p[-1]),
        }

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="semantic-sentry"
            )
        return self._executor

    def _submit(self, fn: Any, *args: Any) -> Future:
        """Submit ``fn(*args)`` to the single background worker (async mode)."""
        fut = self._ensure_executor().submit(fn, *args)
        with self._async_lock:
            self._pending.append(fut)
        fut.add_done_callback(self._on_future_done)
        return fut

    def _on_future_done(self, fut: Future) -> None:
        with self._async_lock:
            if fut in self._pending:
                self._pending.remove(fut)

    def _finalize(
        self,
        reference: Snapshot,
        current: Snapshot,
        history: list[Snapshot],
        times: list[float],
        calibration: "SeverityCalibration | None",
        anchor_set: AnchorSet,
        evaluators: "list[Any] | dict[str, Any] | None",
        logger: Any | None,
        step: int | None,
    ) -> Comparison:
        """Compare + temporal + log. Runs inline (sync) or on the worker (async)."""
        comparison = self.compare(
            reference,
            current,
            calibration=calibration,
            anchor_set=anchor_set if evaluators is not None else None,
            evaluators=evaluators,
        )
        temporal = self._compute_temporal(history, times)
        if temporal:
            comparison = replace(
                comparison,
                metadata={**comparison.metadata, "temporal": temporal},
            )
        if logger is not None:
            logger.log_comparison(comparison, step=step)
        with self._async_lock:
            self._last_result = comparison
        return comparison

    def track(
        self,
        model: Any,
        anchor_set: AnchorSet,
        *,
        step: int | None = None,
        adapter: EncoderAdapter | None = None,
        logger: Any | None = None,
        calibration: "SeverityCalibration | None" = None,
        evaluators: "list[Any] | dict[str, Any] | None" = None,
        keep_on_device: bool = False,
    ) -> "Comparison | Future | None":
        """One-line drift tracking against a rolling baseline.

        The first call records the baseline and returns ``None``. Every
        subsequent call snapshots the current model, compares it to the
        reference (per ``baseline_mode``), optionally logs, and returns the
        `Comparison`. Designed to drop into a training loop or callback:

            monitor = DriftMonitor(track_temporal=True)
            for step, ckpt in enumerate(checkpoints):
                cmp = monitor.track(ckpt, anchors, step=step, logger=wandb_logger)
                if cmp and cmp.metadata.get("temporal", {}).get("plateau"):
                    break  # geometry has settled — stop early

        Behaviour is governed by the monitor's constructor config:

        - **baseline_mode="fixed"** — compare to the first snapshot
          (drift since start); **"previous"** — compare to the prior
          checkpoint (sliding window).
        - **track_temporal=True** — attach velocity / acceleration / plateau
          under ``comparison.metadata["temporal"]``.
        - **async_mode=True** — return a `Future`; embeddings are captured
          synchronously (before the weights move on) but the metric
          computation + logging run on a background worker. Use `drain()` /
          `close()` to flush, and `last_result` for the latest completed
          comparison.

        Args:
            model: Current model/checkpoint to evaluate.
            anchor_set: Fixed anchor set (must match the baseline's).
            step: Optional step/epoch; used as the temporal time axis and
                forwarded to ``logger.log_comparison``.
            adapter: Optional adapter (auto-detected if omitted).
            logger: Optional `DriftLogger`; ``log_comparison`` is called when
                a comparison is produced.
            calibration: Optional `SeverityCalibration` for noise-floor bands.
            evaluators: Optional downstream evaluators (see `compare`). Ignored
                in ``keep_on_device`` mode (evaluators need numpy snapshots).
            keep_on_device: Compute the built-in metrics with the torch backend
                directly from the encoded tensors, skipping the snapshot/numpy
                round-trip. Global metrics only — no per-tower / behavioral /
                downstream / temporal.

        Returns:
            ``None`` on the baseline-setting call; a `Comparison` in sync mode;
            a `Future[Comparison]` in async mode.

        Raises:
            AnchorSetMismatchError: If the anchor set differs from the baseline.
        """
        if keep_on_device:
            return self._track_on_device(
                model, anchor_set, step=step, adapter=adapter,
                logger=logger, calibration=calibration,
            )

        current = self.snapshot(model, anchor_set, adapter=adapter)
        t = float(step) if step is not None else float(len(self._history))

        reference = self._select_reference(current)
        self._append_history(current, t)
        if reference is None:
            # Establishing call — nothing to compare against yet.
            self._baseline_anchor_version = current.anchor_set_version
            return None

        args = (
            reference, current, list(self._history), list(self._history_times),
            calibration, anchor_set, evaluators, logger, step,
        )
        if self._async_mode:
            return self._submit(self._finalize, *args)
        return self._finalize(*args)

    def _track_on_device(
        self,
        model: Any,
        anchor_set: AnchorSet,
        *,
        step: int | None,
        adapter: EncoderAdapter | None,
        logger: Any | None,
        calibration: "SeverityCalibration | None",
    ) -> "Comparison | Future | None":
        """`track(keep_on_device=True)` — torch metrics, no numpy round-trip.

        Honours ``baseline_mode`` (``"previous"`` rolls the device baseline
        forward each call) and ``async_mode`` (the torch metric computation +
        logging are deferred to the worker; the embedding capture stays
        synchronous because it must observe the current weights). Temporal
        signals are not available on this path — there are no `Snapshot`s.
        """
        from semantic_sentry.metrics.torch_backend import compute_drift_metrics_torch

        if adapter is None:
            adapter = detect_adapter(model)

        # encode() returns on-device tensors; we never call encode_numpy here.
        tensors = adapter.encode(anchor_set.inputs)
        tower_names = adapter.list_towers()
        import torch
        current = torch.cat([tensors[name] for name in tower_names], dim=1).detach()
        current_hash = self._compute_checkpoint_hash(model)

        if self._baseline_device_embeddings is None:
            self._baseline_device_embeddings = {"global": current}
            self._baseline_anchor_version = anchor_set.version_hash
            self._baseline_snapshot = None
            return None

        if self._baseline_anchor_version != anchor_set.version_hash:
            raise AnchorSetMismatchError(
                f"track() anchor set changed under the baseline: "
                f"{self._baseline_anchor_version} vs {anchor_set.version_hash}"
            )

        base = self._baseline_device_embeddings["global"]
        tag = anchor_set.distribution_tag
        thresholds_arg = dict(calibration.thresholds) if calibration is not None else {}
        # "previous" mode: the next call compares against this checkpoint.
        if self._baseline_mode == "previous":
            self._baseline_device_embeddings = {"global": current}

        def _compute() -> Comparison:
            global_metrics = compute_drift_metrics_torch(base, current)
            metadata: dict[str, Any] = {"backend": "torch", "keep_on_device": True}
            if tag:
                metadata["distribution_tag"] = tag
            comparison = Comparison(
                snapshot_v0_hash="device-baseline",
                snapshot_v1_hash=current_hash,
                global_metrics=global_metrics,
                thresholds=thresholds_arg,
                metadata=metadata,
            )
            if logger is not None:
                logger.log_comparison(comparison, step=step)
            with self._async_lock:
                self._last_result = comparison
            return comparison

        if self._async_mode:
            return self._submit(_compute)
        return _compute()

    @property
    def last_result(self) -> Comparison | None:
        """Most recently completed `track()` comparison (useful in async mode)."""
        with self._async_lock:
            return self._last_result

    def drain(self, timeout: float | None = None) -> None:
        """Block until all pending async `track()` jobs have finished."""
        with self._async_lock:
            pending = list(self._pending)
        for fut in pending:
            fut.result(timeout=timeout)

    def close(self) -> None:
        """Flush pending async work and shut down the background worker."""
        self.drain()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

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
