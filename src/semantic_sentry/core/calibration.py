"""Severity calibration from multi-seed reference snapshots (lib_enhancement I2).

Replaces the hard-coded NPS/CKA severity thresholds in
``Comparison._DEFAULT_THRESHOLDS`` with per-model noise floors derived from
the standard deviation of metric values across multiple reference runs.

A reference snapshot set is typically: the same checkpoint encoded N times
with different random seeds (e.g., dropout-on inference), or a sequence of
known-good consecutive checkpoints from a stable training run. The
calibrator measures how much the metrics naturally wobble across these
"no real drift" pairs, then sets severity bands at multiples of the
observed standard deviation.

For similarity metrics like ``nps`` and ``cka`` (higher = better,
range = [0, 1]), tier ``i`` is set at ``mu - n_sigmas[i] * sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from semantic_sentry.metrics.registry import MetricRegistry, get_metric_registry

if TYPE_CHECKING:
    from semantic_sentry.core.snapshot import Snapshot


@dataclass(frozen=True)
class SeverityCalibration:
    """Calibration result: per-metric noise floor + derived thresholds.

    Attributes:
        thresholds: Dict mapping ``{metric}_{tier}`` to threshold value, where
            ``tier`` is one of ``low/medium/high`` and matches the keys the
            ``Comparison.severity`` property reads.
        source_metric_stats: Per-metric ``{"mean": μ, "std": σ, "n_pairs": N}``.
        n_seeds: Number of reference snapshots used.
        n_sigmas: The sigma multipliers used to derive each tier (one per
            tier, in increasing order of severity).
    """

    thresholds: dict[str, float]
    source_metric_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    n_seeds: int = 0
    n_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)


def calibrate_thresholds(
    reference_snapshots: list[Snapshot],
    metrics: tuple[str, ...] = ("nps", "cka"),
    n_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
    registry: MetricRegistry | None = None,
) -> SeverityCalibration:
    """Derive severity thresholds from multi-seed reference runs.

    All pairs of reference snapshots are compared (``C(N, 2)`` pairs). Each
    metric's mean and std are computed across those pairs. Severity tiers
    are set at ``mu - n_sigmas[tier] * sigma`` for similarity metrics in
    ``[0, 1]`` (the current built-in NPS and CKA both fit this profile —
    higher value means more similar).

    The three thresholds correspond to the
    ``low / medium / high`` tier boundaries that
    ``Comparison.severity`` checks (a comparison crossing the ``high``
    threshold becomes ``CRITICAL``, etc.).

    Args:
        reference_snapshots: At least 2 snapshots from "no real drift" pairs.
        metrics: Names of registered metrics to calibrate. Default: ``("nps", "cka")``.
        n_sigmas: Sigma multipliers for the three tier boundaries (increasing).
        registry: Target registry (defaults to global singleton).

    Returns:
        ``SeverityCalibration`` whose ``thresholds`` dict can be passed
        directly into ``Comparison(thresholds=...)`` or
        ``DriftMonitor.compare(calibration=...)``.

    Raises:
        ValueError: On insufficient snapshots or wrong sigma count.
    """
    if len(reference_snapshots) < 2:
        raise ValueError(
            f"need at least 2 reference snapshots, got {len(reference_snapshots)}"
        )
    if len(n_sigmas) != 3:
        raise ValueError(
            f"n_sigmas must have exactly 3 entries (low/medium/high), got {n_sigmas}"
        )

    registry = registry or get_metric_registry()

    # Compute each requested metric over every C(N, 2) pair.
    per_metric_values: dict[str, list[float]] = {m: [] for m in metrics}
    for s0, s1 in combinations(reference_snapshots, 2):
        if s0.tower_names != s1.tower_names:
            raise ValueError(
                "all reference snapshots must share tower_names"
            )
        Z0 = np.concatenate([s0.get_tower(n) for n in s0.tower_names], axis=1)
        Z1 = np.concatenate([s1.get_tower(n) for n in s1.tower_names], axis=1)
        for m in metrics:
            per_metric_values[m].append(registry.compute(m, Z0, Z1))

    # Build thresholds + stats.
    thresholds: dict[str, float] = {}
    stats: dict[str, dict[str, float]] = {}
    tier_names = ("low", "medium", "high")
    for m, values in per_metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        stats[m] = {"mean": mu, "std": sigma, "n_pairs": float(len(arr))}
        for tier, n_sigma in zip(tier_names, n_sigmas, strict=True):
            thresholds[f"{m}_{tier}"] = mu - n_sigma * sigma

    return SeverityCalibration(
        thresholds=thresholds,
        source_metric_stats=stats,
        n_seeds=len(reference_snapshots),
        n_sigmas=tuple(n_sigmas),
    )
