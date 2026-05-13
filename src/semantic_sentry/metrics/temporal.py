"""Temporal layer (lib_enhancement Family F).

These are *meta-features*: they wrap any registered scalar metric to produce
per-checkpoint time-series signals over a sequence of snapshots.

They live in a separate module-level registry rather than ``MetricRegistry``
because the function signature is fundamentally different —
``(snapshots, times, metric_name) -> np.ndarray`` instead of
``(Z0, Z1) -> float`` — and overloading the existing registry would either
break the determinism check or leak temporal-only logic into single-pair
compute paths.

F1 velocity:     central-difference dM/dt over consecutive snapshot pairs.
F2 acceleration: second finite difference d²M/dt².
F3 plateau:      boolean signal — |velocity| < eps AND |acceleration| < delta
                 for k consecutive checkpoints. The highest-ROI library
                 addition per lib_enhancement.md (early-stopping signal).
G4 register_with_temporal: convenience wiring — register one pairwise metric
                 and auto-register its velocity/acceleration alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from semantic_sentry.metrics.registry import MetricRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from semantic_sentry.core.snapshot import Snapshot

_TEMPORAL: dict[str, Callable] = {}


def register_temporal(name: str, fn: Callable) -> None:
    """Register a temporal (sequence-level) metric.

    Args:
        name: Unique name.
        fn: Callable accepting ``(snapshots, times)`` and returning an
            ``np.ndarray``.
    """
    _TEMPORAL[name] = fn


def list_temporal() -> list[str]:
    return list(_TEMPORAL.keys())


def get_temporal(name: str) -> Callable:
    return _TEMPORAL[name]


def reset_temporal() -> None:
    _TEMPORAL.clear()


def _pairwise_metric_values(
    snapshots: list[Snapshot],
    metric_name: str,
    registry: MetricRegistry | None = None,
) -> np.ndarray:
    """Compute ``metric_name`` between every consecutive snapshot pair.

    Returns an array of shape ``(T-1,)`` where ``T = len(snapshots)``. Each
    value is the pairwise metric between ``snapshots[i]`` and
    ``snapshots[i+1]``.
    """
    registry = registry or MetricRegistry()
    T = len(snapshots)
    if T < 2:
        raise ValueError(f"need at least 2 snapshots, got {T}")
    vals = np.empty(T - 1, dtype=np.float64)
    for i in range(T - 1):
        s0, s1 = snapshots[i], snapshots[i + 1]
        if s0.tower_names != s1.tower_names:
            raise ValueError(
                f"snapshot {i} and {i + 1} have different tower_names"
            )
        Z0 = np.concatenate(
            [s0.get_tower(n) for n in s0.tower_names], axis=1
        )
        Z1 = np.concatenate(
            [s1.get_tower(n) for n in s1.tower_names], axis=1
        )
        vals[i] = registry.compute(metric_name, Z0, Z1)
    return vals


def _central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central difference of ``y(t)`` at each sample. Endpoints use forward
    and backward differences. ``y`` and ``t`` must have the same shape.
    """
    n = y.shape[0]
    out = np.zeros_like(y)
    if n < 2:
        return out
    out[0] = (y[1] - y[0]) / (t[1] - t[0]) if t[1] != t[0] else 0.0
    out[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2]) if t[-1] != t[-2] else 0.0
    for i in range(1, n - 1):
        dt = t[i + 1] - t[i - 1]
        out[i] = (y[i + 1] - y[i - 1]) / dt if dt != 0 else 0.0
    return out


def velocity(
    snapshots: list[Snapshot],
    times: list[float],
    metric_name: str,
    registry: MetricRegistry | None = None,
) -> np.ndarray:
    """F1 — rate of change of ``metric_name`` along the trajectory.

    Pairwise metric values ``m_i = M(snapshots[i], snapshots[i+1])`` are
    computed at the midpoint between consecutive checkpoints; velocity is
    their central-difference derivative w.r.t. time. For a constant
    trajectory (zero drift between every consecutive pair), velocity is
    zero everywhere — which is the signal the plateau detector relies on.

    Returns an array of shape ``(T,)`` where ``T = len(snapshots)``. The
    per-snapshot values are computed by central-differencing ``m`` at the
    boundary between intervals (interior) and forward/backward differencing
    at the endpoints. With only 2 snapshots, the result is zero everywhere
    (there is only one ``m`` value, so no rate of change).
    """
    T = len(snapshots)
    if len(times) != T:
        raise ValueError(f"snapshots and times length mismatch: {T} vs {len(times)}")
    if T < 2:
        raise ValueError(f"need at least 2 snapshots, got {T}")
    m = _pairwise_metric_values(snapshots, metric_name, registry)
    t = np.asarray(times, dtype=np.float64)

    v = np.zeros(T, dtype=np.float64)
    if T < 3:
        return v  # only one pairwise metric; no rate of change defined
    # Interior: v[i] = (m[i] - m[i-1]) / (t[i+1] - t[i-1])
    for i in range(1, T - 1):
        dt = t[i + 1] - t[i - 1]
        v[i] = (m[i] - m[i - 1]) / dt if dt != 0 else 0.0
    # Endpoints — match the second/second-to-last interior values.
    v[0] = v[1]
    v[T - 1] = v[T - 2]
    return v


def acceleration(
    snapshots: list[Snapshot],
    times: list[float],
    metric_name: str,
    registry: MetricRegistry | None = None,
) -> np.ndarray:
    """F2 — central-difference acceleration d²M/dt². Shape ``(T,)``."""
    v = velocity(snapshots, times, metric_name, registry)
    t = np.asarray(times, dtype=np.float64)
    return _central_diff(v, t)


def plateau(
    snapshots: list[Snapshot],
    times: list[float],
    metric_name: str,
    eps: float = 0.005,
    delta: float = 0.001,
    k: int = 3,
    registry: MetricRegistry | None = None,
) -> np.ndarray:
    """F3 — plateau detector. ``True`` where ``|velocity| < eps`` and
    ``|acceleration| < delta`` for ``k`` consecutive checkpoints.

    Returns a boolean array of shape ``(T,)``.
    """
    v = velocity(snapshots, times, metric_name, registry)
    a = acceleration(snapshots, times, metric_name, registry)
    candidate = (np.abs(v) < eps) & (np.abs(a) < delta)
    T = len(candidate)
    out = np.zeros(T, dtype=bool)
    if k <= 1:
        return candidate
    # Rolling AND: an index i is in plateau if all candidate[i-k+1:i+1] are True.
    for i in range(k - 1, T):
        if candidate[i - k + 1 : i + 1].all():
            out[i] = True
    return out


def register_with_temporal(
    name: str,
    fn: Callable,
    range: tuple[float, float] | None = None,
    description: str | None = None,
    params: dict | None = None,
    registry: MetricRegistry | None = None,
) -> None:
    """G4 — register a pairwise metric and auto-wire its velocity/acceleration
    temporal wrappers.

    Equivalent to::

        registry.register(name, fn, ...)
        register_temporal(f"{name}_velocity",
                          lambda s, t: velocity(s, t, name))
        register_temporal(f"{name}_acceleration",
                          lambda s, t: acceleration(s, t, name))
    """
    registry = registry or MetricRegistry()
    registry.register(name, fn, range=range, description=description, params=params)
    register_temporal(
        f"{name}_velocity",
        lambda snapshots, times: velocity(snapshots, times, name, registry),
    )
    register_temporal(
        f"{name}_acceleration",
        lambda snapshots, times: acceleration(snapshots, times, name, registry),
    )
