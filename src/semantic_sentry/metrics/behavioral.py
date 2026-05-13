"""Behavioral / ranking-stability drift metrics (lib_enhancement Family A).

These metrics measure how the model's *scoring behavior* on (query, document)
pairs shifts between two checkpoints — not how individual neighborhoods
shift. This is the family with a real shot at discriminating aligned vs
anti-aligned drift regimes, because anti-aligned drift can preserve local
neighborhoods while destroying rankings, and aligned drift can reshape local
neighborhoods while preserving rankings.

All behavioral metrics take FOUR embedding matrices — Q-side baseline,
Q-side updated, D-side baseline, D-side updated — because they evaluate
``cos(Z_M(q), Z_M(d))`` for cross-side pairs. They live in a separate
``BehavioralMetricRegistry`` rather than ``MetricRegistry`` because the
``(Z0, Z1) -> float`` contract of the latter cannot accommodate the extra
inputs (see lib_enhancement A1-A5 spec).

Use ``DriftMonitor.compare(..., behavioral=anchor_D, model_v0=..., model_v1=...)``
to compute them end-to-end; or call ``compute_all`` on the registry directly
with pre-computed Q/D embeddings.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau

from semantic_sentry.exceptions import MetricRegistrationError
from semantic_sentry.metrics.nps import _l2_normalize

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BehavioralMetricEntry:
    fn: Callable[..., float]
    range: tuple[float, float] | None
    description: str | None
    params: dict[str, Any] = field(default_factory=dict)


class BehavioralMetricRegistry:
    """Singleton registry for behavioral (4-input) drift metrics."""

    _instance: BehavioralMetricRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> BehavioralMetricRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._metrics: dict[str, BehavioralMetricEntry] = {}

    def reset(self) -> None:
        with self._lock:
            self._metrics = {}

    def register(
        self,
        name: str,
        fn: Callable[..., float],
        range: tuple[float, float] | None = None,
        description: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        bound = dict(params or {})
        # Smoke-test the metric on small random inputs to catch signature
        # bugs at registration time; same spirit as MetricRegistry's
        # determinism check but with 4-input shape.
        try:
            self._validate(fn, bound)
        except MetricRegistrationError:
            raise
        with self._lock:
            self._metrics[name] = BehavioralMetricEntry(
                fn=fn, range=range, description=description, params=bound
            )

    def register_at_k(
        self,
        base_name: str,
        fn: Callable[..., float],
        ks: list[int],
        range: tuple[float, float] | None = None,
        description: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        extras = dict(extra_params or {})
        for k in ks:
            self.register(
                f"{base_name}_at_{k}",
                fn,
                range=range,
                description=description,
                params={**extras, "k": k},
            )

    def _validate(self, fn: Callable[..., float], params: dict[str, Any]) -> None:
        rng = np.random.default_rng(42)
        Z0_Q = _l2_normalize(rng.standard_normal((6, 8)).astype(np.float32))
        Z1_Q = _l2_normalize(rng.standard_normal((6, 8)).astype(np.float32))
        D0 = _l2_normalize(rng.standard_normal((10, 8)).astype(np.float32))
        D1 = _l2_normalize(rng.standard_normal((10, 8)).astype(np.float32))
        try:
            r1 = fn(Z0_Q, Z1_Q, D0=D0, D1=D1, **params)
            r2 = fn(Z0_Q, Z1_Q, D0=D0, D1=D1, **params)
        except Exception as e:
            raise MetricRegistrationError(
                f"Behavioral metric raised at validation: {e}"
            ) from e
        if r1 != r2:
            raise MetricRegistrationError(
                f"Behavioral metric not deterministic: {r1} != {r2}"
            )
        if not isinstance(r1, (float, np.floating)):
            raise MetricRegistrationError(
                f"Behavioral metric must return float, got {type(r1)}"
            )

    def list_metrics(self) -> list[str]:
        with self._lock:
            return list(self._metrics.keys())

    def get_info(self, name: str) -> BehavioralMetricEntry:
        with self._lock:
            return self._metrics[name]

    def unregister(self, name: str) -> None:
        with self._lock:
            del self._metrics[name]

    def compute(
        self,
        name: str,
        Z0_Q: np.ndarray,
        Z1_Q: np.ndarray,
        D0: np.ndarray,
        D1: np.ndarray,
    ) -> float:
        with self._lock:
            entry = self._metrics[name]
        result = entry.fn(Z0_Q, Z1_Q, D0=D0, D1=D1, **entry.params)
        if entry.range is not None:
            mn, mx = entry.range
            if not (mn <= result <= mx):
                raise ValueError(
                    f"Behavioral metric '{name}' returned {result}, "
                    f"outside range [{mn}, {mx}]"
                )
        return float(result)

    def compute_all(
        self,
        Z0_Q: np.ndarray,
        Z1_Q: np.ndarray,
        D0: np.ndarray,
        D1: np.ndarray,
        metric_names: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, float]:
        with self._lock:
            names = metric_names or list(self._metrics.keys())
            entries = {n: self._metrics[n] for n in names}
        if not names:
            return {}

        def _run(name: str) -> tuple[str, float]:
            e = entries[name]
            return name, float(e.fn(Z0_Q, Z1_Q, D0=D0, D1=D1, **e.params))

        if not parallel or len(names) == 1:
            return dict(_run(n) for n in names)
        with ThreadPoolExecutor(max_workers=min(len(names), 4)) as ex:
            return dict(ex.map(_run, names))


def get_behavioral_registry() -> BehavioralMetricRegistry:
    return BehavioralMetricRegistry()


# ---------------------------------------------------------------------------
# Metric functions — A1..A5
# ---------------------------------------------------------------------------


def _scores(Z_Q: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix of shape (|Q|, |D|). Assumes L2-normalized."""
    return Z_Q @ D.T


def score_distribution_jsd(
    Z0_Q: np.ndarray,
    Z1_Q: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
    n_bins: int = 100,
) -> float:
    """A1 — Jensen–Shannon divergence between baseline and updated score
    distributions (cosine sims of every Q-D pair).

    Result is bounded in [0, 1] (using base-2 JSD, squared to convert distance
    to divergence). 0 = identical distributions.
    """
    Z0_Q = _l2_normalize(Z0_Q)
    Z1_Q = _l2_normalize(Z1_Q)
    D0 = _l2_normalize(D0)
    D1 = _l2_normalize(D1)

    S0 = _scores(Z0_Q, D0).ravel()
    S1 = _scores(Z1_Q, D1).ravel()
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    p, _ = np.histogram(S0, bins=edges)
    q, _ = np.histogram(S1, bins=edges)
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p /= p_sum
    q /= q_sum
    # `jensenshannon` returns the JS *distance*; square to get the divergence,
    # which is bounded in [0, 1] with base=2.
    dist = float(jensenshannon(p, q, base=2))
    if np.isnan(dist):
        return 0.0
    return float(min(1.0, max(0.0, dist * dist)))


def mean_abs_score_delta(
    Z0_Q: np.ndarray,
    Z1_Q: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
) -> float:
    """A2 — Mean absolute pointwise score change across all Q-D pairs.

    Bounded in [0, 2] (cosine sims in [-1, 1]).
    """
    Z0_Q = _l2_normalize(Z0_Q)
    Z1_Q = _l2_normalize(Z1_Q)
    D0 = _l2_normalize(D0)
    D1 = _l2_normalize(D1)
    S0 = _scores(Z0_Q, D0)
    S1 = _scores(Z1_Q, D1)
    return float(np.mean(np.abs(S0 - S1)))


def _rbo_pair(rank0: np.ndarray, rank1: np.ndarray, p: float) -> float:
    """RBO between two full ranked lists of the same documents (indices)."""
    d = rank0.shape[0]
    if d == 0:
        return 1.0
    # Convert rank lists to position-of-doc mappings for O(1) membership.
    seen0: set[int] = set()
    seen1: set[int] = set()
    overlap = 0
    total = 0.0
    weight_sum = 0.0
    for depth in range(1, d + 1):
        seen0.add(int(rank0[depth - 1]))
        seen1.add(int(rank1[depth - 1]))
        # |X_d ∩ Y_d| — but adding 0/1/2 new shared elements each step.
        # Recompute via intersection size.
        overlap = len(seen0 & seen1)
        w = p ** (depth - 1)
        total += w * overlap / depth
        weight_sum += w
    return float((1.0 - p) * total / ((1.0 - p) * weight_sum)) if weight_sum > 0 else 0.0


def per_query_rbo(
    Z0_Q: np.ndarray,
    Z1_Q: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
    p: float = 0.9,
) -> float:
    """A3 — Mean Rank-Biased Overlap across queries.

    Top-weighted ranking metric. Bounded in [0, 1].
    """
    Z0_Q = _l2_normalize(Z0_Q)
    Z1_Q = _l2_normalize(Z1_Q)
    D0 = _l2_normalize(D0)
    D1 = _l2_normalize(D1)
    S0 = _scores(Z0_Q, D0)
    S1 = _scores(Z1_Q, D1)
    rank0 = np.argsort(-S0, axis=1)
    rank1 = np.argsort(-S1, axis=1)
    scores = [_rbo_pair(rank0[q], rank1[q], p) for q in range(rank0.shape[0])]
    return float(np.mean(scores))


def per_query_kendall_tau(
    Z0_Q: np.ndarray,
    Z1_Q: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
) -> float:
    """A4 — Mean Kendall tau across queries on the full D-side ranking.

    Bounded in [-1, 1].
    """
    Z0_Q = _l2_normalize(Z0_Q)
    Z1_Q = _l2_normalize(Z1_Q)
    D0 = _l2_normalize(D0)
    D1 = _l2_normalize(D1)
    S0 = _scores(Z0_Q, D0)
    S1 = _scores(Z1_Q, D1)
    taus = []
    for q in range(S0.shape[0]):
        tau = kendalltau(S0[q], S1[q]).statistic
        if np.isnan(tau):
            tau = 0.0
        taus.append(float(tau))
    return float(np.mean(taus))


def self_retrieval_topk(
    Z0_Q: np.ndarray,
    Z1_Q: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
    k: int = 10,
) -> float:
    """A5 — Fraction of queries whose top-k document set is unchanged.

    Bounded in [0, 1]. Coarser than RBO (set equality, no rank weighting)
    but the most interpretable behavioral metric for non-ML stakeholders.
    """
    Z0_Q = _l2_normalize(Z0_Q)
    Z1_Q = _l2_normalize(Z1_Q)
    D0 = _l2_normalize(D0)
    D1 = _l2_normalize(D1)
    S0 = _scores(Z0_Q, D0)
    S1 = _scores(Z1_Q, D1)
    if D0.shape[0] < k:
        raise ValueError(
            f"need at least k={k} documents, got {D0.shape[0]}"
        )
    top0 = np.argpartition(-S0, kth=k - 1, axis=1)[:, :k]
    top1 = np.argpartition(-S1, kth=k - 1, axis=1)[:, :k]
    matches = 0
    for q in range(top0.shape[0]):
        if set(top0[q].tolist()) == set(top1[q].tolist()):
            matches += 1
    return float(matches / top0.shape[0])


def register_behavioral_metrics(
    registry: BehavioralMetricRegistry | None = None,
    n_bins: int = 100,
    rbo_p: float = 0.9,
    self_retrieval_ks: tuple[int, ...] = (1, 5, 10),
) -> None:
    """Register A1-A5 on the behavioral registry."""
    registry = registry or BehavioralMetricRegistry()
    registry.register(
        "score_distribution_jsd",
        score_distribution_jsd,
        range=(0.0, 1.0),
        description="JSD between baseline and updated cosine-sim distributions (A1).",
        params={"n_bins": n_bins},
    )
    registry.register(
        "mean_abs_score_delta",
        mean_abs_score_delta,
        range=(0.0, 2.0),
        description="Mean |Δ score| across Q-D pairs (A2).",
    )
    registry.register(
        "per_query_rbo",
        per_query_rbo,
        range=(0.0, 1.0),
        description="Mean RBO across queries (A3).",
        params={"p": rbo_p},
    )
    registry.register(
        "per_query_kendall_tau",
        per_query_kendall_tau,
        range=(-1.0, 1.0),
        description="Mean Kendall tau across queries (A4).",
    )
    registry.register_at_k(
        "self_retrieval",
        self_retrieval_topk,
        ks=list(self_retrieval_ks),
        range=(0.0, 1.0),
        description="Top-k self-retrieval consistency (A5).",
    )
