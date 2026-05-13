"""Local-structure drift metrics (lib_enhancement Family B).

Trustworthiness and continuity decompose NPS into directional halves:

- **Trustworthiness** penalizes *false* neighbors introduced by Z1 — points
  that appear in Z1's top-k for ``i`` but were ranked far away in Z0. Anti-
  aligned drift tends to inflate this (retrieval picks up garbage).
- **Continuity** penalizes *true* neighbors lost by Z1 — points that were in
  Z0's top-k for ``i`` but are now ranked far away. Aligned drift tends to
  inflate this (the model intentionally reshaped its neighborhoods).

The ratio ``trustworthiness / continuity`` is the discriminating signal for
the §3.3 attack on aligned vs anti-aligned regimes — see plan/lib_enhancement.md
items B1 and B2.

Implementation follows Venna & Kaski 2001:

    T = 1 - 2 / (n*k*(2n - 3k - 1)) * sum_i sum_{j in U_i} (rank_in_Z0(j) - k)

where ``U_i`` is the set of points in Z1's top-k for ``i`` but not in Z0's
top-k. Continuity swaps the roles of Z0 and Z1.
"""

from __future__ import annotations

import numpy as np

from semantic_sentry.metrics.nps import _get_knn_indices, _l2_normalize, nps
from semantic_sentry.metrics.registry import MetricRegistry


def _full_rank_matrix(X: np.ndarray) -> np.ndarray:
    """Return the rank (1-based) of every other point as a neighbor of each i.

    ``ranks[i, j]`` is the rank of point ``j`` in ``i``'s neighbor list by
    decreasing cosine similarity; ``ranks[i, i] == 0`` (self is excluded
    from the ranking, treated as rank 0). ``j`` with rank 1 is i's nearest
    non-self neighbor.

    Cost: ``O(n^2 d)`` for similarities + ``O(n^2 log n)`` for argsort.
    """
    n = X.shape[0]
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    order = np.argsort(-sims, axis=1)  # (n, n) — most similar first
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    # ranks[i, i] currently has value n (because argsort put self last after
    # the -inf masking); force it to 0 so it never contributes.
    ranks[np.arange(n), np.arange(n)] = 0
    return ranks


def _directional(
    Z_base: np.ndarray,
    Z_other: np.ndarray,
    k: int,
) -> float:
    """One side of the trustworthiness/continuity pair.

    Penalizes points in ``Z_other``'s top-k that fall outside ``Z_base``'s
    top-k, weighted by their rank in ``Z_base``. Returns a score in [0, 1]
    where 1 means perfect preservation.
    """
    n = Z_base.shape[0]
    if n < k + 2:
        raise ValueError(f"need at least k+2 = {k + 2} samples for k={k}, got {n}")

    Zb = _l2_normalize(Z_base)
    Zo = _l2_normalize(Z_other)

    nn_other = _get_knn_indices(Zo, k)       # (n, k)
    ranks_in_base = _full_rank_matrix(Zb)    # (n, n)
    nn_base = _get_knn_indices(Zb, k)        # (n, k)

    # Build a mask: for each (i, j) with j in nn_other[i], is j NOT in nn_base[i]?
    # nn_base sets per row
    in_base_set = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    in_base_set[rows, nn_base] = True
    # Mask of false neighbors among nn_other
    false_neighbor_mask = ~in_base_set[rows, nn_other]  # (n, k)
    # Penalty for each false neighbor: rank_in_base(j) - k
    penalties = ranks_in_base[rows, nn_other] - k  # (n, k)
    penalties = np.where(false_neighbor_mask, penalties, 0)
    total = float(penalties.sum())

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    score = 1.0 - norm * total
    return float(max(0.0, min(1.0, score)))


def trustworthiness(Z0: np.ndarray, Z1: np.ndarray, k: int = 10) -> float:
    """Trustworthiness — penalizes false neighbors introduced by Z1.

    Score in [0, 1]; 1 = no false neighbors. See module docstring.
    """
    return _directional(Z_base=Z0, Z_other=Z1, k=k)


def continuity(Z0: np.ndarray, Z1: np.ndarray, k: int = 10) -> float:
    """Continuity — penalizes true neighbors lost by Z1.

    Score in [0, 1]; 1 = no lost neighbors. See module docstring.
    """
    return _directional(Z_base=Z1, Z_other=Z0, k=k)


def register_local_structure_metrics(
    registry: MetricRegistry | None = None,
    k: int = 10,
    nps_curve_ks: tuple[int, ...] = (1, 5, 10, 25, 50, 100),
) -> None:
    """Register B1 (trustworthiness), B2 (continuity), and B5 (NPS curve).

    Opt-in so that existing tests that rely only on the three built-ins
    (``cka``, ``nps``, ``isotropy_delta``) keep their stable baseline.
    Call once from user code to enable the v0.2 local-structure family.

    Args:
        registry: Target registry; defaults to the global singleton.
        k: Default ``k`` baked into ``trustworthiness`` and ``continuity``.
        nps_curve_ks: ``k`` values for the multi-k NPS curve (B5).
    """
    registry = registry or MetricRegistry()
    registry.register(
        "trustworthiness",
        trustworthiness,
        range=(0.0, 1.0),
        description="Penalizes false neighbors introduced by Z1 (B1).",
        params={"k": k},
    )
    registry.register(
        "continuity",
        continuity,
        range=(0.0, 1.0),
        description="Penalizes true neighbors lost by Z1 (B2).",
        params={"k": k},
    )
    registry.register_at_k(
        "nps",
        nps,
        ks=list(nps_curve_ks),
        range=(0.0, 1.0),
        description="NPS curve at multiple k values (B5).",
    )
