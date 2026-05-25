"""Torch-native metric implementations (keep-on-device path).

The numpy metrics (`cka.py`, `nps.py`, `isotropy.py`) require embeddings to
be materialised on the host as numpy arrays. `EncoderAdapter.encode_numpy`
forces a ``tensor.cpu().numpy()`` round-trip for that — fine for one-off
snapshot/compare, but wasteful when `DriftMonitor.track()` runs every N steps
inside a training loop on GPU.

These functions operate directly on `torch.Tensor` inputs and never leave the
device they were given. They return Python floats (a single scalar transfer)
so the result drops straight into a `Comparison.global_metrics` dict.

Semantics match the numpy implementations:
    - ``linear_cka_torch`` — centred-Gram HSIC, clamped to [0, 1].
    - ``nps_torch`` — mean top-k neighbour overlap, self excluded.
    - ``isotropy_delta_torch`` — σ_min/σ_max ratio difference.

Numerical parity with the numpy versions is verified in
``tests/unit/test_torch_backend.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias for torch functional


def linear_cka_torch(Z0: torch.Tensor, Z1: torch.Tensor) -> float:
    """Linear CKA between two embedding tensors (stays on device).

    Args:
        Z0: First embedding tensor of shape (n, d0).
        Z1: Second embedding tensor of shape (n, d1), same n as Z0.

    Returns:
        CKA score in [0, 1].
    """
    Z0c = Z0 - Z0.mean(dim=0, keepdim=True)
    Z1c = Z1 - Z1.mean(dim=0, keepdim=True)

    K0 = Z0c @ Z0c.T
    K1 = Z1c @ Z1c.T

    hsic = (K0 * K1).sum()
    norm0 = torch.sqrt((K0 * K0).sum())
    norm1 = torch.sqrt((K1 * K1).sum())

    if norm0 == 0 or norm1 == 0:
        return 0.0

    cka = hsic / (norm0 * norm1)
    return float(cka.clamp(0.0, 1.0).item())


def _knn_indices_torch(Xn: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k neighbour indices by cosine similarity, self excluded.

    Self-exclusion is enforced by masking the diagonal to -inf before the
    top-k selection, which is equivalent to the numpy path's "request k+1,
    drop the self match" approach but cheaper.

    Args:
        Xn: L2-normalised tensor of shape (n, d).
        k: Number of non-self neighbours.

    Returns:
        Long tensor of shape (n, k) with neighbour indices.
    """
    sims = Xn @ Xn.T
    sims.fill_diagonal_(float("-inf"))
    return torch.topk(sims, k, dim=1).indices


def nps_torch(Z0: torch.Tensor, Z1: torch.Tensor, k: int = 10) -> float:
    """Neighborhood Preservation Score between two tensors (stays on device).

    Args:
        Z0: First embedding tensor of shape (n, d0).
        Z1: Second embedding tensor of shape (n, d1), same n as Z0.
        k: Number of neighbours to consider.

    Returns:
        Mean per-point top-k neighbour overlap in [0, 1]. Degenerate inputs
        (n <= k + 1) return 1.0, matching ``metrics.nps.nps_per_point``.
    """
    n = Z0.shape[0]
    if n <= k + 1:
        return 1.0

    Z0n = F.normalize(Z0, p=2, dim=1)
    Z1n = F.normalize(Z1, p=2, dim=1)

    nbr0 = _knn_indices_torch(Z0n, k)  # (n, k)
    nbr1 = _knn_indices_torch(Z1n, k)  # (n, k)

    # Per-row set intersection: (n, k, 1) == (n, 1, k) -> any over last dim.
    matches = (nbr0.unsqueeze(2) == nbr1.unsqueeze(1)).any(dim=2)  # (n, k)
    overlap = matches.sum(dim=1).to(torch.float32) / k
    return float(overlap.mean().item())


def _isotropy_torch(Z: torch.Tensor) -> float:
    """σ_min / σ_max of the centred embedding tensor."""
    Zc = Z - Z.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(Zc)
    except RuntimeError:
        return 0.0
    if S[0] == 0:
        return 0.0
    return float((S[-1] / S[0]).item())


def isotropy_delta_torch(Z0: torch.Tensor, Z1: torch.Tensor) -> float:
    """Change in isotropy (σ_min/σ_max ratio) between two tensors."""
    return _isotropy_torch(Z1) - _isotropy_torch(Z0)


def compute_drift_metrics_torch(
    Z0: torch.Tensor,
    Z1: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    """Compute the three built-in metrics on-device, mirroring the registry.

    Returns a dict with the same keys the numpy ``MetricRegistry`` produces
    for the built-ins: ``cka``, ``nps``, ``isotropy_delta``.
    """
    return {
        "cka": linear_cka_torch(Z0, Z1),
        "nps": nps_torch(Z0, Z1, k=k),
        "isotropy_delta": isotropy_delta_torch(Z0, Z1),
    }
