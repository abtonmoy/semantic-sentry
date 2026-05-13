"""Neighborhood Preservation Score (NPS) metric."""


import numpy as np

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def nps(Z0: np.ndarray, Z1: np.ndarray, k: int = 10) -> float:
    """Compute Neighborhood Preservation Score between two embedding spaces.
    
    NPS measures how well the local neighborhood structure is preserved
    between two representations. For each point, it computes the overlap
    between its k-nearest neighbors in both spaces.
    
    Algorithm:
        1. L2-normalize both matrices
        2. Build FAISS IndexFlatIP for both
        3. Search top-(k+1) neighbors (exclude self)
        4. Compute per-point overlap fraction
        5. Return mean overlap
    
    Args:
        Z0: First embedding matrix of shape (n, d0)
        Z1: Second embedding matrix of shape (n, d1)
        k: Number of neighbors to consider (default: 10)
        
    Returns:
        NPS score in [0, 1], where 1 means perfect neighborhood preservation
        
    Raises:
        ValueError: If input matrices have different numbers of samples
    """
    per_point_scores = nps_per_point(Z0, Z1, k)
    return float(np.mean(per_point_scores))


def _validate_input(Z0: np.ndarray, Z1: np.ndarray) -> None:
    """Validate input matrices for NaN/Inf values.

    Args:
        Z0: First embedding matrix
        Z1: Second embedding matrix

    Raises:
        ValueError: If inputs have invalid values
    """
    if np.any(np.isnan(Z0)) or np.any(np.isnan(Z1)):
        raise ValueError("Input matrices contain NaN values")

    if np.any(np.isinf(Z0)) or np.any(np.isinf(Z1)):
        raise ValueError("Input matrices contain Inf values")


def nps_per_point(Z0: np.ndarray, Z1: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute per-point Neighborhood Preservation Scores.

    Args:
        Z0: First embedding matrix of shape (n, d0)
        Z1: Second embedding matrix of shape (n, d1)
        k: Number of neighbors to consider (default: 10)

    Returns:
        Array of per-point NPS scores of shape (n,)

    Raises:
        ValueError: If input matrices have different numbers of samples,
                    or contain NaN/Inf values
    """
    if Z0.shape[0] != Z1.shape[0]:
        raise ValueError(
            f"Input matrices must have same number of samples, got {Z0.shape[0]} and {Z1.shape[0]}"
        )

    _validate_input(Z0, Z1)

    n = Z0.shape[0]
    if n <= k + 1:
        # Not enough points for meaningful k-NN
        return np.ones(n)

    # L2-normalize both matrices (for cosine similarity via inner product)
    Z0_norm = _l2_normalize(Z0)
    Z1_norm = _l2_normalize(Z1)

    # Get k-nearest neighbors in each space — self already excluded by the helper.
    neighbors_0 = _get_knn_indices(Z0_norm, k)
    neighbors_1 = _get_knn_indices(Z1_norm, k)

    # Compute overlap for each point
    overlaps = np.zeros(n)
    for i in range(n):
        nn0 = set(neighbors_0[i])
        nn1 = set(neighbors_1[i])
        overlap_size = len(nn0 & nn1)
        overlaps[i] = overlap_size / k

    return overlaps


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2 normalize rows of matrix.
    
    Args:
        X: Matrix of shape (n, d)
        
    Returns:
        L2-normalized matrix
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return X / norms


def _get_knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Get k-nearest neighbor indices using FAISS or numpy, excluding self.

    The returned array has shape (n, k) and contains, for each row index i,
    the indices of i's k nearest neighbours by cosine similarity, EXCLUDING
    i itself. Both the FAISS and the numpy branches enforce self-exclusion
    deterministically by requesting k+1 neighbours, then dropping the
    self-match position from each row.

    Args:
        X: L2-normalized matrix of shape (n, d)
        k: Number of (non-self) neighbours to find

    Returns:
        Array of shape (n, k) with neighbour indices (self never present).
    """
    n = X.shape[0]
    k_plus_1 = min(k + 1, n)

    if FAISS_AVAILABLE and n > 1000:
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product index
        index.add(X.astype(np.float32))
        if n > 50000:
            batch_size = 10000
            chunks = []
            for i in range(0, n, batch_size):
                batch = X[i:min(i + batch_size, n)].astype(np.float32)
                _, idx = index.search(batch, k_plus_1)
                chunks.append(idx)
            raw = np.vstack(chunks)
        else:
            _, raw = index.search(X.astype(np.float32), k_plus_1)
    else:
        # Pairwise dot products (cosine sim since X is L2-normalised)
        similarities = X @ X.T
        raw = np.argsort(-similarities, axis=1)[:, :k_plus_1]

    return _drop_self_column(raw, k)


def _drop_self_column(raw: np.ndarray, k: int) -> np.ndarray:
    """Drop the self-match from a (n, k+1) neighbour matrix.

    For row i, the self-match is the column where the value equals i. In the
    overwhelming majority of cases this is column 0 (a normalised vector's
    nearest neighbour is itself), but ties at unit cosine similarity could
    push it elsewhere — so we locate and remove it row-wise rather than
    blindly slicing [:, 1:].
    """
    n, kp1 = raw.shape
    self_idx = np.arange(n)[:, None]
    is_self = raw == self_idx                              # (n, k+1)
    self_pos = is_self.argmax(axis=1)                       # column of self
    has_self = is_self.any(axis=1)                          # bool per row
    # Default: drop column 0 (self was first); if self-match found elsewhere,
    # drop that column instead.
    keep = np.ones_like(raw, dtype=bool)
    drop_col = np.where(has_self, self_pos, 0)
    keep[np.arange(n), drop_col] = False
    out = raw[keep].reshape(n, kp1 - 1)
    # Caller asked for k neighbours; if the raw matrix had only k columns to
    # begin with (n == k + 1 case is excluded upstream, but be defensive),
    # truncate.
    return out[:, :k]


def nps_bounds(nps_score: float) -> tuple[float, float]:
    """Return theoretical bounds on downstream task degradation.
    
    Args:
        nps_score: The NPS score
        
    Returns:
        Tuple of (lower_bound, upper_bound) on degradation
    """
    delta = 1.0 - nps_score
    lower = max(0.0, delta)
    # Upper bound is looser
    upper = min(1.0, 2.0 * delta)
    return lower, upper
