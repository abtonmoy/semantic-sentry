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

    # Get k-nearest neighbors in each space (excluding self)
    neighbors_0 = _get_knn_indices(Z0_norm, k + 1)
    neighbors_1 = _get_knn_indices(Z1_norm, k + 1)

    # Compute overlap for each point
    overlaps = np.zeros(n)
    for i in range(n):
        # Get k neighbors (skip first which is self)
        nn0 = set(neighbors_0[i, 1:k+1])
        nn1 = set(neighbors_1[i, 1:k+1])

        # Jaccard-like overlap: |intersection| / k
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
    """Get k-nearest neighbor indices using FAISS or numpy.
    
    Args:
        X: L2-normalized matrix of shape (n, d)
        k: Number of neighbors to find
        
    Returns:
        Array of shape (n, k) with neighbor indices
    """
    n = X.shape[0]

    if FAISS_AVAILABLE and n > 1000:
        # Use FAISS for larger datasets
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product index
        index.add(X.astype(np.float32))

        # Search k nearest neighbors
        _, indices = index.search(X.astype(np.float32), k)
        return indices
    else:
        # Use numpy for smaller datasets
        # Compute pairwise dot products (cosine similarity since X is normalized)
        similarities = X @ X.T

        # Get top k indices (excluding self which has highest similarity)
        indices = np.argsort(-similarities, axis=1)[:, :k]
        return indices


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
