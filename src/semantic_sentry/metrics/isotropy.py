"""Isotropy metrics for embedding spaces."""

import numpy as np


def _validate_input(Z: np.ndarray) -> None:
    """Validate input matrix for NaN/Inf values.

    Args:
        Z: Embedding matrix

    Raises:
        ValueError: If input has invalid values
    """
    if np.any(np.isnan(Z)):
        raise ValueError("Input matrix contains NaN values")

    if np.any(np.isinf(Z)):
        raise ValueError("Input matrix contains Inf values")


def isotropy(Z: np.ndarray) -> float:
    """Compute isotropy of an embedding space.

    Isotropy measures how uniformly distributed an embedding space is
    across all dimensions. A perfectly isotropic space has singular
    values that are all equal. Anisotropic spaces have dominant directions.

    Algorithm:
        1. Center Z
        2. Compute SVD: U, S, Vt = svd(Z_centered, full_matrices=False)
        3. Return S[-1] / S[0] (smallest / largest singular value)

    Args:
        Z: Embedding matrix of shape (n, d)

    Returns:
        Isotropy score in [0, 1], where 1 means perfectly isotropic

    Raises:
        ValueError: If input contains NaN/Inf values
    """
    _validate_input(Z)

    # Center the embeddings
    Z_centered = Z - Z.mean(axis=0, keepdims=True)

    # Compute SVD
    try:
        _, S, _ = np.linalg.svd(Z_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD failed, return 0 (anisotropic)
        return 0.0

    if S[0] == 0:
        return 0.0

    # Return ratio of smallest to largest singular value
    return float(S[-1] / S[0])


def isotropy_delta(Z0: np.ndarray, Z1: np.ndarray) -> float:
    """Compute change in isotropy between two embedding spaces.
    
    Args:
        Z0: First embedding matrix of shape (n, d0)
        Z1: Second embedding matrix of shape (n, d1)
        
    Returns:
        Delta isotropy in [-1, 1], where:
            - Positive: Z1 is more isotropic than Z0
            - Negative: Z1 is less isotropic than Z0
            - Zero: Same isotropy
    """
    iso0 = isotropy(Z0)
    iso1 = isotropy(Z1)

    return iso1 - iso0


def effective_dimensionality(Z: np.ndarray, threshold: float = 0.9) -> int:
    """Compute effective dimensionality using participation ratio.
    
    Args:
        Z: Embedding matrix of shape (n, d)
        threshold: Variance threshold for dimensionality calculation
        
    Returns:
        Effective dimensionality (integer)
    """
    # Center and compute covariance
    Z_centered = Z - Z.mean(axis=0, keepdims=True)

    # Handle case where n < d (use SVD instead of eig)
    if Z.shape[0] < Z.shape[1]:
        _, S, _ = np.linalg.svd(Z_centered, full_matrices=False)
        eigenvalues = S ** 2
    else:
        cov = Z_centered.T @ Z_centered / Z.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[::-1]  # Sort descending

    # Compute cumulative explained variance
    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        return 0

    cumulative_variance = np.cumsum(eigenvalues) / total_variance

    # Find number of dimensions needed to explain threshold variance
    effective_dim = np.searchsorted(cumulative_variance, threshold) + 1

    return int(effective_dim)
