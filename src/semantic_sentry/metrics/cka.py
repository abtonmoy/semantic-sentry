"""Linear Centered Kernel Alignment (CKA) metric."""


import numpy as np


def _validate_input(Z0: np.ndarray, Z1: np.ndarray) -> None:
    """Validate input matrices for NaN/Inf values and shape compatibility.

    Args:
        Z0: First embedding matrix
        Z1: Second embedding matrix

    Raises:
        ValueError: If inputs have invalid values or incompatible shapes
    """
    if Z0.shape[0] != Z1.shape[0]:
        raise ValueError(
            f"Input matrices must have same number of samples, got {Z0.shape[0]} and {Z1.shape[0]}"
        )

    if np.any(np.isnan(Z0)) or np.any(np.isnan(Z1)):
        raise ValueError("Input matrices contain NaN values")

    if np.any(np.isinf(Z0)) or np.any(np.isinf(Z1)):
        raise ValueError("Input matrices contain Inf values")


def linear_cka(Z0: np.ndarray, Z1: np.ndarray) -> float:
    """Compute Linear CKA between two embedding matrices.

    CKA measures the similarity between two representations by computing
    the normalized Hilbert-Schmidt Independence Criterion (HSIC).

    Algorithm:
        1. Center both matrices: Z = Z - Z.mean(axis=0)
        2. Compute Gram matrices: K0 = Z0 @ Z0.T, K1 = Z1 @ Z1.T
        3. HSIC: hsic_01 = (K0 * K1).sum()
        4. Normalize: cka = hsic_01 / sqrt((K0*K0).sum() * (K1*K1).sum())

    Args:
        Z0: First embedding matrix of shape (n, d0)
        Z1: Second embedding matrix of shape (n, d1)

    Returns:
        CKA score in [0, 1], where 1 means identical representations

    Raises:
        ValueError: If input matrices have different numbers of samples,
                    or contain NaN/Inf values
    """
    _validate_input(Z0, Z1)

    # Center both matrices
    Z0_centered = Z0 - Z0.mean(axis=0, keepdims=True)
    Z1_centered = Z1 - Z1.mean(axis=0, keepdims=True)

    # Compute Gram matrices
    K0 = Z0_centered @ Z0_centered.T
    K1 = Z1_centered @ Z1_centered.T

    # Compute HSIC
    hsic_01 = np.sum(K0 * K1)

    # Normalize
    norm_K0 = np.sqrt(np.sum(K0 * K0))
    norm_K1 = np.sqrt(np.sum(K1 * K1))

    if norm_K0 == 0 or norm_K1 == 0:
        return 0.0

    cka = hsic_01 / (norm_K0 * norm_K1)

    # Clamp to [0, 1] to handle numerical errors
    return float(np.clip(cka, 0.0, 1.0))


def center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix.

    Args:
        K: Gram matrix of shape (n, n)

    Returns:
        Centered Gram matrix
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H
