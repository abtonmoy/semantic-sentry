"""Fixtures for stress tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Return a deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def make_embeddings(rng):
    """Factory fixture: generate L2-normalized embeddings of any shape."""
    def _make(n, d, seed=None):
        r = np.random.default_rng(seed or 42)
        Z = r.standard_normal((n, d)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return Z
    return _make


@pytest.fixture
def make_drifted_pair(make_embeddings):
    """Factory: generate base + drifted embedding pair with controlled noise."""
    def _make(n, d, noise_scale=0.3, seed=42):
        Z_base = make_embeddings(n, d, seed=seed)
        rng = np.random.default_rng(seed + 1)
        noise = rng.standard_normal((n, d)).astype(np.float32) * noise_scale
        Z_drifted = Z_base + noise
        Z_drifted = Z_drifted / np.linalg.norm(Z_drifted, axis=1, keepdims=True)
        return Z_base, Z_drifted
    return _make
