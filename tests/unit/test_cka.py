"""Tests for Linear CKA metric."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from semantic_sentry.metrics.cka import linear_cka


class TestLinearCKA:
    """Test Linear CKA properties."""
    
    def test_identity_property(self):
        """CKA(Z, Z) == 1.0 for any Z."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        cka = linear_cka(Z, Z)
        assert abs(cka - 1.0) < 1e-5, f"Expected 1.0, got {cka}"
    
    def test_symmetry_property(self):
        """CKA(Z0, Z1) == CKA(Z1, Z0)."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 48).astype(np.float32)
        
        cka_01 = linear_cka(Z0, Z1)
        cka_10 = linear_cka(Z1, Z0)
        
        assert abs(cka_01 - cka_10) < 1e-5, f"Asymmetric: {cka_01} vs {cka_10}"
    
    def test_range_property(self):
        """0.0 <= CKA(Z0, Z1) <= 1.0."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 48).astype(np.float32)
        
        cka = linear_cka(Z0, Z1)
        
        assert 0.0 <= cka <= 1.0, f"CKA out of range: {cka}"
    
    def test_orthogonal_invariance(self):
        """CKA is invariant to orthogonal transformations."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        
        # Create random orthogonal matrix via QR decomposition
        Q, _ = np.linalg.qr(np.random.randn(64, 64).astype(np.float32))
        Z_transformed = Z @ Q
        
        cka = linear_cka(Z, Z_transformed)
        assert abs(cka - 1.0) < 1e-4, f"CKA should be 1.0 for orthogonal transform, got {cka}"
    
    def test_different_representations_low_cka(self):
        """Completely different representations should have low CKA."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 64).astype(np.float32)
        
        cka = linear_cka(Z0, Z1)
        # Random matrices should have low CKA
        assert cka < 0.5, f"Expected low CKA for random matrices, got {cka}"
    
    def test_small_matrices(self):
        """Test with small matrices (n=2, d=2)."""
        Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        cka = linear_cka(Z0, Z1)
        assert abs(cka - 1.0) < 1e-5
    
    def test_mismatched_n_raises_error(self):
        """Mismatched number of samples should raise ValueError."""
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(50, 64).astype(np.float32)
        
        with pytest.raises(ValueError, match="same number of samples"):
            linear_cka(Z0, Z1)
    
    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=5, max_value=20),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_identity_property_hypothesis(self, n, d0, d1):
        """Property-based test: CKA(Z, Z) == 1.0."""
        np.random.seed(42)
        Z = np.random.randn(n, d0).astype(np.float32)
        cka = linear_cka(Z, Z)
        assert abs(cka - 1.0) < 1e-4
    
    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_range_property_hypothesis(self, n, d):
        """Property-based test: 0 <= CKA <= 1."""
        np.random.seed(42)
        Z0 = np.random.randn(n, d).astype(np.float32)
        Z1 = np.random.randn(n, d).astype(np.float32)
        cka = linear_cka(Z0, Z1)
        assert 0.0 <= cka <= 1.0
    
    def test_highly_anisotropic_embeddings(self):
        """Test with highly anisotropic embeddings (one dominant singular value)."""
        np.random.seed(42)
        n = 100
        # Create matrix with one dominant direction
        direction = np.random.randn(64).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        
        Z = np.random.randn(n, 1).astype(np.float32) @ direction.reshape(1, -1)
        Z += np.random.randn(n, 64).astype(np.float32) * 0.01  # Small noise
        
        cka = linear_cka(Z, Z)
        assert abs(cka - 1.0) < 1e-5, "Identity should still hold"
