"""Tests for Isotropy metrics."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from semantic_sentry.metrics.isotropy import isotropy, isotropy_delta, effective_dimensionality


class TestIsotropy:
    """Test isotropy properties."""
    
    def test_range_property(self):
        """0.0 <= isotropy(Z) <= 1.0."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        iso = isotropy(Z)
        
        assert 0.0 <= iso <= 1.0, f"Isotropy out of range: {iso}"
    
    def test_perfectly_isotropic(self):
        """Spherical Gaussian should be relatively isotropic."""
        np.random.seed(42)
        # Generate isotropic data
        Z = np.random.randn(1000, 64).astype(np.float32)
        iso = isotropy(Z)
        
        # Should be reasonably high (not necessarily 1.0 due to finite samples)
        assert iso > 0.5, f"Expected high isotropy for Gaussian, got {iso}"
    
    def test_highly_anisotropic(self):
        """Data with one dominant direction should have low isotropy."""
        np.random.seed(42)
        n = 100
        direction = np.random.randn(64).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        
        # Create data with one dominant direction
        Z = np.random.randn(n, 1).astype(np.float32) @ direction.reshape(1, -1)
        Z += np.random.randn(n, 64).astype(np.float32) * 0.001
        
        iso = isotropy(Z)
        assert iso < 0.1, f"Expected low isotropy for anisotropic data, got {iso}"
    
    def test_small_matrices(self):
        """Test with small matrices (n=2, d=2)."""
        Z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        iso = isotropy(Z)
        assert 0.0 <= iso <= 1.0
    
    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_range_property_hypothesis(self, n, d):
        """Property-based test: 0 <= isotropy <= 1."""
        np.random.seed(42)
        Z = np.random.randn(n, d).astype(np.float32)
        iso = isotropy(Z)
        assert 0.0 <= iso <= 1.0


class TestIsotropyDelta:
    """Test isotropy_delta properties."""
    
    def test_identity_property(self):
        """isotropy_delta(Z, Z) == 0.0."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        delta = isotropy_delta(Z, Z)
        
        assert abs(delta) < 1e-5, f"Expected 0.0, got {delta}"
    
    def test_range_property(self):
        """-1.0 <= isotropy_delta(Z0, Z1) <= 1.0."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 64).astype(np.float32)
        
        delta = isotropy_delta(Z0, Z1)
        
        assert -1.0 <= delta <= 1.0, f"Delta out of range: {delta}"
    
    def test_positive_for_more_isotropic(self):
        """Delta is positive when Z1 is more isotropic."""
        np.random.seed(42)
        
        # Anisotropic data
        direction = np.random.randn(64).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        Z_aniso = np.random.randn(100, 1).astype(np.float32) @ direction.reshape(1, -1)
        
        # Isotropic data
        Z_iso = np.random.randn(100, 64).astype(np.float32)
        
        delta = isotropy_delta(Z_aniso, Z_iso)
        assert delta > 0, f"Expected positive delta (more isotropic), got {delta}"
    
    def test_negative_for_less_isotropic(self):
        """Delta is negative when Z1 is less isotropic."""
        np.random.seed(42)
        
        # Isotropic data
        Z_iso = np.random.randn(100, 64).astype(np.float32)
        
        # Anisotropic data
        direction = np.random.randn(64).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        Z_aniso = np.random.randn(100, 1).astype(np.float32) @ direction.reshape(1, -1)
        
        delta = isotropy_delta(Z_iso, Z_aniso)
        assert delta < 0, f"Expected negative delta (less isotropic), got {delta}"
    
    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_identity_property_hypothesis(self, n, d):
        """Property-based test: isotropy_delta(Z, Z) == 0."""
        np.random.seed(42)
        Z = np.random.randn(n, d).astype(np.float32)
        delta = isotropy_delta(Z, Z)
        assert abs(delta) < 1e-4
    
    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_range_property_hypothesis(self, n, d):
        """Property-based test: -1 <= isotropy_delta <= 1."""
        np.random.seed(42)
        Z0 = np.random.randn(n, d).astype(np.float32)
        Z1 = np.random.randn(n, d).astype(np.float32)
        delta = isotropy_delta(Z0, Z1)
        assert -1.0 <= delta <= 1.0


class TestEffectiveDimensionality:
    """Test effective dimensionality computation."""
    
    def test_returns_positive_integer(self):
        """Effective dimensionality should be positive integer."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        eff_dim = effective_dimensionality(Z)
        
        assert isinstance(eff_dim, int)
        assert eff_dim > 0
        assert eff_dim <= 64
    
    def test_low_dimensional_structure(self):
        """Data with low-rank structure should have low effective dim."""
        np.random.seed(42)
        # Create rank-10 data in 64-dim space
        basis = np.random.randn(10, 64).astype(np.float32)
        coeffs = np.random.randn(100, 10).astype(np.float32)
        Z = coeffs @ basis
        
        eff_dim = effective_dimensionality(Z, threshold=0.95)
        assert eff_dim < 20, f"Expected low effective dim for rank-10 data, got {eff_dim}"
    
    def test_high_dimensional_structure(self):
        """Full-rank data should have high effective dim."""
        np.random.seed(42)
        # Create full-rank data
        Z = np.random.randn(1000, 64).astype(np.float32)
        
        eff_dim = effective_dimensionality(Z, threshold=0.95)
        assert eff_dim > 30, f"Expected high effective dim for full-rank data, got {eff_dim}"
