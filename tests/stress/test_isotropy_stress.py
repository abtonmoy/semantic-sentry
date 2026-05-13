"""Phase 1.3: Isotropy stress tests."""

import numpy as np
import pytest

from semantic_sentry.metrics.isotropy import effective_dimensionality, isotropy, isotropy_delta


@pytest.mark.stress
class TestIsotropyStress:
    """Stress tests for isotropy metrics."""

    def test_iso_001_spherical_gaussian(self, rng):
        """ISO-001: Spherical Gaussian should be highly isotropic."""
        Z = rng.standard_normal((1000, 64)).astype(np.float32)
        result = isotropy(Z)
        assert result > 0.5, f"Spherical Gaussian isotropy was only {result}"

    def test_iso_002_rank_1(self, rng):
        """ISO-002: Rank-1 matrix should have near-zero isotropy."""
        v = rng.standard_normal(64).astype(np.float32)
        scales = rng.uniform(0.1, 2.0, size=(100, 1)).astype(np.float32)
        Z = v * scales
        result = isotropy(Z)
        assert result < 0.1, f"Rank-1 isotropy was {result}"

    def test_iso_003_square_matrix(self, make_embeddings):
        """ISO-003: n == d should compute stably."""
        Z = make_embeddings(64, 64)
        result = isotropy(Z)
        assert np.isfinite(result) and 0 <= result <= 1

    def test_iso_004_tall_matrix(self, make_embeddings):
        """ISO-004: n << d (10 samples, 512 dims)."""
        Z = make_embeddings(10, 512)
        result = isotropy(Z)
        assert np.isfinite(result)

    def test_iso_005_wide_matrix(self, make_embeddings):
        """ISO-005: n >> d (10000 samples, 8 dims)."""
        Z = make_embeddings(10_000, 8)
        result = isotropy(Z)
        assert np.isfinite(result) and 0 <= result <= 1

    def test_iso_006_delta_same_matrix(self, make_embeddings):
        """ISO-006: Delta isotropy of same matrix should be 0."""
        Z = make_embeddings(100, 64)
        result = isotropy_delta(Z, Z)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_iso_007_delta_different_isotropy(self, rng):
        """ISO-007: Delta between low and high isotropy matrices."""
        # Low isotropy (rank-1)
        v = rng.standard_normal(64).astype(np.float32)
        Z_low = np.tile(v, (100, 1)) * rng.uniform(0.1, 2.0, size=(100, 1)).astype(np.float32)

        # High isotropy (spherical Gaussian)
        Z_high = rng.standard_normal((100, 64)).astype(np.float32)

        result = isotropy_delta(Z_low, Z_high)
        # Should be positive (high is more isotropic than low)
        assert result > 0, f"Expected positive delta, got {result}"

    def test_iso_008_effective_dim_full(self, rng):
        """ISO-008: Effective dimensionality of full-rank matrix."""
        Z = rng.standard_normal((100, 64)).astype(np.float32)
        ed = effective_dimensionality(Z, threshold=0.9)
        assert 0 < ed <= 64

    def test_iso_009_effective_dim_rank_deficient(self, rng):
        """ISO-009: Effective dimensionality of rank-deficient matrix."""
        v = rng.standard_normal(64).astype(np.float32)
        Z = np.tile(v, (100, 1)) * rng.uniform(0.1, 2.0, size=(100, 1)).astype(np.float32)
        ed = effective_dimensionality(Z, threshold=0.9)
        # Should be close to 1 for rank-1 matrix
        assert ed <= 5, f"Expected low effective dim for rank-1, got {ed}"

    def test_iso_010_effective_dim_tall(self, rng):
        """ISO-010: Effective dimensionality for tall matrix."""
        Z = rng.standard_normal((10, 512)).astype(np.float32)
        ed = effective_dimensionality(Z, threshold=0.9)
        # Cannot exceed number of samples
        assert ed <= 10
