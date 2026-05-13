"""Tests for Neighborhood Preservation Score (NPS) metric."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_sentry.metrics.nps import nps, nps_bounds, nps_per_point


class TestNPS:
    """Test NPS properties."""

    def test_self_preservation(self):
        """NPS(Z, Z, k) == 1.0 (self-preservation)."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)
        score = nps(Z, Z, k=10)
        assert abs(score - 1.0) < 1e-5, f"Expected 1.0, got {score}"

    def test_range_property(self):
        """0.0 <= NPS(Z0, Z1, k) <= 1.0."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 64).astype(np.float32)

        score = nps(Z0, Z1, k=10)

        assert 0.0 <= score <= 1.0, f"NPS out of range: {score}"

    def test_monotonic_with_perturbation(self):
        """Larger perturbations yield lower NPS."""
        np.random.seed(42)
        Z = np.random.randn(100, 64).astype(np.float32)

        # Small perturbation
        Z_small = Z + np.random.randn(100, 64).astype(np.float32) * 0.01
        nps_small = nps(Z, Z_small, k=10)

        # Large perturbation
        Z_large = Z + np.random.randn(100, 64).astype(np.float32) * 0.5
        nps_large = nps(Z, Z_large, k=10)

        assert nps_small > nps_large, (
            f"Small perturbation NPS ({nps_small}) should be > large ({nps_large})"
        )

    def test_per_point_returns_correct_shape(self):
        """nps_per_point returns array of shape (n,)."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 64).astype(np.float32)

        per_point = nps_per_point(Z0, Z1, k=10)

        assert per_point.shape == (100,), f"Expected shape (100,), got {per_point.shape}"
        assert np.all(per_point >= 0) and np.all(per_point <= 1), "Per-point scores out of range"

    def test_per_point_mean_equals_nps(self):
        """Mean of per-point NPS equals global NPS."""
        np.random.seed(42)
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(100, 64).astype(np.float32)

        global_nps = nps(Z0, Z1, k=10)
        per_point = nps_per_point(Z0, Z1, k=10)

        assert abs(np.mean(per_point) - global_nps) < 1e-5

    def test_small_matrices(self):
        """Test with small matrices (n=20, k=5)."""
        np.random.seed(42)
        Z0 = np.random.randn(20, 8).astype(np.float32)
        Z1 = Z0 + np.random.randn(20, 8).astype(np.float32) * 0.1

        score = nps(Z0, Z1, k=5)
        assert 0.0 <= score <= 1.0

    def test_mismatched_n_raises_error(self):
        """Mismatched number of samples should raise ValueError."""
        Z0 = np.random.randn(100, 64).astype(np.float32)
        Z1 = np.random.randn(50, 64).astype(np.float32)

        with pytest.raises(ValueError, match="same number of samples"):
            nps(Z0, Z1)

    def test_k_larger_than_n(self):
        """k larger than n should handle gracefully."""
        Z0 = np.random.randn(10, 8).astype(np.float32)
        Z1 = np.random.randn(10, 8).astype(np.float32)

        # Should not raise error, returns all ones (perfect preservation by default)
        score = nps(Z0, Z1, k=20)
        assert score == 1.0

    @given(
        st.integers(min_value=20, max_value=50),
        st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_self_preservation_hypothesis(self, n, d):
        """Property-based test: NPS(Z, Z) == 1.0."""
        np.random.seed(42)
        Z = np.random.randn(n, d).astype(np.float32)
        score = nps(Z, Z, k=5)
        assert abs(score - 1.0) < 1e-4

    @given(
        st.integers(min_value=20, max_value=50),
        st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_range_property_hypothesis(self, n, d):
        """Property-based test: 0 <= NPS <= 1."""
        np.random.seed(42)
        Z0 = np.random.randn(n, d).astype(np.float32)
        Z1 = np.random.randn(n, d).astype(np.float32)
        score = nps(Z0, Z1, k=5)
        assert 0.0 <= score <= 1.0


class TestNPSBounds:
    """Test NPS theoretical bounds."""

    def test_lower_bound_zero_at_one(self):
        """lower_bound(1.0) == 0.0."""
        lower, _ = nps_bounds(1.0)
        assert lower == 0.0

    def test_lower_bound_half_at_half(self):
        """lower_bound(0.5) == 0.5."""
        lower, _ = nps_bounds(0.5)
        assert lower == 0.5

    def test_lower_bound_zero_at_high(self):
        """lower_bound(0.9) == 0.1."""
        lower, _ = nps_bounds(0.9)
        assert abs(lower - 0.1) < 1e-6

    def test_bounds_valid(self):
        """lower <= upper for all valid NPS."""
        for nps_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            lower, upper = nps_bounds(nps_val)
            assert lower <= upper, f"Lower {lower} > upper {upper} for NPS={nps_val}"
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
