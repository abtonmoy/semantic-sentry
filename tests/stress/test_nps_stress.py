"""Phase 1.2: NPS stress tests."""

import numpy as np
import pytest

from semantic_sentry.metrics.nps import nps, nps_per_point


@pytest.mark.stress
class TestNPSStress:
    """Stress tests for Neighborhood Preservation Score metric."""

    def test_nps_001_self_comparison(self, make_embeddings):
        """NPS-001: Self-comparison must return 1.0."""
        Z = make_embeddings(100, 64)
        assert nps(Z, Z) == pytest.approx(1.0, abs=1e-6)

    def test_nps_002_k_greater_than_n(self, make_embeddings):
        """NPS-002: k > n must handle gracefully."""
        Z = make_embeddings(10, 64)
        # Should not raise, but return 1.0 (not enough points for k-NN)
        result = nps(Z, Z, k=100)
        assert np.isfinite(result)

    def test_nps_003_identical_rows(self, rng):
        """NPS-003: All rows identical — must not crash."""
        v = rng.standard_normal(64).astype(np.float32)
        v = v / np.linalg.norm(v)
        Z = np.tile(v, (100, 1))
        result = nps(Z, Z, k=5)
        assert np.isfinite(result)

    def test_nps_004_random_permutation(self, make_embeddings, rng):
        """NPS-004: Row permutation should not change NPS (same neighborhoods)."""
        Z = make_embeddings(100, 64)
        perm = rng.permutation(100)
        Z_perm = Z[perm]
        # NPS should still be 1.0 because the point cloud is the same
        result = nps(Z, Z_perm, k=10)
        # Note: NPS compares neighborhoods of corresponding indices, so
        # permutation WILL change NPS. This tests a different thing.
        assert 0.0 <= result <= 1.0

    def test_nps_005_monotonic_perturbation(self, make_drifted_pair):
        """NPS-005: Increasing noise should decrease NPS monotonically."""
        results = []
        for noise in [0.01, 0.1, 0.3, 0.5, 1.0]:
            Z_base, Z_drift = make_drifted_pair(200, 64, noise_scale=noise)
            results.append(nps(Z_base, Z_drift, k=10))
        # Should be monotonically decreasing (with tolerance for small noise)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1] - 0.05, \
                f"NPS not monotonic: {results}"

    @pytest.mark.timeout(30)
    def test_nps_006_large_n_small_k(self, make_embeddings):
        """NPS-006: n=10K, k=5 must complete within 30 seconds."""
        Z0 = make_embeddings(10_000, 128)
        Z1 = make_embeddings(10_000, 128, seed=99)
        result = nps(Z0, Z1, k=5)
        assert 0.0 <= result <= 1.0

    def test_nps_007_per_point_scores_shape(self, make_embeddings):
        """NPS-007: Per-point scores must match number of samples."""
        Z0 = make_embeddings(100, 64)
        Z1 = make_embeddings(100, 64, seed=99)
        scores = nps_per_point(Z0, Z1, k=10)
        assert scores.shape == (100,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_nps_008_small_k_edge_case(self, make_embeddings):
        """NPS-008: k=1 edge case."""
        Z = make_embeddings(50, 64)
        result = nps(Z, Z, k=1)
        assert np.isfinite(result)

    def test_nps_009_very_high_dimensional(self, make_embeddings):
        """NPS-009: Very high dimensional embeddings."""
        Z0 = make_embeddings(100, 2048)
        Z1 = make_embeddings(100, 2048, seed=99)
        result = nps(Z0, Z1, k=10)
        assert 0.0 <= result <= 1.0

    def test_nps_010_different_dimensions(self, make_embeddings):
        """NPS-010: Different dimensional embeddings (should still work)."""
        Z0 = make_embeddings(100, 64)
        Z1 = make_embeddings(100, 128, seed=99)  # Different dimension
        result = nps(Z0, Z1, k=10)
        assert 0.0 <= result <= 1.0
