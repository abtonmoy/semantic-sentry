"""Phase 1.1: CKA stress tests."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from semantic_sentry.metrics.cka import linear_cka


@pytest.mark.stress
class TestCKAStress:
    """Stress tests for Linear CKA metric."""

    def test_cka_001_identical_matrices(self, make_embeddings):
        """CKA-001: Identical matrices must return exactly 1.0."""
        Z = make_embeddings(100, 64)
        assert linear_cka(Z, Z) == pytest.approx(1.0, abs=1e-6)

    def test_cka_002_rank_deficient(self, rng):
        """CKA-002: Rank-1 matrix (all rows are scaled versions of one vector)."""
        v = rng.standard_normal(64).astype(np.float32)
        scales = rng.uniform(0.5, 2.0, size=(100, 1)).astype(np.float32)
        Z = v * scales
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        result = linear_cka(Z, Z)
        assert result == pytest.approx(1.0, abs=1e-4), f"Rank-1 self-CKA was {result}"

    def test_cka_003_orthogonal_invariance(self, make_embeddings, rng):
        """CKA-003: CKA must be invariant to orthogonal transformations."""
        Z = make_embeddings(100, 64)
        # Generate random orthogonal matrix via QR decomposition
        H = rng.standard_normal((64, 64)).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        Z_rotated = Z @ Q
        result = linear_cka(Z, Z_rotated)
        assert result == pytest.approx(1.0, abs=1e-4), f"CKA after rotation was {result}"

    def test_cka_004_very_different_matrices(self, make_embeddings):
        """CKA-004: Random independent matrices should have CKA < 0.5."""
        Z0 = make_embeddings(100, 64, seed=1)
        Z1 = make_embeddings(100, 64, seed=999)
        result = linear_cka(Z0, Z1)
        assert result < 0.5, f"Independent matrices had CKA = {result}"

    def test_cka_005_zero_matrix(self):
        """CKA-005: Zero matrix should return 0.0 or raise, not NaN/Inf."""
        Z = np.zeros((100, 64), dtype=np.float32)
        result = linear_cka(Z, Z)
        assert np.isfinite(result), "CKA returned NaN or Inf for zero matrix"

    def test_cka_006_single_sample(self, rng):
        """CKA-006: n=1 must not crash."""
        Z0 = rng.standard_normal((1, 64)).astype(np.float32)
        Z1 = rng.standard_normal((1, 64)).astype(np.float32)
        result = linear_cka(Z0, Z1)
        assert np.isfinite(result), "CKA crashed on single sample"

    def test_cka_007_high_condition_number(self, rng):
        """CKA-007: Ill-conditioned matrix should not produce NaN."""
        Z = rng.standard_normal((100, 64)).astype(np.float32)
        Z[:, 0] *= 1e8  # make one dimension dominate
        Z[:, 1:] *= 1e-8
        result = linear_cka(Z, Z)
        assert np.isfinite(result), f"CKA was {result} for ill-conditioned matrix"

    @given(Z=arrays(np.float32, (50, 16), elements=dict(min_value=-10, max_value=10)))
    @settings(max_examples=50, deadline=5000)
    def test_cka_property_symmetry(self, Z):
        """CKA must be symmetric: CKA(Z0, Z1) == CKA(Z1, Z0)."""
        if np.any(np.isnan(Z)) or np.all(Z == 0):
            return  # skip degenerate
        rng = np.random.default_rng(0)
        Z1 = Z + rng.standard_normal(Z.shape).astype(np.float32) * 0.1
        assert linear_cka(Z, Z1) == pytest.approx(linear_cka(Z1, Z), abs=1e-5)

    @pytest.mark.timeout(10)
    def test_cka_008_large_dimensions(self, make_embeddings):
        """CKA-008: Large dimensions should complete within timeout."""
        Z0 = make_embeddings(1000, 2048)
        Z1 = make_embeddings(1000, 2048, seed=99)
        result = linear_cka(Z0, Z1)
        assert 0.0 <= result <= 1.0

    def test_cka_009_partial_overlap(self, make_embeddings):
        """CKA-009: Partially overlapping embeddings should have intermediate CKA."""
        Z_base = make_embeddings(100, 64, seed=42)
        # Create Z1 as mostly Z_base with some noise
        rng = np.random.default_rng(43)
        noise = rng.standard_normal((100, 64)).astype(np.float32) * 0.1
        Z1 = Z_base + noise
        result = linear_cka(Z_base, Z1)
        assert 0.5 < result < 0.99, f"Partial overlap CKA was {result}"

    def test_cka_010_different_sample_counts_raises(self, make_embeddings):
        """CKA-010: Different sample counts should raise ValueError."""
        Z0 = make_embeddings(100, 64)
        Z1 = make_embeddings(50, 64)
        with pytest.raises(ValueError):
            linear_cka(Z0, Z1)
