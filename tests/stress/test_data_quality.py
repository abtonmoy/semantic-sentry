"""Phase 4: Data quality edge case tests."""

import numpy as np
import pytest

from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.isotropy import isotropy
from semantic_sentry.probes.anchor_set import AnchorSet


@pytest.mark.stress
class TestDataQuality:
    """Data quality edge case stress tests."""

    def test_dqa_001_nan_values(self, make_embeddings):
        """DQA-001: NaN values must raise ValueError, not silently corrupt."""
        Z = make_embeddings(100, 64)
        Z[0, 0] = np.nan
        with pytest.raises((ValueError, RuntimeError)):
            linear_cka(Z, Z)

    def test_dqa_002_inf_values(self, make_embeddings):
        """DQA-002: Inf values must raise ValueError."""
        Z = make_embeddings(100, 64)
        Z[0, 0] = np.inf
        with pytest.raises((ValueError, RuntimeError)):
            linear_cka(Z, Z)

    def test_dqa_003_all_zeros(self):
        """DQA-003: All-zero matrix must not return NaN."""
        Z = np.zeros((100, 64), dtype=np.float32)
        result = linear_cka(Z, Z)
        assert np.isfinite(result)

    def test_dqa_004_very_large_values(self, rng):
        """DQA-004: Large values (1e15) must not overflow."""
        Z = rng.standard_normal((100, 64)).astype(np.float64) * 1e15
        result = linear_cka(Z, Z)
        assert np.isfinite(result)

    def test_dqa_005_very_small_values(self, rng):
        """DQA-005: Very small values (1e-15) must not underflow to zero."""
        Z = rng.standard_normal((100, 64)).astype(np.float64) * 1e-15
        result = isotropy(Z)
        assert np.isfinite(result)

    def test_dqa_006_empty_anchor_set(self):
        """DQA-006: Empty inputs must handle gracefully."""
        # Empty anchor set - should handle gracefully
        anchor = AnchorSet(inputs=[], labels=np.array([]), modality="text")
        assert anchor.n_samples == 0

    def test_dqa_007_single_sample_anchor(self, rng):
        """DQA-007: Single-sample anchor set must be handled."""
        anchor = AnchorSet(
            inputs=["single"],
            labels=np.array(["only"]),
            modality="text",
        )
        assert anchor.n_samples == 1

    def test_dqa_008_mixed_precision(self, rng):
        """DQA-008: Mixed precision inputs should work."""
        Z0 = rng.standard_normal((100, 64)).astype(np.float32)
        Z1 = rng.standard_normal((100, 64)).astype(np.float64)
        result = linear_cka(Z0, Z1)
        assert np.isfinite(result)

    def test_dqa_009_negative_zero(self):
        """DQA-009: Negative zero should be handled."""
        Z = np.zeros((100, 64), dtype=np.float32)
        Z[0, 0] = -0.0  # Negative zero
        result = linear_cka(Z, Z)
        assert np.isfinite(result)

    def test_dqa_010_subnormal_values(self, rng):
        """DQA-010: Subnormal (denormal) floating point values."""
        Z = rng.standard_normal((100, 64)).astype(np.float32) * 1e-40
        result = isotropy(Z)
        assert np.isfinite(result) or result == 0.0
