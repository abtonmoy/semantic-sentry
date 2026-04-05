"""Phase 5.2: Transfer function stress tests."""

import numpy as np
import pytest
from datetime import datetime, timezone

from semantic_sentry.transfer.function import LinearTransfer
from semantic_sentry.transfer.nps_bound import NPSBound
from semantic_sentry.core.comparison import Comparison, AlertSeverity


@pytest.mark.stress
class TestTransferStress:
    """Transfer function stress tests."""

    def test_trf_001_r_squared_synthetic(self, rng):
        """TRF-001: LinearTransfer must achieve R^2 > 0.7 on synthetic data."""
        # Generate 100 synthetic drift-degradation pairs
        n = 100
        cka_vals = rng.uniform(0.7, 1.0, n)
        nps_vals = rng.uniform(0.5, 1.0, n)
        iso_vals = rng.uniform(-0.2, 0.2, n)

        # True degradation is a linear function of drift + noise
        true_weights = np.array([0.5, 0.8, 0.3])
        features = np.column_stack([1 - cka_vals, 1 - nps_vals, np.abs(iso_vals)])
        degradations = features @ true_weights + rng.normal(0, 0.02, n)

        comparisons = []
        for i in range(n):
            c = Comparison(
                snapshot_v0_hash="base", snapshot_v1_hash=f"upd_{i}",
                global_metrics={"cka": cka_vals[i], "nps": nps_vals[i], "isotropy_delta": iso_vals[i]},
                per_tower_metrics=None, alignment_deltas=None,
            )
            comparisons.append(c)

        # Fit on 80%, test on 20%
        tf = LinearTransfer()
        tf.fit(comparisons[:80], degradations[:80].tolist())

        # Predict on held-out
        predictions = [tf.predict(c) for c in comparisons[80:]]
        actual = degradations[80:]

        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.7, f"R^2 was {r_squared:.3f}, expected > 0.7"

    def test_trf_nps_bound_identity(self):
        """NPSBound(1.0) must equal 0.0."""
        assert NPSBound.lower_bound(1.0) == 0.0

    def test_trf_nps_bound_half(self):
        """NPSBound(0.5) must equal 0.5."""
        assert NPSBound.lower_bound(0.5) == pytest.approx(0.5)

    def test_trf_predict_before_fit(self):
        """Predict before fit must raise ValueError."""
        tf = LinearTransfer()
        # Create a dummy comparison
        c = Comparison(
            snapshot_v0_hash="a", snapshot_v1_hash="b",
            global_metrics={"cka": 0.9, "nps": 0.85, "isotropy_delta": -0.01},
            per_tower_metrics=None, alignment_deltas=None,
        )
        with pytest.raises(ValueError):
            tf.predict(c)

    def test_trf_upper_bound(self):
        """NPS upper bound should be >= lower bound."""
        for nps_score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            lower = NPSBound.lower_bound(nps_score)
            upper = NPSBound.upper_bound(nps_score, k=10)
            assert lower <= upper, f"Lower bound {lower} > upper bound {upper} for NPS={nps_score}"

    def test_trf_confidence_interval(self):
        """NPS confidence interval should contain point estimate."""
        for nps_score in [0.3, 0.5, 0.7, 0.9]:
            for n_samples in [50, 100, 500]:
                lower, upper = NPSBound.confidence_interval(nps_score, n_samples)
                point_est = 1.0 - nps_score
                assert lower <= point_est <= upper, \
                    f"Point estimate {point_est} not in CI [{lower}, {upper}]"

    def test_trf_fit_insufficient_samples(self):
        """Fit with < 3 samples must raise ValueError."""
        tf = LinearTransfer()
        comparisons = [
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash="b",
                global_metrics={"cka": 0.9, "nps": 0.85, "isotropy_delta": 0.0},
            ),
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash="c",
                global_metrics={"cka": 0.8, "nps": 0.75, "isotropy_delta": 0.1},
            ),
        ]
        with pytest.raises(ValueError):
            tf.fit(comparisons, [0.1, 0.2])

    def test_trf_length_mismatch(self):
        """Fit with mismatched lengths must raise ValueError."""
        tf = LinearTransfer()
        comparisons = [
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={"cka": 0.9 - i*0.01, "nps": 0.85 - i*0.01, "isotropy_delta": 0.0},
            )
            for i in range(5)
        ]
        with pytest.raises(ValueError):
            tf.fit(comparisons, [0.1, 0.2, 0.3])  # Only 3 degradations for 5 comparisons

    def test_trf_prediction_clamped(self, rng):
        """Predictions must be clamped to [0, 1]."""
        # Create comparisons that would predict outside [0, 1]
        comparisons = []
        for i in range(10):
            c = Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={"cka": 0.5, "nps": 0.3, "isotropy_delta": 10.0},  # Extreme values
            )
            comparisons.append(c)

        # Fit with small degradations
        tf = LinearTransfer()
        tf.fit(comparisons, [0.1] * 10)

        # Predict - should be clamped
        pred = tf.predict(comparisons[0])
        assert 0.0 <= pred <= 1.0, f"Prediction {pred} not in [0, 1]"

    def test_trf_feature_importance(self, rng):
        """Feature importance should be extractable after fit."""
        n = 50
        comparisons = [
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={
                    "cka": rng.uniform(0.7, 1.0),
                    "nps": rng.uniform(0.5, 1.0),
                    "isotropy_delta": rng.uniform(-0.2, 0.2),
                },
            )
            for i in range(n)
        ]
        degradations = rng.uniform(0, 0.5, n).tolist()

        tf = LinearTransfer()
        tf.fit(comparisons, degradations)

        importance = tf.get_feature_importance()
        assert "1-cka" in importance
        assert "1-nps" in importance
        assert "|isotropy_delta|" in importance
        assert all(v >= 0 for v in importance.values())
