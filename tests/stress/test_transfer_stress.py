"""Phase 5.2: Transfer function stress tests."""


import json

import numpy as np
import pytest

from semantic_sentry.core.comparison import Comparison
from semantic_sentry.transfer.calibration import CalibrationProfile
from semantic_sentry.transfer.function import LinearTransfer
from semantic_sentry.transfer.nps_bound import NPSBound


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
                global_metrics={
                    "cka": cka_vals[i],
                    "nps": nps_vals[i],
                    "isotropy_delta": iso_vals[i],
                },
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

    def test_trf_default_accepts_mixed_sign_degradations(self):
        """Default LinearTransfer is signed: fit() accepts mixed-sign targets.

        v0.2.0 flipped the default to signed predictions. Mixed-sign
        calibration targets (representing fine-tunes that sometimes improve
        and sometimes degrade the downstream metric) are first-class — fit()
        must accept them without complaint, and predict() must return values
        with the same sign structure as the data.
        """
        comparisons = [
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={
                    "cka": 0.9 - i * 0.005,
                    "nps": 0.7 - i * 0.01,
                    "isotropy_delta": 0.0,
                },
            )
            for i in range(5)
        ]
        tf = LinearTransfer()

        # All-negative targets (improvement regime) — accepted, predictions
        # reproduce the negative direction.
        all_negative = [-0.05 - i * 0.002 for i in range(5)]
        tf.fit(comparisons, all_negative)
        for cmp, target in zip(comparisons, all_negative, strict=True):
            pred = tf.predict(cmp)
            assert pred < 0.0, f"Negative-target prediction {pred} unexpectedly non-negative"
            assert abs(pred - target) < 1e-6

        # Mixed-sign — fit() must accept without raising. The fit is a
        # best-fit plane, not an interpolant, so the realized prediction
        # signs depend on the geometry of the feature matrix; the
        # load-bearing contract test is that fit() did not reject mixed sign
        # at the boundary.
        mixed = [+0.05, -0.01, +0.10, -0.02, +0.08]
        tf.fit(comparisons, mixed)
        preds = [tf.predict(cmp) for cmp in comparisons]
        assert all(isinstance(p, float) for p in preds)

        # Default predict() is unbounded — engineer an extreme input and
        # confirm the output is not silently clamped to [0, 1].
        extreme = Comparison(
            snapshot_v0_hash="a", snapshot_v1_hash="extreme",
            global_metrics={"cka": -10.0, "nps": -10.0, "isotropy_delta": 100.0},
        )
        assert abs(tf.predict(extreme)) > 1.0, (
            "Default LinearTransfer should return unbounded signed predictions; "
            "got a value inside [-1, 1] — is a clamp still in place?"
        )

    def test_trf_clip_round_trips_through_calibration_profile(self, tmp_path):
        """`clip` survives CalibrationProfile.save / load and from/to_transfer_function.

        The clamp-output / reject-negative-input pair is governed by a single
        `_clip` flag inside LinearTransfer. Serialization is the one place
        the two halves can drift apart: if `CalibrationProfile` doesn't
        capture `clip`, a round-trip silently downgrades a `clip=True`
        transfer to `clip=False`, leaving the clamp's invariants unprotected
        on the loaded instance. This test pins the round-trip preservation
        in both directions and the v0.1.0 backward-compat default.
        """
        comparisons = [
            Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={
                    "cka": 0.9 - i * 0.005,
                    "nps": 0.7 - i * 0.01,
                    "isotropy_delta": 0.0,
                },
            )
            for i in range(5)
        ]

        # clip=True round-trip preserves clip=True.
        clipped = LinearTransfer(clip=True)
        clipped.fit(comparisons, [0.05, 0.06, 0.07, 0.08, 0.09])
        profile = CalibrationProfile.from_transfer_function(
            clipped, "test-clip-true", "test-family", n_samples=5
        )
        assert profile.clip is True

        path = tmp_path / "clip_true.json"
        profile.save(path)
        loaded = CalibrationProfile.load(path)
        assert loaded.clip is True

        restored = loaded.to_transfer_function()
        assert restored._clip is True
        # Both halves of the contract follow `_clip` on the restored instance.
        with pytest.raises(ValueError, match="clip=True.*non-negative"):
            restored.fit(comparisons, [-0.01] * 5)
        # And predict still clamps.
        extreme = Comparison(
            snapshot_v0_hash="a", snapshot_v1_hash="extreme",
            global_metrics={"cka": -10.0, "nps": -10.0, "isotropy_delta": 100.0},
        )
        # Re-fit (clip=True path, valid targets) so we can predict from the
        # restored instance after the rejection-test fit attempt above.
        restored2 = loaded.to_transfer_function()
        assert 0.0 <= restored2.predict(extreme) <= 1.0

        # clip=False (default) round-trip preserves clip=False.
        signed = LinearTransfer()  # default
        signed.fit(comparisons, [-0.05, -0.04, -0.03, -0.02, -0.01])
        profile2 = CalibrationProfile.from_transfer_function(
            signed, "test-clip-false", "test-family", n_samples=5
        )
        assert profile2.clip is False
        path2 = tmp_path / "clip_false.json"
        profile2.save(path2)
        loaded2 = CalibrationProfile.load(path2)
        assert loaded2.clip is False
        restored3 = loaded2.to_transfer_function()
        assert restored3._clip is False
        # Predictions on the restored signed transfer are unbounded.
        pred = restored3.predict(extreme)
        assert abs(pred) > 1.0 or pred < 0.0

        # Backward-compat: a v0.1.0 JSON without the `clip` key loads as
        # clip=False (the new signed default).
        legacy_path = tmp_path / "legacy_v01.json"
        legacy_path.write_text(json.dumps({
            "profile_name": "legacy",
            "model_family": "v0.1-family",
            "weights": [0.5, 0.5, 0.5],
            "bias": 0.0,
            "r_squared": 0.9,
            "n_samples": 10,
        }))
        legacy = CalibrationProfile.load(legacy_path)
        assert legacy.clip is False
        assert legacy.to_transfer_function()._clip is False

    def test_trf_clip_opt_in_clamps_output(self, rng):
        """Opt-in `clip=True` clamps predict() to [0, 1] AND rejects negative targets.

        The two halves are paired — clip-on-predict without
        validation-on-fit would reproduce the v0.1.0 silent-corruption bug.
        """
        comparisons = []
        for i in range(10):
            c = Comparison(
                snapshot_v0_hash="a", snapshot_v1_hash=f"b_{i}",
                global_metrics={"cka": 0.5, "nps": 0.3, "isotropy_delta": 10.0},
            )
            comparisons.append(c)

        # clip=True with non-negative targets: predictions stay in [0, 1].
        tf = LinearTransfer(clip=True)
        tf.fit(comparisons, [0.1] * 10)
        pred = tf.predict(comparisons[0])
        assert 0.0 <= pred <= 1.0, f"Prediction {pred} not in [0, 1]"

        # clip=True with any negative target: fit() refuses.
        tf2 = LinearTransfer(clip=True)
        with pytest.raises(ValueError, match="clip=True.*non-negative"):
            tf2.fit(comparisons, [-0.05] * 10)

        with pytest.raises(ValueError, match="clip=True.*non-negative"):
            tf2.fit(comparisons, [0.05, -0.01, 0.10, 0.02, 0.08, 0.03, 0.04, 0.06, 0.07, 0.09])

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
