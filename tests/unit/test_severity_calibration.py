"""Tests for SeverityCalibration and calibrate_thresholds (lib_enhancement I2)."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.calibration import (
    SeverityCalibration,
    calibrate_thresholds,
)
from semantic_sentry.core.comparison import AlertSeverity
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.probes.anchor_set import AnchorSet


def _make_snapshot(seed: int) -> Snapshot:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((40, 16)).astype(np.float32)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Snapshot(
        model_id=f"m{seed}",
        checkpoint_hash=f"c{seed}",
        anchor_set_version="anchor",
        tower_count=1,
        tower_names=("encoder",),
        embeddings={"encoder": Z},
    )


class TestCalibrateThresholds:
    def test_needs_at_least_two_snapshots(self):
        with pytest.raises(ValueError, match="at least 2"):
            calibrate_thresholds([_make_snapshot(0)])

    def test_returns_thresholds_for_nps_and_cka(self):
        snaps = [_make_snapshot(s) for s in range(4)]
        cal = calibrate_thresholds(snaps)
        for tier in ("low", "medium", "high"):
            assert f"nps_{tier}" in cal.thresholds
            assert f"cka_{tier}" in cal.thresholds

    def test_thresholds_are_strictly_descending_across_tiers(self):
        # mu - 1*sigma > mu - 2*sigma > mu - 3*sigma (for sigma > 0).
        snaps = [_make_snapshot(s) for s in range(5)]
        cal = calibrate_thresholds(snaps)
        for metric in ("nps", "cka"):
            assert (
                cal.thresholds[f"{metric}_low"]
                >= cal.thresholds[f"{metric}_medium"]
                >= cal.thresholds[f"{metric}_high"]
            )

    def test_stats_carry_mean_and_std(self):
        snaps = [_make_snapshot(s) for s in range(4)]
        cal = calibrate_thresholds(snaps)
        for m in ("nps", "cka"):
            assert "mean" in cal.source_metric_stats[m]
            assert "std" in cal.source_metric_stats[m]
            assert cal.source_metric_stats[m]["n_pairs"] == 6.0  # C(4, 2)

    def test_wrong_sigma_count_rejected(self):
        snaps = [_make_snapshot(0), _make_snapshot(1)]
        with pytest.raises(ValueError, match="exactly 3"):
            calibrate_thresholds(snaps, n_sigmas=(1.0, 2.0))


class TestCalibrationFlowsIntoSeverity:
    def _make_adapter(self, rng_seed: int, n: int = 30) -> CustomAdapter:
        rng = np.random.default_rng(rng_seed)
        Z = rng.standard_normal((n, 16)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[: len(inputs)]},
            tower_names=["encoder"],
        )

    def _near_identical_adapter(self, seed: int, noise: float = 0.0, n: int = 30) -> CustomAdapter:
        rng = np.random.default_rng(0)
        base = rng.standard_normal((n, 16)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        if noise > 0:
            jitter = noise * np.random.default_rng(seed).standard_normal((n, 16)).astype(np.float32)
            Z = base + jitter
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        else:
            Z = base
        return CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[: len(inputs)]},
            tower_names=["encoder"],
        )

    def test_aggressive_calibration_flags_critical(self):
        """Calibration with absurdly tight thresholds escalates severity."""
        anchor = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        monitor = DriftMonitor()
        aggressive = SeverityCalibration(
            thresholds={
                "nps_low": 1.5, "nps_medium": 1.5, "nps_high": 1.5,
                "cka_low": 1.5, "cka_medium": 1.5, "cka_high": 1.5,
            },
            n_seeds=4,
        )
        s0 = monitor.snapshot(
            model=None, anchor_set=anchor, adapter=self._near_identical_adapter(0)
        )
        s1 = monitor.snapshot(
            model=None, anchor_set=anchor, adapter=self._near_identical_adapter(1, noise=0.01)
        )
        comp = monitor.compare(s0, s1, calibration=aggressive)
        assert comp.severity is AlertSeverity.CRITICAL

    def test_default_thresholds_pass_near_identical_snapshots(self):
        """Sanity: without calibration, near-identical snapshots should NOT trigger CRITICAL."""
        anchor = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        monitor = DriftMonitor()
        s0 = monitor.snapshot(
            model=None, anchor_set=anchor, adapter=self._near_identical_adapter(0)
        )
        s1 = monitor.snapshot(
            model=None, anchor_set=anchor, adapter=self._near_identical_adapter(1, noise=0.01)
        )
        comp = monitor.compare(s0, s1)
        assert comp.severity is not AlertSeverity.CRITICAL
