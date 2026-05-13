"""Tests for the temporal layer F1/F2/F3 and G4 register_with_temporal."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.metrics.registry import MetricRegistry
from semantic_sentry.metrics.temporal import (
    acceleration,
    list_temporal,
    plateau,
    register_with_temporal,
    velocity,
)


def _snapshot(Z: np.ndarray, model_id: str = "m", anchor_hash: str = "anchor") -> Snapshot:
    return Snapshot(
        model_id=model_id,
        checkpoint_hash=model_id,
        anchor_set_version=anchor_hash,
        tower_count=1,
        tower_names=("encoder",),
        embeddings={"encoder": Z},
    )


def _trajectory_constant(T: int, n: int = 20, d: int = 8) -> list[Snapshot]:
    """T identical snapshots — every pairwise metric should be 1 (no drift)."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n, d)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return [_snapshot(Z, model_id=f"m{i}") for i in range(T)]


def _trajectory_drifting(T: int, n: int = 20, d: int = 8) -> list[Snapshot]:
    """T snapshots with monotonically growing perturbation per step."""
    rng = np.random.default_rng(0)
    Z0 = rng.standard_normal((n, d)).astype(np.float32)
    Z0 /= np.linalg.norm(Z0, axis=1, keepdims=True)
    snaps = [_snapshot(Z0, model_id="m0")]
    for i in range(1, T):
        Z = Z0 + 0.05 * i * np.random.default_rng(i).standard_normal((n, d)).astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        snaps.append(_snapshot(Z, model_id=f"m{i}"))
    return snaps


class TestVelocity:
    def test_constant_trajectory_velocity_constant(self):
        # Identical snapshots -> every pairwise NPS is 1.0 -> cumulative
        # trajectory is a straight line -> velocity is constant.
        snaps = _trajectory_constant(T=5)
        v = velocity(snaps, [0.0, 1.0, 2.0, 3.0, 4.0], "nps")
        assert v.shape == (5,)
        assert np.allclose(v, v[0])

    def test_drifting_trajectory_velocity_finite(self):
        snaps = _trajectory_drifting(T=5)
        v = velocity(snaps, [0.0, 1.0, 2.0, 3.0, 4.0], "nps")
        assert v.shape == (5,)
        assert np.all(np.isfinite(v))

    def test_length_mismatch_raises(self):
        snaps = _trajectory_constant(T=3)
        with pytest.raises(ValueError, match="length mismatch"):
            velocity(snaps, [0.0, 1.0], "nps")

    def test_too_short_raises(self):
        snaps = _trajectory_constant(T=1)
        with pytest.raises(ValueError, match="at least 2"):
            velocity(snaps, [0.0], "nps")


class TestAcceleration:
    def test_constant_trajectory_acceleration_zero(self):
        snaps = _trajectory_constant(T=5)
        a = acceleration(snaps, [0.0, 1.0, 2.0, 3.0, 4.0], "nps")
        assert a.shape == (5,)
        assert np.allclose(a, 0.0, atol=1e-6)


class TestPlateau:
    def test_constant_trajectory_is_plateau(self):
        # NPS between identical snapshots is constant -> zero velocity and
        # acceleration after the first step; expect a True signal from
        # the kth checkpoint onwards.
        snaps = _trajectory_constant(T=8)
        sig = plateau(snaps, list(range(8)), "nps", eps=1e-3, delta=1e-3, k=3)
        assert sig.shape == (8,)
        assert sig[-1]

    def test_drifting_trajectory_no_plateau(self):
        snaps = _trajectory_drifting(T=5)
        sig = plateau(snaps, list(range(5)), "nps", eps=1e-4, delta=1e-4, k=3)
        assert not sig.any()


class TestRegisterWithTemporal:
    def test_register_with_temporal_creates_three_entries(self):
        def my_metric(Z0, Z1):
            return float(np.mean(np.abs(Z0 - Z1)))

        register_with_temporal("custom_drift", my_metric, range=(0.0, 10.0))
        assert "custom_drift" in MetricRegistry().list_metrics()
        temporal_names = list_temporal()
        assert "custom_drift_velocity" in temporal_names
        assert "custom_drift_acceleration" in temporal_names

    def test_registered_temporal_callable(self):
        def my_metric(Z0, Z1):
            return float(np.mean(np.abs(Z0 - Z1)))

        register_with_temporal("custom_drift", my_metric, range=(0.0, 10.0))
        from semantic_sentry.metrics.temporal import get_temporal

        snaps = _trajectory_constant(T=4)
        v_fn = get_temporal("custom_drift_velocity")
        v = v_fn(snaps, [0.0, 1.0, 2.0, 3.0])
        assert v.shape == (4,)
