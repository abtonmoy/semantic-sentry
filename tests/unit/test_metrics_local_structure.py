"""Tests for B1 (trustworthiness), B2 (continuity), and B5 (NPS curve)."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.metrics.local_structure import (
    continuity,
    register_local_structure_metrics,
    trustworthiness,
)
from semantic_sentry.metrics.registry import MetricRegistry


def _unit_normal(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class TestTrustworthinessContinuity:
    def test_identical_inputs_give_score_one(self):
        X = _unit_normal(50, 16, seed=0)
        assert trustworthiness(X, X, k=5) == pytest.approx(1.0)
        assert continuity(X, X, k=5) == pytest.approx(1.0)

    def test_random_remap_lowers_both(self):
        Z0 = _unit_normal(60, 16, seed=0)
        # Permute rows of Z1 — neighborhoods are now random relative to Z0.
        rng = np.random.default_rng(1)
        perm = rng.permutation(60)
        Z1 = Z0[perm]
        t = trustworthiness(Z0, Z1, k=5)
        c = continuity(Z0, Z1, k=5)
        assert t < 0.95
        assert c < 0.95

    def test_small_perturbation_close_to_one(self):
        Z0 = _unit_normal(80, 16, seed=0)
        Z1 = Z0 + 0.01 * np.random.default_rng(1).standard_normal(Z0.shape).astype(np.float32)
        Z1 = Z1 / np.linalg.norm(Z1, axis=1, keepdims=True)
        assert trustworthiness(Z0, Z1, k=5) > 0.85
        assert continuity(Z0, Z1, k=5) > 0.85

    def test_returns_in_unit_interval(self):
        Z0 = _unit_normal(40, 8, seed=2)
        Z1 = _unit_normal(40, 8, seed=3)
        for k in (1, 3, 8):
            t = trustworthiness(Z0, Z1, k=k)
            c = continuity(Z0, Z1, k=k)
            assert 0.0 <= t <= 1.0
            assert 0.0 <= c <= 1.0

    def test_too_few_samples_raises(self):
        Z = _unit_normal(5, 4, seed=0)
        with pytest.raises(ValueError, match="need at least"):
            trustworthiness(Z, Z, k=10)


class TestRegistration:
    def test_register_local_structure_adds_entries(self):
        registry = MetricRegistry()
        register_local_structure_metrics(registry, k=5, nps_curve_ks=(1, 5, 10))
        names = set(registry.list_metrics())
        assert {"trustworthiness", "continuity"}.issubset(names)
        assert {"nps_at_1", "nps_at_5", "nps_at_10"}.issubset(names)

    def test_registered_trustworthiness_computes(self):
        registry = MetricRegistry()
        register_local_structure_metrics(registry, k=5)
        Z = _unit_normal(40, 16, seed=0)
        assert registry.compute("trustworthiness", Z, Z) == pytest.approx(1.0)
        assert registry.compute("continuity", Z, Z) == pytest.approx(1.0)

    def test_nps_curve_returns_per_k(self):
        registry = MetricRegistry()
        register_local_structure_metrics(registry, nps_curve_ks=(1, 5, 10))
        Z0 = _unit_normal(50, 16, seed=0)
        Z1 = Z0 + 0.05 * np.random.default_rng(1).standard_normal(Z0.shape).astype(np.float32)
        Z1 = Z1 / np.linalg.norm(Z1, axis=1, keepdims=True)
        v1 = registry.compute("nps_at_1", Z0, Z1)
        v5 = registry.compute("nps_at_5", Z0, Z1)
        v10 = registry.compute("nps_at_10", Z0, Z1)
        # All bounded in [0, 1].
        for v in (v1, v5, v10):
            assert 0.0 <= v <= 1.0
