"""Tests for the params-dict extension to MetricRegistry (lib_enhancement G1, G2)."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.exceptions import MetricRegistrationError
from semantic_sentry.metrics.registry import MetricRegistry


class TestG1Params:
    def test_register_with_params_passes_them_at_compute_time(self):
        registry = MetricRegistry()

        def my_metric(Z0, Z1, k):
            return float(k)

        registry.register("kp", my_metric, range=(0.0, 100.0), params={"k": 7})

        Z = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
        assert registry.compute("kp", Z, Z) == 7.0

    def test_compute_all_uses_per_entry_params(self):
        registry = MetricRegistry()

        def m(Z0, Z1, k):
            return float(k)

        registry.register("a", m, range=(0.0, 100.0), params={"k": 1})
        registry.register("b", m, range=(0.0, 100.0), params={"k": 9})

        Z = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
        results = registry.compute_all(Z, Z, metric_names=["a", "b"], parallel=False)
        assert results == {"a": 1.0, "b": 9.0}

    def test_builtins_still_work_with_empty_params(self):
        """Built-in metrics take no extra params; **{} unpacking is a no-op."""
        registry = MetricRegistry()
        Z0 = np.random.default_rng(0).standard_normal((30, 16)).astype(np.float32)
        Z1 = Z0 + 0.01 * np.random.default_rng(1).standard_normal((30, 16)).astype(np.float32)

        results = registry.compute_all(Z0, Z1, parallel=False)
        assert set(results.keys()) >= {"cka", "nps", "isotropy_delta"}
        assert 0.0 <= results["cka"] <= 1.0
        assert 0.0 <= results["nps"] <= 1.0

    def test_validate_determinism_threads_params(self):
        """If params change the result, determinism must hold for fixed params."""
        registry = MetricRegistry()

        def m(Z0, Z1, k):
            # Result depends on k, but is deterministic for fixed k.
            return float(k) + float(np.sum(Z0 - Z1))

        registry.register("d", m, params={"k": 3})
        Z = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        assert registry.compute("d", Z, Z) == 3.0

    def test_non_deterministic_metric_rejected(self):
        registry = MetricRegistry()

        def bad(Z0, Z1):
            return float(np.random.default_rng().standard_normal())

        with pytest.raises(MetricRegistrationError, match="not deterministic"):
            registry.register("bad", bad)


class TestG2RegisterAtK:
    def test_register_at_k_creates_one_entry_per_k(self):
        registry = MetricRegistry()

        def m(Z0, Z1, k):
            return float(k) / 100.0

        registry.register_at_k(
            "demo", m, ks=[1, 5, 10], range=(0.0, 1.0), description="demo"
        )
        names = registry.list_metrics()
        assert {"demo_at_1", "demo_at_5", "demo_at_10"}.issubset(set(names))

    def test_register_at_k_bakes_k_into_params(self):
        registry = MetricRegistry()

        def m(Z0, Z1, k):
            return float(k) / 100.0

        registry.register_at_k("demo", m, ks=[1, 25], range=(0.0, 1.0))
        Z = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
        assert registry.compute("demo_at_1", Z, Z) == 0.01
        assert registry.compute("demo_at_25", Z, Z) == 0.25

    def test_register_at_k_merges_extra_params(self):
        registry = MetricRegistry()

        def m(Z0, Z1, k, scale):
            return float(k) * float(scale)

        registry.register_at_k(
            "demo", m, ks=[2, 4], extra_params={"scale": 0.5}, range=(0.0, 10.0)
        )
        Z = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
        assert registry.compute("demo_at_2", Z, Z) == 1.0
        assert registry.compute("demo_at_4", Z, Z) == 2.0
