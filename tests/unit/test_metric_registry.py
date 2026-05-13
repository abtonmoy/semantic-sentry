"""Tests for MetricRegistry."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from semantic_sentry.exceptions import MetricRegistrationError
from semantic_sentry.metrics.registry import MetricRegistry, get_metric_registry


class TestMetricRegistrySingleton:
    """Test MetricRegistry singleton behavior."""

    def test_singleton_same_instance(self):
        """Multiple calls return same instance."""
        registry1 = MetricRegistry()
        registry2 = MetricRegistry()

        assert registry1 is registry2

    def test_get_metric_registry_same_instance(self):
        """get_metric_registry returns same instance."""
        registry1 = get_metric_registry()
        registry2 = get_metric_registry()

        assert registry1 is registry2

    def test_singleton_across_calls(self):
        """Singleton works across multiple instantiations."""
        registries = [MetricRegistry() for _ in range(5)]

        assert all(r is registries[0] for r in registries)


class TestMetricRegistryBuiltins:
    """Test built-in metric registration."""

    def test_builtins_registered_on_init(self):
        """Built-in metrics are registered on initialization."""
        registry = MetricRegistry()
        metrics = registry.list_metrics()

        assert "cka" in metrics
        assert "nps" in metrics
        assert "isotropy_delta" in metrics

    def test_builtin_cka_computes(self):
        """CKA metric computes correctly."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        result = registry.compute("cka", Z0, Z0)

        assert abs(result - 1.0) < 1e-5

    def test_builtin_nps_computes(self):
        """NPS metric computes correctly."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        result = registry.compute("nps", Z0, Z0)

        assert abs(result - 1.0) < 1e-5

    def test_builtin_isotropy_delta_computes(self):
        """Isotropy delta metric computes correctly."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        result = registry.compute("isotropy_delta", Z0, Z0)

        assert abs(result) < 1e-5


class TestMetricRegistryCustom:
    """Test custom metric registration."""

    def test_register_valid_metric(self):
        """Valid custom metric can be registered."""
        registry = MetricRegistry()

        def custom_metric(Z0, Z1):
            return float(np.mean(Z0) - np.mean(Z1))

        registry.register(
            "custom",
            custom_metric,
            range=(-10.0, 10.0),
            description="Test custom metric"
        )

        assert "custom" in registry.list_metrics()

    def test_register_non_deterministic_fails(self):
        """Non-deterministic metric fails registration."""
        registry = MetricRegistry()

        def non_deterministic(Z0, Z1):
            import random
            return random.random()

        with pytest.raises(MetricRegistrationError, match="not deterministic"):
            registry.register("bad", non_deterministic)

    def test_register_non_float_return_fails(self):
        """Metric returning non-float fails registration."""
        registry = MetricRegistry()

        def bad_return_type(Z0, Z1):
            return "string"

        with pytest.raises(MetricRegistrationError, match="must return float"):
            registry.register("bad", bad_return_type)

    def test_register_raising_exception_fails(self):
        """Metric raising exception fails registration."""
        registry = MetricRegistry()

        def failing(Z0, Z1):
            raise ValueError("test error")

        with pytest.raises(MetricRegistrationError, match="raised exception"):
            registry.register("bad", failing)

    def test_registered_metric_computes(self):
        """Registered custom metric can be computed."""
        registry = MetricRegistry()

        def custom_metric(Z0, Z1):
            return 0.5

        registry.register("my_metric", custom_metric)

        Z0 = np.random.randn(50, 32).astype(np.float32)
        Z1 = np.random.randn(50, 32).astype(np.float32)

        result = registry.compute("my_metric", Z0, Z1)
        assert result == 0.5

    def test_range_validation(self):
        """Range validation works for metrics with range."""
        registry = MetricRegistry()

        def custom_metric(Z0, Z1):
            return 5.0  # Out of range

        registry.register("ranged", custom_metric, range=(0.0, 1.0))

        Z0 = np.random.randn(50, 32).astype(np.float32)
        Z1 = np.random.randn(50, 32).astype(np.float32)

        with pytest.raises(ValueError, match="outside range"):
            registry.compute("ranged", Z0, Z1)


class TestMetricRegistryComputeAll:
    """Test compute_all functionality."""

    def test_compute_all_returns_all_metrics(self):
        """compute_all returns all registered metrics."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        results = registry.compute_all(Z0, Z0)

        assert "cka" in results
        assert "nps" in results
        assert "isotropy_delta" in results

    def test_compute_all_with_subset(self):
        """compute_all can compute subset of metrics."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        results = registry.compute_all(Z0, Z0, metric_names=["cka", "nps"])

        assert "cka" in results
        assert "nps" in results
        assert "isotropy_delta" not in results


class TestMetricRegistryThreadSafety:
    """Test thread-safe registration."""

    def test_concurrent_registration(self):
        """Concurrent registration does not corrupt state."""
        registry = MetricRegistry()
        errors = []

        def register_metric(i):
            try:
                def metric_fn(Z0, Z1):
                    return float(i)
                registry.register(f"metric_{i}", metric_fn, range=(0.0, 100.0))
            except Exception as e:
                errors.append(e)

        # Concurrent registration
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(register_metric, range(10)))

        assert len(errors) == 0, f"Errors during concurrent registration: {errors}"

        # Verify all registered
        metrics = registry.list_metrics()
        for i in range(10):
            assert f"metric_{i}" in metrics

    def test_thread_safe_compute(self):
        """Concurrent compute does not corrupt state."""
        registry = MetricRegistry()

        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)

        results = []

        def compute_metric(i):
            result = registry.compute("cka", Z0, Z0)
            results.append(result)

        # Concurrent computation
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(compute_metric, range(10)))

        # All should be 1.0 (same matrix)
        assert all(abs(r - 1.0) < 1e-5 for r in results)


class TestMetricRegistryInfo:
    """Test metric info retrieval."""

    def test_get_info_returns_entry(self):
        """get_info returns MetricEntry."""
        registry = MetricRegistry()

        entry = registry.get_info("cka")

        assert entry.description is not None
        assert entry.range == (0.0, 1.0)

    def test_get_info_not_found(self):
        """get_info raises KeyError for unknown metric."""
        registry = MetricRegistry()

        with pytest.raises(KeyError):
            registry.get_info("unknown_metric")


class TestMetricRegistryUnregister:
    """Test metric unregistration."""

    def test_unregister_removes_metric(self):
        """unregister removes metric from registry."""
        registry = MetricRegistry()

        def custom(Z0, Z1):
            return 0.5

        registry.register("temp", custom)
        assert "temp" in registry.list_metrics()

        registry.unregister("temp")
        assert "temp" not in registry.list_metrics()

    def test_unregister_not_found(self):
        """unregister raises KeyError for unknown metric."""
        registry = MetricRegistry()

        with pytest.raises(KeyError):
            registry.unregister("unknown_metric")
