"""Phase 3: Concurrency and thread safety tests."""

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from semantic_sentry.metrics.registry import MetricRegistry


@pytest.mark.stress
class TestConcurrency:
    """Concurrency and thread safety stress tests."""

    @pytest.mark.timeout(30)
    def test_thr_001_concurrent_register(self):
        """THR-001: 100 threads registering custom metrics must not corrupt state."""
        registry = MetricRegistry()
        errors = []

        def register_metric(i):
            try:
                registry.register(
                    name=f"test_metric_{i}",
                    fn=lambda Z0, Z1: float(np.mean(Z0 - Z1)),
                    description=f"Test metric {i}",
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as pool:
            futures = [pool.submit(register_metric, i) for i in range(100)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Registration errors: {errors}"

    @pytest.mark.timeout(30)
    def test_thr_002_concurrent_compute(self, make_drifted_pair):
        """THR-002: 100 concurrent compute_all must return consistent results."""
        Z0, Z1 = make_drifted_pair(100, 64, noise_scale=0.2)
        registry = MetricRegistry()

        results = []
        def compute():
            r = registry.compute_all(Z0, Z1)
            results.append(r)

        threads = [threading.Thread(target=compute) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # All results must be identical
        for r in results[1:]:
            for key in results[0]:
                assert r[key] == pytest.approx(results[0][key], abs=1e-6), \
                    f"Inconsistent result for {key}: {r[key]} vs {results[0][key]}"

    @pytest.mark.timeout(30)
    def test_thr_003_register_during_compute(self, make_drifted_pair):
        """THR-003: Concurrent registration and computation must not deadlock."""
        Z0, Z1 = make_drifted_pair(100, 64, noise_scale=0.2)
        registry = MetricRegistry()
        completed = threading.Event()

        def compute_loop():
            for _ in range(50):
                registry.compute_all(Z0, Z1)
            completed.set()

        def register_loop():
            for i in range(50):
                # may fail if name exists — that's the race we're stressing
                with contextlib.suppress(Exception):
                    registry.register(
                        name=f"dyn_{i}",
                        fn=lambda Z0, Z1, _i=i: float(_i),
                    )

        t1 = threading.Thread(target=compute_loop)
        t2 = threading.Thread(target=register_loop)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert completed.is_set(), "Compute loop did not complete (possible deadlock)"

    @pytest.mark.timeout(30)
    def test_thr_004_parallel_metric_computation(self, make_drifted_pair):
        """THR-004: Parallel metric computation should return same results."""
        Z0, Z1 = make_drifted_pair(500, 128, noise_scale=0.2)
        registry = MetricRegistry()

        # Sequential
        seq_result = registry.compute_all(Z0, Z1, parallel=False)

        # Parallel
        par_result = registry.compute_all(Z0, Z1, parallel=True)

        # Results should be same
        for key in seq_result:
            assert seq_result[key] == pytest.approx(par_result[key], abs=1e-6)
