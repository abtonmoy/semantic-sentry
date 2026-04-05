"""Phase 2: Scale and performance tests."""

import time

import numpy as np
import pytest

from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
from semantic_sentry.metrics.isotropy import isotropy_delta


@pytest.mark.stress
class TestScale:
    """Scale and performance stress tests."""

    @pytest.mark.parametrize("n,d,time_limit", [
        (1_000, 128, 1.0),
        (10_000, 512, 5.0),
        (100_000, 1024, 30.0),
    ])
    def test_scl_metric_performance(self, n, d, time_limit, make_drifted_pair):
        """SCL-001/002/003: Metrics must complete within time budget."""
        Z0, Z1 = make_drifted_pair(n, d, noise_scale=0.1)

        start = time.perf_counter()
        cka_val = linear_cka(Z0, Z1)
        nps_val = nps(Z0, Z1, k=min(10, n - 1))
        iso_val = isotropy_delta(Z0, Z1)
        elapsed = time.perf_counter() - start

        assert elapsed < time_limit, f"n={n}, d={d} took {elapsed:.1f}s (limit {time_limit}s)"
        assert all(np.isfinite(v) for v in [cka_val, nps_val, iso_val])

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_scl_004_million_samples(self, make_drifted_pair):
        """SCL-004: 1M samples x 768 dims within 5 minutes."""
        Z0, Z1 = make_drifted_pair(1_000_000, 768, noise_scale=0.05)

        start = time.perf_counter()
        nps_val = nps(Z0, Z1, k=5)
        elapsed = time.perf_counter() - start

        assert elapsed < 300, f"1M samples took {elapsed:.1f}s"
        assert 0.0 <= nps_val <= 1.0

    def test_memory_leak_snapshot_cycle(self, make_embeddings):
        """MEM: Create/destroy 1000 snapshots, check for memory growth."""
        import tracemalloc
        tracemalloc.start()

        from semantic_sentry.core.snapshot import Snapshot
        from datetime import datetime, timezone

        baseline = tracemalloc.get_traced_memory()[0]

        for i in range(1000):
            Z = make_embeddings(100, 64, seed=i)
            snap = Snapshot(
                model_id="test",
                checkpoint_hash="a" * 64,
                timestamp=datetime.now(timezone.utc).isoformat(),
                anchor_set_version="b" * 32,
                tower_count=1,
                tower_names=("encoder",),
                embeddings={"encoder": Z},
                cross_tower_alignment=None,
                metadata={},
            )
            del snap

        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        growth_ratio = current / max(baseline, 1)
        assert growth_ratio < 1.5, f"Memory grew {growth_ratio:.2f}x after 1000 snapshot cycles"

    @pytest.mark.timeout(60)
    def test_scl_cka_scaling(self, make_embeddings):
        """SCL-CKA: CKA scaling with increasing dimensions."""
        for d in [64, 128, 256, 512, 1024]:
            Z0 = make_embeddings(1000, d)
            Z1 = make_embeddings(1000, d, seed=99)
            start = time.perf_counter()
            result = linear_cka(Z0, Z1)
            elapsed = time.perf_counter() - start
            assert np.isfinite(result)
            assert elapsed < 10.0, f"CKA for d={d} took {elapsed:.2f}s"

    @pytest.mark.timeout(60)
    def test_scl_nps_scaling(self, make_embeddings):
        """SCL-NPS: NPS scaling with increasing samples."""
        for n in [100, 1000, 5000, 10000]:
            Z0 = make_embeddings(n, 128)
            Z1 = make_embeddings(n, 128, seed=99)
            start = time.perf_counter()
            result = nps(Z0, Z1, k=10)
            elapsed = time.perf_counter() - start
            assert np.isfinite(result)
            assert elapsed < 15.0, f"NPS for n={n} took {elapsed:.2f}s"
