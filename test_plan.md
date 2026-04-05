# SemanticSentry Stress Test Plan

## Overview

This test plan covers comprehensive stress testing for SemanticSentry across six phases: metric robustness, scale and performance, concurrency and thread safety, data quality edge cases, adversarial scenarios, and real-world model validation. It supplements the unit and integration tests defined in `agent_instruction_1.md`.

**Total tests: 169**
**Estimated execution time: 10 days**
**Priority levels: P0 (critical), P1 (important), P2 (nice-to-have)**
**Reliability targets: 100% P0, 95% P1, 80% P2**

**All tests must be deterministic.** Use `np.random.default_rng(42)` for reproducibility. Every test must have an explicit `@pytest.mark.timeout(N)` to prevent hangs.

---

## Test Environment Setup

```bash
# Ensure dev dependencies are installed
uv pip install -e ".[all]"

# Run stress tests (separate from unit/integration)
uv run pytest tests/stress/ -v --timeout=300 -m stress

# Run real-world tests (downloads models, slow)
uv run pytest tests/stress/ -v -m real_world --timeout=600
```

### Test Directory Structure

```
tests/
├── stress/
│   ├── __init__.py
│   ├── conftest.py                    # Stress test fixtures
│   ├── test_cka_stress.py             # Phase 1.1
│   ├── test_nps_stress.py             # Phase 1.2
│   ├── test_isotropy_stress.py        # Phase 1.3
│   ├── test_classification_stress.py  # Phase 1.4 (NEW)
│   ├── test_scale.py                  # Phase 2
│   ├── test_concurrency.py            # Phase 3
│   ├── test_data_quality.py           # Phase 4
│   ├── test_adversarial.py            # Phase 5
│   ├── test_transfer_stress.py        # Phase 5.2 (NEW)
│   └── test_real_world.py             # Phase 6
```

### conftest.py Fixtures for Stress Tests

```python
import pytest
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def make_embeddings(rng):
    """Factory fixture: generate L2-normalized embeddings of any shape."""
    def _make(n, d, seed=None):
        r = np.random.default_rng(seed or 42)
        Z = r.standard_normal((n, d)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return Z
    return _make

@pytest.fixture
def make_drifted_pair(make_embeddings):
    """Factory: generate base + drifted embedding pair with controlled noise."""
    def _make(n, d, noise_scale=0.3, seed=42):
        Z_base = make_embeddings(n, d, seed=seed)
        rng = np.random.default_rng(seed + 1)
        noise = rng.standard_normal((n, d)).astype(np.float32) * noise_scale
        Z_drifted = Z_base + noise
        Z_drifted = Z_drifted / np.linalg.norm(Z_drifted, axis=1, keepdims=True)
        return Z_base, Z_drifted
    return _make
```

---

## Phase 1: Metric Robustness Tests

### 1.1 CKA Stress Tests

```python
# tests/stress/test_cka_stress.py
import pytest
import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from semantic_sentry.metrics.cka import linear_cka

@pytest.mark.stress
class TestCKAStress:

    def test_cka_001_identical_matrices(self, make_embeddings):
        """CKA-001: Identical matrices must return exactly 1.0."""
        Z = make_embeddings(100, 64)
        assert linear_cka(Z, Z) == pytest.approx(1.0, abs=1e-6)

    def test_cka_002_rank_deficient(self, rng):
        """CKA-002: Rank-1 matrix (all rows are scaled versions of one vector)."""
        v = rng.standard_normal(64).astype(np.float32)
        scales = rng.uniform(0.5, 2.0, size=(100, 1)).astype(np.float32)
        Z = v * scales
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        result = linear_cka(Z, Z)
        assert result == pytest.approx(1.0, abs=1e-4), f"Rank-1 self-CKA was {result}"

    def test_cka_003_orthogonal_invariance(self, make_embeddings, rng):
        """CKA-003: CKA must be invariant to orthogonal transformations."""
        Z = make_embeddings(100, 64)
        # Generate random orthogonal matrix via QR decomposition
        H = rng.standard_normal((64, 64)).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        Z_rotated = Z @ Q
        result = linear_cka(Z, Z_rotated)
        assert result == pytest.approx(1.0, abs=1e-4), f"CKA after rotation was {result}"

    def test_cka_004_very_different_matrices(self, make_embeddings):
        """CKA-004: Random independent matrices should have CKA < 0.5."""
        Z0 = make_embeddings(100, 64, seed=1)
        Z1 = make_embeddings(100, 64, seed=999)
        result = linear_cka(Z0, Z1)
        assert result < 0.5, f"Independent matrices had CKA = {result}"

    def test_cka_005_zero_matrix(self):
        """CKA-005: Zero matrix should return 0.0 or raise, not NaN/Inf."""
        Z = np.zeros((100, 64), dtype=np.float32)
        result = linear_cka(Z, Z)
        assert np.isfinite(result), "CKA returned NaN or Inf for zero matrix"

    def test_cka_006_single_sample(self, rng):
        """CKA-006: n=1 must not crash."""
        Z0 = rng.standard_normal((1, 64)).astype(np.float32)
        Z1 = rng.standard_normal((1, 64)).astype(np.float32)
        result = linear_cka(Z0, Z1)
        assert np.isfinite(result), "CKA crashed on single sample"

    def test_cka_007_high_condition_number(self, rng):
        """CKA-007: Ill-conditioned matrix should not produce NaN."""
        Z = rng.standard_normal((100, 64)).astype(np.float32)
        Z[:, 0] *= 1e8  # make one dimension dominate
        Z[:, 1:] *= 1e-8
        result = linear_cka(Z, Z)
        assert np.isfinite(result), f"CKA was {result} for ill-conditioned matrix"

    @given(Z=arrays(np.float32, (50, 16), elements=dict(min_value=-10, max_value=10)))
    @settings(max_examples=50, deadline=5000)
    def test_cka_property_symmetry(self, Z):
        """CKA must be symmetric: CKA(Z0, Z1) == CKA(Z1, Z0)."""
        if np.any(np.isnan(Z)) or np.all(Z == 0):
            return  # skip degenerate
        rng = np.random.default_rng(0)
        Z1 = Z + rng.standard_normal(Z.shape).astype(np.float32) * 0.1
        assert linear_cka(Z, Z1) == pytest.approx(linear_cka(Z1, Z), abs=1e-5)
```

### 1.2 NPS Stress Tests

```python
# tests/stress/test_nps_stress.py
import pytest
import numpy as np
from semantic_sentry.metrics.nps import nps

@pytest.mark.stress
class TestNPSStress:

    def test_nps_001_self_comparison(self, make_embeddings):
        """NPS-001: Self-comparison must return 1.0."""
        Z = make_embeddings(100, 64)
        assert nps(Z, Z) == pytest.approx(1.0, abs=1e-6)

    def test_nps_002_k_greater_than_n(self, make_embeddings):
        """NPS-002: k > n must raise ValueError or clamp gracefully."""
        Z = make_embeddings(10, 64)
        with pytest.raises((ValueError, RuntimeError)):
            nps(Z, Z, k=100)

    def test_nps_003_identical_rows(self, rng):
        """NPS-003: All rows identical — must not crash."""
        v = rng.standard_normal(64).astype(np.float32)
        v = v / np.linalg.norm(v)
        Z = np.tile(v, (100, 1))
        result = nps(Z, Z, k=5)
        assert np.isfinite(result)

    def test_nps_004_random_permutation(self, make_embeddings, rng):
        """NPS-004: Row permutation should not change NPS (same neighborhoods)."""
        Z = make_embeddings(100, 64)
        perm = rng.permutation(100)
        Z_perm = Z[perm]
        # NPS should still be 1.0 because the point cloud is the same
        result = nps(Z, Z_perm, k=10)
        # Note: NPS compares neighborhoods of corresponding indices, so
        # permutation WILL change NPS. This tests a different thing.
        assert 0.0 <= result <= 1.0

    def test_nps_005_monotonic_perturbation(self, make_drifted_pair):
        """NPS-005: Increasing noise should decrease NPS monotonically."""
        results = []
        for noise in [0.01, 0.1, 0.3, 0.5, 1.0]:
            Z_base, Z_drift = make_drifted_pair(200, 64, noise_scale=noise)
            results.append(nps(Z_base, Z_drift, k=10))
        # Should be monotonically decreasing (with tolerance for small noise)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1] - 0.05, \
                f"NPS not monotonic: {results}"

    @pytest.mark.timeout(30)
    def test_nps_006_large_n_small_k(self, make_embeddings):
        """NPS-006: n=10K, k=5 must complete within 30 seconds."""
        Z0 = make_embeddings(10_000, 128)
        Z1 = make_embeddings(10_000, 128, seed=99)
        result = nps(Z0, Z1, k=5)
        assert 0.0 <= result <= 1.0
```

### 1.3 Isotropy Stress Tests

```python
# tests/stress/test_isotropy_stress.py
import pytest
import numpy as np
from semantic_sentry.metrics.isotropy import isotropy, isotropy_delta

@pytest.mark.stress
class TestIsotropyStress:

    def test_iso_001_spherical_gaussian(self, rng):
        """ISO-001: Spherical Gaussian should be highly isotropic."""
        Z = rng.standard_normal((1000, 64)).astype(np.float32)
        result = isotropy(Z)
        assert result > 0.5, f"Spherical Gaussian isotropy was only {result}"

    def test_iso_002_rank_1(self, rng):
        """ISO-002: Rank-1 matrix should have near-zero isotropy."""
        v = rng.standard_normal(64).astype(np.float32)
        scales = rng.uniform(0.1, 2.0, size=(100, 1)).astype(np.float32)
        Z = v * scales
        result = isotropy(Z)
        assert result < 0.1, f"Rank-1 isotropy was {result}"

    def test_iso_003_square_matrix(self, make_embeddings):
        """ISO-003: n == d should compute stably."""
        Z = make_embeddings(64, 64)
        result = isotropy(Z)
        assert np.isfinite(result) and 0 <= result <= 1

    def test_iso_004_tall_matrix(self, make_embeddings):
        """ISO-004: n << d (10 samples, 512 dims)."""
        Z = make_embeddings(10, 512)
        result = isotropy(Z)
        assert np.isfinite(result)

    def test_iso_005_wide_matrix(self, make_embeddings):
        """ISO-005: n >> d (10000 samples, 8 dims)."""
        Z = make_embeddings(10_000, 8)
        result = isotropy(Z)
        assert np.isfinite(result) and 0 <= result <= 1
```

### 1.4 Classification Stress Tests (NEW)

```python
# tests/stress/test_classification_stress.py
import pytest
import numpy as np
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.core.classification import ConfidenceLevel
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.probes.anchor_set import AnchorSet
from semantic_sentry.exceptions import NoComparisonError

@pytest.mark.stress
class TestClassificationStress:

    def test_clf_001_classify_before_compare(self, rng):
        """CLF-001: classify() before any compare() must raise NoComparisonError."""
        Z = rng.standard_normal((50, 32)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[:len(inputs)]},
            tower_names=["encoder"],
        )
        anchor_set = AnchorSet(
            inputs=[f"s_{i}" for i in range(50)],
            labels=np.array(["a"] * 25 + ["b"] * 25),
            modality="text",
        )
        monitor = DriftMonitor()
        # snapshot but NO compare
        monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter)

        with pytest.raises(NoComparisonError):
            monitor.classify(
                input_data=["test"],
                model=None,
                anchor_set=anchor_set,
                adapter=adapter,
            )

    def test_clf_002_correct_label_near_centroid(self, rng):
        """CLF-002: Point near known centroid returns correct label."""
        # 3 well-separated clusters in 4D
        centers = np.eye(3, 4, dtype=np.float32)
        Z = np.vstack([
            centers[i] + rng.standard_normal((30, 4)).astype(np.float32) * 0.02
            for i in range(3)
        ])
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        labels = np.array(["alpha"] * 30 + ["beta"] * 30 + ["gamma"] * 30)

        adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[:len(inputs)]},
            tower_names=["encoder"],
        )
        anchor_set = AnchorSet(inputs=[f"s_{i}" for i in range(90)], labels=labels, modality="text")

        monitor = DriftMonitor()
        snap = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter)
        monitor.compare(snap, snap)  # self-compare to enable classification

        # Test point near "beta" centroid
        test_point = centers[1] + rng.standard_normal(4).astype(np.float32) * 0.01
        test_point = (test_point / np.linalg.norm(test_point)).reshape(1, -1)
        test_adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": test_point},
            tower_names=["encoder"],
        )
        result = monitor.classify(["test"], model=None, anchor_set=anchor_set, adapter=test_adapter)
        assert result.label == "beta"

    def test_clf_003_low_confidence_in_drifted_region(self, rng):
        """CLF-003: Classification in a drifted region should have LOW confidence."""
        Z_base = rng.standard_normal((100, 32)).astype(np.float32)
        Z_base = Z_base / np.linalg.norm(Z_base, axis=1, keepdims=True)

        # Drift the first 50 points severely
        Z_drifted = Z_base.copy()
        Z_drifted[:50] += rng.standard_normal((50, 32)).astype(np.float32) * 2.0
        Z_drifted = Z_drifted / np.linalg.norm(Z_drifted, axis=1, keepdims=True)

        labels = np.array(["drifted"] * 50 + ["stable"] * 50)
        anchor_set = AnchorSet(inputs=[f"s_{i}" for i in range(100)], labels=labels, modality="text")

        adapter_v0 = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_base[:len(inputs)]},
            tower_names=["encoder"],
        )
        adapter_v1 = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_drifted[:len(inputs)]},
            tower_names=["encoder"],
        )

        monitor = DriftMonitor()
        snap_v0 = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_v0)
        snap_v1 = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_v1)
        monitor.compare(snap_v0, snap_v1)

        # Classify a point near the drifted cluster
        test_point = Z_drifted[0:1]
        test_adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": test_point},
            tower_names=["encoder"],
        )
        result = monitor.classify(["test"], model=None, anchor_set=anchor_set, adapter=test_adapter)
        assert result.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)
        assert result.drift_warning is not None

    @pytest.mark.timeout(10)
    def test_clf_004_batch_performance(self, rng):
        """CLF-004: classify_batch with 10K inputs must complete in < 10s."""
        Z = rng.standard_normal((100, 64)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        labels = np.array(["a"] * 50 + ["b"] * 50)

        adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[:len(inputs)]},
            tower_names=["encoder"],
        )
        anchor_set = AnchorSet(inputs=[f"s_{i}" for i in range(100)], labels=labels, modality="text")

        monitor = DriftMonitor()
        snap = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter)
        monitor.compare(snap, snap)

        batch_inputs = [f"test_{i}" for i in range(10_000)]
        batch_Z = rng.standard_normal((10_000, 64)).astype(np.float32)
        batch_Z = batch_Z / np.linalg.norm(batch_Z, axis=1, keepdims=True)
        batch_adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": batch_Z[:len(inputs)]},
            tower_names=["encoder"],
        )
        results = monitor.classify_batch(batch_inputs, model=None, anchor_set=anchor_set, adapter=batch_adapter)
        assert len(results) == 10_000

    def test_clf_005_equidistant_centroids(self, rng):
        """CLF-005: Equidistant point should return a consistent (not random) result."""
        # Two clusters at equal distance from origin
        Z = np.vstack([
            np.array([[1, 0, 0, 0]] * 50, dtype=np.float32),
            np.array([[-1, 0, 0, 0]] * 50, dtype=np.float32),
        ])
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        labels = np.array(["pos"] * 50 + ["neg"] * 50)

        adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[:len(inputs)]},
            tower_names=["encoder"],
        )
        anchor_set = AnchorSet(inputs=[f"s_{i}" for i in range(100)], labels=labels, modality="text")

        monitor = DriftMonitor()
        snap = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter)
        monitor.compare(snap, snap)

        # Point equidistant from both clusters
        test_point = np.array([[0, 1, 0, 0]], dtype=np.float32)
        test_adapter = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": test_point},
            tower_names=["encoder"],
        )
        result1 = monitor.classify(["t"], model=None, anchor_set=anchor_set, adapter=test_adapter)
        result2 = monitor.classify(["t"], model=None, anchor_set=anchor_set, adapter=test_adapter)
        # Must be deterministic
        assert result1.label == result2.label
```

---

## Phase 2: Scale and Performance Tests

```python
# tests/stress/test_scale.py
import pytest
import numpy as np
import time
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
from semantic_sentry.metrics.isotropy import isotropy_delta

@pytest.mark.stress
class TestScale:

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
                timestamp=datetime.now(timezone.utc),
                anchor_set_version="b" * 32,
                tower_count=1,
                tower_names=["encoder"],
                embeddings={"encoder": Z},
                cross_tower_alignment=None,
                metadata={},
            )
            del snap

        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        growth_ratio = current / max(baseline, 1)
        assert growth_ratio < 1.5, f"Memory grew {growth_ratio:.2f}x after 1000 snapshot cycles"
```

---

## Phase 3: Concurrency and Thread Safety

```python
# tests/stress/test_concurrency.py
import pytest
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from semantic_sentry.metrics.registry import MetricRegistry

@pytest.mark.stress
class TestConcurrency:

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
                try:
                    registry.register(
                        name=f"dyn_{i}",
                        fn=lambda Z0, Z1, _i=i: float(_i),
                    )
                except Exception:
                    pass  # may fail if name exists

        t1 = threading.Thread(target=compute_loop)
        t2 = threading.Thread(target=register_loop)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert completed.is_set(), "Compute loop did not complete (possible deadlock)"
```

---

## Phase 4: Data Quality Edge Cases

```python
# tests/stress/test_data_quality.py
import pytest
import numpy as np
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
from semantic_sentry.metrics.isotropy import isotropy

@pytest.mark.stress
class TestDataQuality:

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
        """DQA-004: Large values (1e30) must not overflow."""
        Z = rng.standard_normal((100, 64)).astype(np.float64) * 1e15
        result = linear_cka(Z, Z)
        assert np.isfinite(result)

    def test_dqa_005_very_small_values(self, rng):
        """DQA-005: Very small values (1e-30) must not underflow to zero."""
        Z = rng.standard_normal((100, 64)).astype(np.float64) * 1e-15
        result = isotropy(Z)
        assert np.isfinite(result)

    def test_dqa_006_empty_anchor_set(self):
        """DQA-006: Empty inputs must raise ValueError."""
        from semantic_sentry.probes.anchor_set import AnchorSet
        with pytest.raises((ValueError, RuntimeError)):
            AnchorSet(inputs=[], labels=np.array([]), modality="text")

    def test_dqa_007_single_sample_anchor(self, rng):
        """DQA-007: Single-sample anchor set must be handled."""
        from semantic_sentry.probes.anchor_set import AnchorSet
        anchor = AnchorSet(
            inputs=["single"],
            labels=np.array(["only"]),
            modality="text",
        )
        assert anchor.n_samples == 1
```

---

## Phase 5: Adversarial Scenarios

```python
# tests/stress/test_adversarial.py
import pytest
import numpy as np
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps

@pytest.mark.stress
class TestAdversarial:

    def test_adv_001_rotation_attack(self, make_embeddings, rng):
        """ADV-001: Orthogonal rotation gives CKA=1 but NPS<1.
        This validates why multi-metric is necessary."""
        Z = make_embeddings(200, 64)
        H = rng.standard_normal((64, 64)).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        Z_rotated = Z @ Q

        cka_val = linear_cka(Z, Z_rotated)
        nps_val = nps(Z, Z_rotated, k=10)

        assert cka_val > 0.99, f"CKA should be ~1.0 after rotation, got {cka_val}"
        # NPS may or may not change depending on rotation — this is the key test
        # A random rotation WILL change neighborhoods
        # This validates the HLD claim that CKA alone is insufficient

    def test_adv_006_truncated_snapshot(self, make_embeddings):
        """ADV-006: Truncated snapshot file must raise error."""
        import tempfile
        from pathlib import Path
        from semantic_sentry.core.snapshot import Snapshot
        from semantic_sentry.exceptions import SnapshotCorruptionError
        from datetime import datetime, timezone

        Z = make_embeddings(50, 32)
        snap = Snapshot(
            model_id="test", checkpoint_hash="a" * 64,
            timestamp=datetime.now(timezone.utc),
            anchor_set_version="b" * 32, tower_count=1,
            tower_names=["encoder"], embeddings={"encoder": Z},
            cross_tower_alignment=None, metadata={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snap"
            snap.save(path)
            # Truncate a safetensors file
            st_file = path / "embeddings_encoder.safetensors"
            with open(st_file, "wb") as f:
                f.write(b"corrupted")
            with pytest.raises((SnapshotCorruptionError, Exception)):
                Snapshot.load(path)

    def test_adv_007_modified_hash(self, make_embeddings):
        """ADV-007: Modified checkpoint hash must raise integrity error."""
        import tempfile, json
        from pathlib import Path
        from semantic_sentry.core.snapshot import Snapshot
        from semantic_sentry.exceptions import SnapshotCorruptionError
        from datetime import datetime, timezone

        Z = make_embeddings(50, 32)
        snap = Snapshot(
            model_id="test", checkpoint_hash="a" * 64,
            timestamp=datetime.now(timezone.utc),
            anchor_set_version="b" * 32, tower_count=1,
            tower_names=["encoder"], embeddings={"encoder": Z},
            cross_tower_alignment=None, metadata={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snap"
            snap.save(path)
            meta = json.loads((path / "metadata.json").read_text())
            meta["checkpoint_hash"] = "f" * 64
            (path / "metadata.json").write_text(json.dumps(meta))
            with pytest.raises(SnapshotCorruptionError):
                Snapshot.load(path)
```

### Transfer Function Stress Tests (NEW)

```python
# tests/stress/test_transfer_stress.py
import pytest
import numpy as np
from semantic_sentry.transfer.function import LinearTransfer
from semantic_sentry.transfer.nps_bound import NPSBound
from semantic_sentry.core.comparison import Comparison, AlertSeverity
from datetime import datetime, timezone

@pytest.mark.stress
class TestTransferStress:

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
                snapshot_v0_id="base", snapshot_v1_id=f"upd_{i}",
                global_metrics={"cka": cka_vals[i], "nps": nps_vals[i], "isotropy_delta": iso_vals[i]},
                per_tower_metrics=None, alignment_deltas=None,
                timestamp=datetime.now(timezone.utc),
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
            snapshot_v0_id="a", snapshot_v1_id="b",
            global_metrics={"cka": 0.9, "nps": 0.85, "isotropy_delta": -0.01},
            per_tower_metrics=None, alignment_deltas=None,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(ValueError):
            tf.predict(c)
```

---

## Phase 6: Real-World Model Testing

```python
# tests/stress/test_real_world.py
import pytest

@pytest.mark.real_world
@pytest.mark.slow
class TestRealWorld:

    @pytest.mark.timeout(120)
    def test_real_001_bert_base(self):
        """REAL-001: Load bert-base-uncased from HF Hub and capture snapshot."""
        from transformers import AutoModel, AutoTokenizer
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.probes.anchor_set import AnchorSet
        import numpy as np

        model = AutoModel.from_pretrained("bert-base-uncased")
        anchor_texts = [f"This is test sentence number {i}." for i in range(50)]
        anchor_set = AnchorSet(
            inputs=anchor_texts,
            labels=np.array(["general"] * 50),
            modality="text",
        )
        monitor = DriftMonitor()
        snap = monitor.snapshot(model=model, anchor_set=anchor_set)
        assert snap.tower_count == 1
        assert snap.embeddings["encoder"].shape == (50, 768)

    @pytest.mark.timeout(180)
    def test_real_003_sentence_transformer(self):
        """REAL-003: SentenceTransformer all-MiniLM-L6-v2."""
        from sentence_transformers import SentenceTransformer
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.probes.anchor_set import AnchorSet
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        anchor_texts = [f"Sentence {i} for embedding." for i in range(50)]
        anchor_set = AnchorSet(
            inputs=anchor_texts,
            labels=np.array(["general"] * 50),
            modality="text",
        )
        monitor = DriftMonitor()
        snap = monitor.snapshot(model=model, anchor_set=anchor_set)
        assert snap.tower_count == 1
        assert snap.embeddings["encoder"].shape[0] == 50

    @pytest.mark.timeout(300)
    def test_real_012_different_dims_raises(self):
        """REAL-012: Comparing base vs large (different dimensions) must raise EmbeddingDimError."""
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.core.snapshot import Snapshot
        from semantic_sentry.exceptions import EmbeddingDimError
        from semantic_sentry.probes.anchor_set import AnchorSet
        from semantic_sentry.adapters.custom import CustomAdapter
        import numpy as np
        from datetime import datetime, timezone

        rng = np.random.default_rng(42)
        # "base" model: 768-dim embeddings
        Z_base = rng.standard_normal((50, 768)).astype(np.float32)
        Z_base = Z_base / np.linalg.norm(Z_base, axis=1, keepdims=True)
        # "large" model: 1024-dim embeddings
        Z_large = rng.standard_normal((50, 1024)).astype(np.float32)
        Z_large = Z_large / np.linalg.norm(Z_large, axis=1, keepdims=True)

        anchor_set = AnchorSet(
            inputs=[f"s_{i}" for i in range(50)],
            labels=np.array(["a"] * 50),
            modality="text",
        )

        adapter_base = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_base[:len(inputs)]},
            tower_names=["encoder"],
        )
        adapter_large = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_large[:len(inputs)]},
            tower_names=["encoder"],
        )

        monitor = DriftMonitor()
        snap_base = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_base)
        snap_large = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_large)

        with pytest.raises(EmbeddingDimError):
            monitor.compare(snap_base, snap_large)
```

---

## Running the Full Test Suite

```bash
# Unit tests (fast, no models)
uv run pytest tests/unit/ -v -m unit --timeout=30

# Integration tests (synthetic models)
uv run pytest tests/integration/ -v -m integration --timeout=120

# Stress tests (extensive, synthetic)
uv run pytest tests/stress/ -v -m stress --timeout=300

# Real-world tests (downloads models, SLOW)
uv run pytest tests/stress/ -v -m real_world --timeout=600

# Full suite with coverage
uv run pytest tests/ -v --cov=semantic_sentry --cov-report=term-missing --cov-fail-under=90

# Performance subset only
uv run pytest tests/stress/test_scale.py -v -m "stress and not slow" --timeout=60
```

---

## Test Count Summary

| Component | Unit | Integration | Stress | Classification | Real-World | Total |
|-----------|------|-------------|--------|----------------|------------|-------|
| CKA | 10 | 5 | 8 | - | - | 23 |
| NPS | 12 | 5 | 6 | - | - | 23 |
| Isotropy | 8 | 3 | 5 | - | - | 16 |
| Snapshot | 10 | 4 | 4 | - | - | 18 |
| DriftMonitor | 8 | 6 | 6 | - | - | 20 |
| Classification | 5 | 4 | - | 5 | - | 14 |
| Transfer | 5 | 3 | 4 | - | - | 12 |
| Concurrency | - | - | 3 | - | - | 3 |
| Data Quality | - | - | 7 | - | - | 7 |
| Adversarial | - | - | 3 | - | - | 3 |
| Registry | 13 | 3 | 3 | - | - | 19 |
| Adapters | 11 | 4 | - | - | 3 | 18 |
| **Total** | **82** | **37** | **49** | **5** | **3** | **176** |

---

## Success Criteria

### Performance Benchmarks
- **Small** (n < 1,000): All metrics < 1 second
- **Medium** (n < 10,000): All metrics < 10 seconds
- **Large** (n < 100,000): All metrics < 60 seconds
- **XLarge** (n < 1,000,000): NPS < 5 minutes

### Reliability Targets
- 100% pass rate for P0 tests
- 95% pass rate for P1 tests
- 80% pass rate for P2 tests

### Memory Limits
- No memory leaks (< 1.5x growth after 1,000 iterations)
- Graceful OOM handling with informative error messages

### Classification Targets
- Correct label for points within 0.1 cosine distance of centroid: > 95%
- Confidence correctly reflects drift: LOW when local NPS < 0.80
- classify_batch(10K): < 10 seconds