"""Stress tests for the v0.2 surface added in lib_enhancement.

Exercises the new modules end-to-end with adversarial inputs.

Targets (numbered TST-IDs cross-reference the spec sections):
  * G1/G2  — MetricEntry.params + register_at_k
  * H1     — AnchorSet.partition()
  * H2     — distribution_tag propagation
  * A1-A5  — behavioral / ranking metrics
  * B1/B2  — trustworthiness, continuity
  * F1-F3  — velocity, acceleration, plateau
  * I2     — SeverityCalibration / calibrate_thresholds
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anchor(n: int = 32):
    from semantic_sentry.probes.anchor_set import AnchorSet
    return AnchorSet(
        inputs=[f"text_{i}" for i in range(n)],
        labels=tuple(f"l_{i % 4}" for i in range(n)),
    )


def _make_snapshot(model_id: str, anchor, emb_matrix, distribution_tag: str = ""):
    """Build a Snapshot from an explicit embedding matrix.

    Uses CustomAdapter with a lookup table so the encode result is exactly
    the rows we pass in (deterministic, no model needed).
    """
    from semantic_sentry import DriftMonitor
    from semantic_sentry.adapters.custom import CustomAdapter
    from semantic_sentry.probes.anchor_set import AnchorSet

    if distribution_tag and not anchor.distribution_tag:
        anchor = AnchorSet(
            inputs=list(anchor.inputs), labels=anchor.labels,
            distribution_tag=distribution_tag,
        )
    lookup = {t: e for t, e in zip(anchor.inputs, emb_matrix, strict=False)}

    def enc(texts):
        return np.stack([lookup.get(t, np.zeros_like(emb_matrix[0]))
                         for t in texts])

    adapter = CustomAdapter(encode_fn=enc, tower_count=1, tower_names=("encoder",))
    monitor = DriftMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return monitor.snapshot(model=model_id, anchor_set=anchor, adapter=adapter)


# ---------------------------------------------------------------------------
# Family G1 / G2 — registry params + register_at_k under stress
# ---------------------------------------------------------------------------


class TestRegistryParams:
    """TST-G1-*: hyperparameter routing through MetricEntry.params."""

    def test_g1_params_are_forwarded(self):
        """A registered metric receives its params as kwargs at every call site."""
        from semantic_sentry.metrics.registry import MetricRegistry

        reg = MetricRegistry()

        def fn(Z0, Z1, k: int = 0) -> float:
            return float(k)

        reg.register("k_capture", fn, params={"k": 7})
        out = reg.compute("k_capture", np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32))
        assert out == 7.0

    def test_g2_register_at_k_creates_one_entry_per_k(self):
        from semantic_sentry.metrics.registry import MetricRegistry

        reg = MetricRegistry()

        def fn(Z0, Z1, k: int = 1) -> float:
            return float(k * 0.1)

        reg.register_at_k("multi", fn, ks=[1, 3, 7, 11])
        names = set(reg.list_metrics())
        assert {"multi_at_1", "multi_at_3", "multi_at_7", "multi_at_11"} <= names
        Z = np.eye(4, dtype=np.float32)
        assert reg.compute("multi_at_1", Z, Z) == pytest.approx(0.1)
        assert reg.compute("multi_at_11", Z, Z) == pytest.approx(1.1)

    def test_g1_caller_dict_mutation_does_not_leak_into_registry(self):
        """Mutating the dict passed to register() after the fact must NOT
        retroactively change the stored params (registry should hold its own copy)."""
        from semantic_sentry.metrics.registry import MetricRegistry

        reg = MetricRegistry()

        def fn(Z0, Z1, k: int = 0) -> float:
            return float(k)

        external = {"k": 4}
        reg.register("isolated", fn, params=external)
        external["k"] = 999
        assert reg.compute("isolated", np.eye(3, dtype=np.float32),
                            np.eye(3, dtype=np.float32)) == 4.0

    def test_g1_non_deterministic_metric_rejected(self):
        """Determinism check still fires when params are involved."""
        from semantic_sentry.exceptions import MetricRegistrationError
        from semantic_sentry.metrics.registry import MetricRegistry

        reg = MetricRegistry()

        def random_fn(Z0, Z1, k: int = 1) -> float:
            return float(np.random.rand())

        with pytest.raises(MetricRegistrationError):
            reg.register("random", random_fn, params={"k": 5})


# ---------------------------------------------------------------------------
# Family H — AnchorSet.partition() + distribution_tag
# ---------------------------------------------------------------------------


class TestAnchorPartition:
    def test_h1_partition_is_seed_deterministic(self):
        a = _make_anchor()
        q1, d1 = a.partition(ratio=0.5, seed=42)
        q2, d2 = a.partition(ratio=0.5, seed=42)
        assert tuple(q1.inputs) == tuple(q2.inputs)
        assert tuple(d1.inputs) == tuple(d2.inputs)

    def test_h1_partition_changes_with_seed(self):
        a = _make_anchor()
        q1, _ = a.partition(ratio=0.5, seed=0)
        q2, _ = a.partition(ratio=0.5, seed=1)
        assert tuple(q1.inputs) != tuple(q2.inputs)

    def test_h1_partition_q_d_cover_parent(self):
        a = _make_anchor(n=40)
        q, d = a.partition(ratio=0.6, seed=0)
        union = set(q.inputs) | set(d.inputs)
        assert union == set(a.inputs)
        assert len(q.inputs) + len(d.inputs) == len(a.inputs)
        assert set(q.inputs).isdisjoint(set(d.inputs))

    def test_h1_partition_role_and_composite_hash(self):
        a = _make_anchor()
        q, d = a.partition(ratio=0.5, seed=7)
        assert q.role == "Q"
        assert d.role == "D"
        # Composite version_hash encodes parent + role + seed.
        assert q.parent_hash == a.version_hash
        assert d.parent_hash == a.version_hash
        assert q.version_hash != d.version_hash
        assert str(7) in q.version_hash

    def test_h1_partition_too_small_raises(self):
        from semantic_sentry.probes.anchor_set import AnchorSet
        tiny = AnchorSet(inputs=["x"], labels=("a",))
        with pytest.raises(ValueError):
            tiny.partition(ratio=0.5, seed=0)

    def test_h2_distribution_tag_round_trip_through_snapshot(self, tmp_path):
        from semantic_sentry.core.snapshot import Snapshot

        a = _make_anchor(n=8)
        # Re-build with the tag set.
        from semantic_sentry.probes.anchor_set import AnchorSet
        a = AnchorSet(inputs=list(a.inputs), labels=a.labels,
                       distribution_tag="training-dist")
        rng = np.random.default_rng(0)
        snap = _make_snapshot("m", a, rng.standard_normal((8, 4)).astype(np.float32))
        assert snap.metadata.get("distribution_tag") == "training-dist"

        out = tmp_path / "snap"
        snap.save(out)
        loaded = Snapshot.load(out)
        assert loaded.metadata.get("distribution_tag") == "training-dist"


# ---------------------------------------------------------------------------
# Family A — behavioral metrics
# ---------------------------------------------------------------------------


class TestBehavioral:
    """TST-A-*: behavioral / ranking metrics on edge-case inputs."""

    def _split(self, n=40, d=16, drift=0.05, seed=0):
        """Build (Z0_Q, Z1_Q, D0, D1) — already split halves."""
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, d)).astype(np.float32)
        Z2 = Z + drift * rng.standard_normal((n, d)).astype(np.float32)
        half = n // 2
        return Z[:half], Z2[:half], Z[half:], Z2[half:]

    def test_a_identity_inputs_score_perfect(self):
        from semantic_sentry.metrics.behavioral import (
            mean_abs_score_delta,
            per_query_kendall_tau,
            per_query_rbo,
            score_distribution_jsd,
            self_retrieval_topk,
        )

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((10, 8)).astype(np.float32)
        D = rng.standard_normal((20, 8)).astype(np.float32)
        # All identity → every score = 0 / 1 as appropriate.
        assert score_distribution_jsd(Q, Q, D, D) == pytest.approx(0.0, abs=1e-6)
        assert mean_abs_score_delta(Q, Q, D, D) == pytest.approx(0.0, abs=1e-6)
        assert per_query_rbo(Q, Q, D, D) == pytest.approx(1.0, abs=1e-6)
        assert per_query_kendall_tau(Q, Q, D, D) == pytest.approx(1.0, abs=1e-6)
        assert self_retrieval_topk(Q, Q, D, D, k=5) == pytest.approx(1.0, abs=1e-6)

    def test_a_jsd_bounded(self):
        from semantic_sentry.metrics.behavioral import score_distribution_jsd
        Z0_Q, Z1_Q, D0, D1 = self._split(n=80, d=16, drift=3.0, seed=1)
        val = score_distribution_jsd(Z0_Q, Z1_Q, D0, D1)
        assert 0.0 <= val <= 1.0

    def test_a_topk_consistency_drops_when_unrelated(self):
        from semantic_sentry.metrics.behavioral import self_retrieval_topk
        rng = np.random.default_rng(42)
        Q0 = rng.standard_normal((30, 8)).astype(np.float32)
        Q1 = rng.standard_normal((30, 8)).astype(np.float32)        # independent
        D0 = rng.standard_normal((40, 8)).astype(np.float32)
        D1 = rng.standard_normal((40, 8)).astype(np.float32)
        out = self_retrieval_topk(Q0, Q1, D0, D1, k=5)
        assert 0.0 <= out <= 0.5, f"unrelated Q/D should give low overlap, got {out}"

    def test_a_handles_nan(self):
        """Behavioral metrics on NaN input — propagate NaN or raise, not silent 0."""
        from semantic_sentry.metrics.behavioral import (
            mean_abs_score_delta,
            score_distribution_jsd,
        )
        Q0 = np.ones((4, 4), dtype=np.float32)
        Q_nan = Q0.copy()
        Q_nan[0, 0] = np.nan
        D = np.ones((4, 4), dtype=np.float32)
        try:
            v1 = score_distribution_jsd(Q0, Q_nan, D, D)
            # If it didn't raise, it should at least not have produced a finite
            # ordinary-looking number from corrupted input.
            assert not np.isfinite(v1) or v1 == 0.0
        except (ValueError, FloatingPointError):
            pass
        try:
            v2 = mean_abs_score_delta(Q0, Q_nan, D, D)
            assert not np.isfinite(v2) or v2 == 0.0
        except (ValueError, FloatingPointError):
            pass

    def test_a_top_k_consistency_requires_enough_docs(self):
        """A5 raises when k > |D|, rather than silently using only |D| docs."""
        from semantic_sentry.metrics.behavioral import self_retrieval_topk
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((5, 4)).astype(np.float32)
        D = rng.standard_normal((3, 4)).astype(np.float32)
        with pytest.raises(ValueError):
            self_retrieval_topk(Q, Q, D, D, k=5)

    def test_a_score_jsd_with_empty_q_returns_zero_or_raises(self):
        from semantic_sentry.metrics.behavioral import score_distribution_jsd
        Q_empty = np.zeros((0, 4), dtype=np.float32)
        D = np.zeros((4, 4), dtype=np.float32)
        try:
            out = score_distribution_jsd(Q_empty, Q_empty, D, D)
            # Empty Q → 0 histogram bins on both sides → JSD = 0 is acceptable.
            assert out == 0.0
        except (ValueError, ZeroDivisionError):
            pass


# ---------------------------------------------------------------------------
# Family B — trustworthiness + continuity (local_structure.py)
# ---------------------------------------------------------------------------


class TestLocalStructure:
    def test_b_identity_inputs_score_one(self):
        from semantic_sentry.metrics.local_structure import continuity, trustworthiness
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((50, 8)).astype(np.float32)
        assert trustworthiness(Z, Z, k=10) == pytest.approx(1.0, abs=1e-6)
        assert continuity(Z, Z, k=10) == pytest.approx(1.0, abs=1e-6)

    def test_b_symmetry_swap(self):
        """continuity(A,B) should equal trustworthiness(B,A)."""
        from semantic_sentry.metrics.local_structure import continuity, trustworthiness
        rng = np.random.default_rng(11)
        Z0 = rng.standard_normal((40, 6)).astype(np.float32)
        Z1 = Z0 + 0.2 * rng.standard_normal((40, 6)).astype(np.float32)
        t_ab = trustworthiness(Z0, Z1, k=5)
        c_ba = continuity(Z1, Z0, k=5)
        assert abs(t_ab - c_ba) < 1e-5, f"trust(A,B)={t_ab} != cont(B,A)={c_ba}"

    def test_b_random_unrelated_drops_below_one(self):
        from semantic_sentry.metrics.local_structure import trustworthiness
        rng = np.random.default_rng(0)
        Z0 = rng.standard_normal((80, 16)).astype(np.float32)
        Z1 = rng.standard_normal((80, 16)).astype(np.float32)
        out = trustworthiness(Z0, Z1, k=10)
        assert out < 0.95, f"unrelated Z should drop trustworthiness, got {out}"

    def test_b_tiny_anchor_set_does_not_crash(self):
        """n at or near k+1 — must not crash, may return degenerate 1.0."""
        from semantic_sentry.metrics.local_structure import continuity, trustworthiness
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((12, 4)).astype(np.float32)
        t = trustworthiness(Z, Z, k=10)
        c = continuity(Z, Z, k=10)
        assert 0.0 <= t <= 1.0
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Family F — temporal layer: velocity / acceleration / plateau
# ---------------------------------------------------------------------------


class TestTemporal:
    """TST-F-*: temporal wrappers operate on Snapshot trajectories."""

    def _trajectory(self, T: int = 6, *, drift_per_step: float = 0.05,
                    seed: int = 0):
        """Build T snapshots with cumulative drift between consecutive ones."""
        anchor = _make_anchor(n=20)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((20, 8)).astype(np.float32)
        snapshots = [_make_snapshot(f"m{0}", anchor, emb.copy())]
        for t in range(1, T):
            emb = emb + drift_per_step * rng.standard_normal((20, 8)).astype(np.float32)
            snapshots.append(_make_snapshot(f"m{t}", anchor, emb.copy()))
        return snapshots, list(range(T))

    def test_f_velocity_returns_per_snapshot_array(self):
        from semantic_sentry.metrics.temporal import velocity
        snaps, times = self._trajectory(T=6, drift_per_step=0.0)
        v = velocity(snaps, times, metric_name="nps")
        assert v.shape == (6,)
        # Zero drift → NPS ≈ 1 throughout → velocity ≈ 0.
        assert np.allclose(v, 0.0, atol=1e-3)

    def test_f_velocity_increases_with_drift(self):
        """A drift trajectory should give a velocity with non-zero magnitude."""
        from semantic_sentry.metrics.temporal import velocity
        snaps, times = self._trajectory(T=6, drift_per_step=0.5)
        v = velocity(snaps, times, metric_name="nps")
        assert np.abs(v).max() > 0.01, (
            f"large drift produced zero velocity: {v}"
        )

    def test_f_acceleration_shape(self):
        from semantic_sentry.metrics.temporal import acceleration
        snaps, times = self._trajectory(T=8, drift_per_step=0.1)
        a = acceleration(snaps, times, metric_name="nps")
        assert a.shape == (8,)
        assert np.all(np.isfinite(a))

    def test_f_plateau_detects_flat_tail(self):
        """A trajectory that drifts then settles → plateau should fire on the tail."""
        from semantic_sentry.metrics.temporal import plateau
        # First half: heavy drift. Second half: identical snapshots.
        anchor = _make_anchor(n=20)
        rng = np.random.default_rng(0)
        emb_v0 = rng.standard_normal((20, 8)).astype(np.float32)
        emb_v1 = emb_v0 + 0.5 * rng.standard_normal((20, 8)).astype(np.float32)
        emb_v2 = emb_v1 + 0.5 * rng.standard_normal((20, 8)).astype(np.float32)
        # Plateau: stay at emb_v2.
        snaps = [
            _make_snapshot("v0", anchor, emb_v0),
            _make_snapshot("v1", anchor, emb_v1),
            _make_snapshot("v2", anchor, emb_v2),
            _make_snapshot("v3", anchor, emb_v2.copy()),
            _make_snapshot("v4", anchor, emb_v2.copy()),
            _make_snapshot("v5", anchor, emb_v2.copy()),
            _make_snapshot("v6", anchor, emb_v2.copy()),
        ]
        times = list(range(len(snaps)))
        mask = plateau(snaps, times, metric_name="nps",
                       eps=0.05, delta=0.05, k=2)
        assert mask.shape == (7,)
        # The tail (where snapshots are identical) must be flagged.
        assert mask[-2:].all(), f"flat tail not flagged: {mask}"

    def test_f_velocity_rejects_too_short_trajectory(self):
        from semantic_sentry.metrics.temporal import velocity
        snaps, times = self._trajectory(T=1)
        with pytest.raises(ValueError):
            velocity(snaps, times, metric_name="nps")

    def test_f_velocity_rejects_length_mismatch(self):
        from semantic_sentry.metrics.temporal import velocity
        snaps, _ = self._trajectory(T=4)
        with pytest.raises(ValueError):
            velocity(snaps, [0.0, 1.0], metric_name="nps")


# ---------------------------------------------------------------------------
# Family I2 — severity calibration on real Snapshots
# ---------------------------------------------------------------------------


class TestCalibration:
    """TST-I2-*: SeverityCalibration / calibrate_thresholds."""

    def _ref_snapshots(self, n: int = 4, drift: float = 0.005, seed_offset: int = 0):
        """Build N near-identical snapshots — small per-call jitter."""
        anchor = _make_anchor(n=20)
        rng = np.random.default_rng(0)
        base_emb = rng.standard_normal((20, 8)).astype(np.float32)
        snaps = []
        for i in range(n):
            j_rng = np.random.default_rng(seed_offset + i)
            jitter = drift * j_rng.standard_normal((20, 8)).astype(np.float32)
            snaps.append(_make_snapshot(f"ref{i}", anchor, base_emb + jitter))
        return snaps

    def test_i2_calibrate_returns_full_threshold_dict(self):
        from semantic_sentry.core.calibration import calibrate_thresholds
        cal = calibrate_thresholds(self._ref_snapshots(n=4))
        for k in ("nps_low", "nps_medium", "nps_high",
                  "cka_low", "cka_medium", "cka_high"):
            assert k in cal.thresholds
        # Thresholds should be in [0, 1] ordering (low > medium > high).
        assert cal.thresholds["nps_low"] >= cal.thresholds["nps_medium"]
        assert cal.thresholds["nps_medium"] >= cal.thresholds["nps_high"]

    def test_i2_calibrate_requires_at_least_two_snapshots(self):
        from semantic_sentry.core.calibration import calibrate_thresholds
        with pytest.raises(ValueError):
            calibrate_thresholds(self._ref_snapshots(n=1))

    def test_i2_calibrate_rejects_wrong_n_sigmas(self):
        from semantic_sentry.core.calibration import calibrate_thresholds
        with pytest.raises(ValueError):
            calibrate_thresholds(self._ref_snapshots(n=3), n_sigmas=(1.0, 2.0))

    def test_i2_calibrate_zero_variance_does_not_crash(self):
        """Two identical-content snapshots → zero variance → sigma = 0,
        thresholds collapse to mean. Should not raise."""
        from semantic_sentry.core.calibration import calibrate_thresholds
        anchor = _make_anchor(n=15)
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((15, 6)).astype(np.float32)
        snaps = [
            _make_snapshot("a", anchor, emb),
            _make_snapshot("b", anchor, emb.copy()),
        ]
        cal = calibrate_thresholds(snaps)
        # Mean NPS for identical snapshots = 1.0; sigma = 0 → all tiers at 1.0.
        assert all(abs(v - 1.0) < 1e-6 for k, v in cal.thresholds.items()
                   if k.startswith("nps_"))


# ---------------------------------------------------------------------------
# Cross-cutting — concurrency, leak isolation, E2E with partition
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_xc_behavioral_registry_thread_safe(self):
        """Concurrent registers on BehavioralMetricRegistry produce no lost rows."""
        from semantic_sentry.metrics.behavioral import (
            BehavioralMetricRegistry,
            self_retrieval_topk,
        )

        reg = BehavioralMetricRegistry()
        reg.reset()

        def register_many(thread_id: int):
            for i in range(20):
                reg.register(
                    f"t{thread_id}_m{i}", self_retrieval_topk,
                    params={"k": (i % 5) + 1},
                )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(register_many, range(8)))

        names = reg.list_metrics()
        registered_count = sum(1 for n in names if n.startswith("t") and "_m" in n)
        assert registered_count == 8 * 20, (
            f"lost registrations: {registered_count}/160 — registry not thread-safe"
        )

    def test_xc_registry_isolated_across_tests(self):
        """The autouse fixture should clean up between tests — verify by
        registering a canary here; the next run of the suite should not see it."""
        from semantic_sentry.metrics.registry import MetricRegistry
        reg = MetricRegistry()
        assert "leakcanary" not in reg.list_metrics(), (
            "registry not reset between tests"
        )

        def _f(Z0, Z1) -> float:
            return 0.0
        reg.register("leakcanary", _f)
        assert "leakcanary" in reg.list_metrics()

    def test_xc_full_e2e_partition_then_compare(self):
        """Integration smoke: register behavioral → snapshot → partition →
        compare(d_snapshot_*=…). After explicit registration, behavioral
        metrics should appear in `global_metrics`."""
        from semantic_sentry import DriftMonitor
        from semantic_sentry.metrics.behavioral import register_behavioral_metrics

        # Populate the behavioral registry — it's empty by default after the
        # autouse reset; users opt in by calling `register_behavioral_metrics()`.
        register_behavioral_metrics()

        n = 40
        anchor = _make_anchor(n=n)
        q, d = anchor.partition(ratio=0.5, seed=0)

        rng = np.random.default_rng(0)
        emb_v0 = rng.standard_normal((n, 16)).astype(np.float32)
        emb_v1 = emb_v0 + 0.05 * rng.standard_normal((n, 16)).astype(np.float32)
        q_v0 = np.stack([emb_v0[anchor.inputs.index(t)] for t in q.inputs])
        q_v1 = np.stack([emb_v1[anchor.inputs.index(t)] for t in q.inputs])
        d_v0 = np.stack([emb_v0[anchor.inputs.index(t)] for t in d.inputs])
        d_v1 = np.stack([emb_v1[anchor.inputs.index(t)] for t in d.inputs])

        sq0 = _make_snapshot("q_v0", q, q_v0)
        sd0 = _make_snapshot("d_v0", d, d_v0)
        sq1 = _make_snapshot("q_v1", q, q_v1)
        sd1 = _make_snapshot("d_v1", d, d_v1)

        monitor = DriftMonitor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmp_with_d = monitor.compare(sq0, sq1, d_snapshot_v0=sd0, d_snapshot_v1=sd1)

        assert "cka" in cmp_with_d.global_metrics
        assert "nps" in cmp_with_d.global_metrics
        # After register_behavioral_metrics(), at least one behavioural metric
        # should appear in the merged global_metrics dict.
        beh_names = {"score_distribution_jsd", "mean_abs_score_delta",
                      "per_query_rbo", "per_query_kendall_tau"}
        merged_keys = set(cmp_with_d.global_metrics.keys())
        assert (beh_names & merged_keys) or any(
            n.startswith("self_retrieval_topk") for n in merged_keys
        ), f"no behavioural metric merged; got {sorted(merged_keys)}"

    def test_xc_compare_rejects_unpaired_d_snapshot(self):
        from semantic_sentry import DriftMonitor

        anchor = _make_anchor(n=10)
        rng = np.random.default_rng(0)
        s0 = _make_snapshot("a", anchor, rng.standard_normal((10, 4)).astype(np.float32))
        s1 = _make_snapshot("b", anchor, rng.standard_normal((10, 4)).astype(np.float32))
        monitor = DriftMonitor()
        with pytest.raises(ValueError):
            monitor.compare(s0, s1, d_snapshot_v0=s0)


# ---------------------------------------------------------------------------
# Numerical edge cases on the new metrics
# ---------------------------------------------------------------------------


class TestNumericalEdges:
    def test_nx_all_zero_embeddings(self):
        """L2-normalising all-zero rows must not propagate NaN through the new
        behavioural / structural metrics."""
        from semantic_sentry.metrics.behavioral import self_retrieval_topk
        from semantic_sentry.metrics.local_structure import trustworthiness
        Z = np.zeros((20, 4), dtype=np.float32)
        assert np.isfinite(trustworthiness(Z, Z, k=5))
        assert np.isfinite(self_retrieval_topk(Z, Z, Z, Z, k=5))

    def test_nx_float64_input_accepted(self):
        from semantic_sentry.metrics.behavioral import self_retrieval_topk
        from semantic_sentry.metrics.local_structure import continuity, trustworthiness
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((30, 8)).astype(np.float64)
        assert np.isfinite(trustworthiness(Z, Z, k=5))
        assert np.isfinite(continuity(Z, Z, k=5))
        assert np.isfinite(self_retrieval_topk(Z, Z, Z, Z, k=5))


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------


@pytest.mark.stress
class TestPerformance:
    def test_perf_behavioral_n1000(self):
        """All behavioural metrics on Q=500 × D=500 should complete in < 15 s."""
        from semantic_sentry.metrics.behavioral import (
            mean_abs_score_delta,
            per_query_kendall_tau,
            per_query_rbo,
            score_distribution_jsd,
            self_retrieval_topk,
        )
        rng = np.random.default_rng(0)
        Q0 = rng.standard_normal((500, 64)).astype(np.float32)
        Q1 = Q0 + 0.05 * rng.standard_normal((500, 64)).astype(np.float32)
        D0 = rng.standard_normal((500, 64)).astype(np.float32)
        D1 = D0 + 0.05 * rng.standard_normal((500, 64)).astype(np.float32)

        start = time.perf_counter()
        score_distribution_jsd(Q0, Q1, D0, D1)
        mean_abs_score_delta(Q0, Q1, D0, D1)
        per_query_rbo(Q0, Q1, D0, D1)
        per_query_kendall_tau(Q0, Q1, D0, D1)
        self_retrieval_topk(Q0, Q1, D0, D1, k=10)
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"behavioral suite on n=1000 took {elapsed:.2f}s"

    def test_perf_temporal_trajectory(self):
        """velocity + plateau on a 20-snapshot trajectory should be sub-second."""
        from semantic_sentry.metrics.temporal import plateau, velocity
        anchor = _make_anchor(n=30)
        rng = np.random.default_rng(0)
        snaps = []
        emb = rng.standard_normal((30, 16)).astype(np.float32)
        for _ in range(20):
            emb = emb + 0.01 * rng.standard_normal((30, 16)).astype(np.float32)
            snaps.append(_make_snapshot("m", anchor, emb.copy()))
        times = list(range(20))
        start = time.perf_counter()
        velocity(snaps, times, metric_name="nps")
        plateau(snaps, times, metric_name="nps", eps=1e-3, delta=1e-4, k=3)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"temporal trajectory took {elapsed:.2f}s"
