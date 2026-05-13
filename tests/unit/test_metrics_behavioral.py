"""Tests for behavioral metrics A1-A5 and the BehavioralMetricRegistry."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.metrics.behavioral import (
    BehavioralMetricRegistry,
    mean_abs_score_delta,
    per_query_kendall_tau,
    per_query_rbo,
    register_behavioral_metrics,
    score_distribution_jsd,
    self_retrieval_topk,
)
from semantic_sentry.probes.anchor_set import AnchorSet


def _unit(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class TestA1JSD:
    def test_identical_distributions_zero_divergence(self):
        Q = _unit(20, 16, 0)
        D = _unit(50, 16, 1)
        assert score_distribution_jsd(Q, Q, D, D) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_score_distribution_positive(self):
        Q0 = _unit(40, 16, 0)
        D = _unit(60, 16, 1)
        # Concentrate Q1 near a single direction so its score distribution
        # with D shifts away from Q0's near-zero-mean distribution.
        rng = np.random.default_rng(2)
        bias = rng.standard_normal(16).astype(np.float32)
        bias /= np.linalg.norm(bias)
        Q1 = np.tile(bias, (40, 1))
        v_baseline = score_distribution_jsd(Q0, Q0, D, D)
        v_shifted = score_distribution_jsd(Q0, Q1, D, D)
        # Identical input gives ~0; biased input must be measurably larger.
        assert v_baseline < 1e-6
        assert v_shifted > v_baseline + 0.01


class TestA2MeanAbsDelta:
    def test_identical_zero(self):
        Q = _unit(20, 16, 0)
        D = _unit(50, 16, 1)
        assert mean_abs_score_delta(Q, Q, D, D) == pytest.approx(0.0, abs=1e-6)

    def test_negation_max_delta(self):
        Q = _unit(20, 16, 0)
        D = _unit(50, 16, 1)
        # Negating Q flips every cosine sim — |Δ| = 2 * |original score|.
        original = float(np.mean(np.abs(Q @ D.T)))
        delta = mean_abs_score_delta(Q, -Q, D, D)
        assert delta == pytest.approx(2 * original, rel=1e-5)


class TestA3RBO:
    def test_identical_rankings_score_one(self):
        Q = _unit(10, 16, 0)
        D = _unit(30, 16, 1)
        v = per_query_rbo(Q, Q, D, D, p=0.9)
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_random_lower_than_identical(self):
        Q0 = _unit(10, 16, 0)
        Q1 = _unit(10, 16, 99)
        D = _unit(30, 16, 1)
        v = per_query_rbo(Q0, Q1, D, D, p=0.9)
        assert v < 0.95


class TestA4Kendall:
    def test_identical_rankings_tau_one(self):
        Q = _unit(10, 16, 0)
        D = _unit(30, 16, 1)
        assert per_query_kendall_tau(Q, Q, D, D) == pytest.approx(1.0)

    def test_negated_query_tau_minus_one(self):
        Q = _unit(10, 16, 0)
        D = _unit(30, 16, 1)
        # cos(-q, d) = -cos(q, d) flips the full ranking.
        assert per_query_kendall_tau(Q, -Q, D, D) == pytest.approx(-1.0)


class TestA5SelfRetrieval:
    def test_identical_top_k_match_one(self):
        Q = _unit(10, 16, 0)
        D = _unit(30, 16, 1)
        for k in (1, 5, 10):
            assert self_retrieval_topk(Q, Q, D, D, k=k) == pytest.approx(1.0)

    def test_random_lower(self):
        Q0 = _unit(10, 16, 0)
        Q1 = _unit(10, 16, 99)
        D = _unit(30, 16, 1)
        v = self_retrieval_topk(Q0, Q1, D, D, k=5)
        assert 0.0 <= v <= 1.0
        # Two random rankings unlikely to fully agree on top-5
        assert v < 1.0

    def test_too_few_docs_raises(self):
        Q = _unit(5, 8, 0)
        D = _unit(3, 8, 1)
        with pytest.raises(ValueError, match="at least"):
            self_retrieval_topk(Q, Q, D, D, k=10)


class TestRegistration:
    def test_register_behavioral_adds_entries(self):
        registry = BehavioralMetricRegistry()
        register_behavioral_metrics(registry, self_retrieval_ks=(1, 5))
        names = set(registry.list_metrics())
        assert {
            "score_distribution_jsd",
            "mean_abs_score_delta",
            "per_query_rbo",
            "per_query_kendall_tau",
            "self_retrieval_at_1",
            "self_retrieval_at_5",
        }.issubset(names)

    def test_registered_compute(self):
        registry = BehavioralMetricRegistry()
        register_behavioral_metrics(registry, self_retrieval_ks=(5,))
        Q = _unit(10, 16, 0)
        D = _unit(20, 16, 1)
        results = registry.compute_all(Q, Q, D, D)
        assert results["mean_abs_score_delta"] == pytest.approx(0.0, abs=1e-6)
        assert results["per_query_rbo"] == pytest.approx(1.0, abs=1e-6)


class TestDriftMonitorIntegration:
    """End-to-end: AnchorSet.partition → snapshots → compare with D-side."""

    def _make_adapter(self, Z: np.ndarray) -> CustomAdapter:
        Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_norm[: len(inputs)]},
            tower_names=["encoder"],
        )

    def test_compare_with_behavioral_d_side(self):
        register_behavioral_metrics(self_retrieval_ks=(5,))

        # Build a 40-input anchor set, partition into Q (20) and D (20).
        anchor = AnchorSet(inputs=[f"x_{i}" for i in range(40)])
        q, d = anchor.partition(ratio=0.5, seed=0)

        # Same model embeddings for both versions -> behavioral metrics
        # should report perfect preservation.
        Z_q = _unit(q.n_samples, 16, 7)
        Z_d = _unit(d.n_samples, 16, 11)

        monitor = DriftMonitor()
        s0_q = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(Z_q))
        s1_q = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(Z_q))
        s0_d = monitor.snapshot(model=None, anchor_set=d, adapter=self._make_adapter(Z_d))
        s1_d = monitor.snapshot(model=None, anchor_set=d, adapter=self._make_adapter(Z_d))

        comp = monitor.compare(
            s0_q, s1_q, d_snapshot_v0=s0_d, d_snapshot_v1=s1_d
        )
        assert "mean_abs_score_delta" in comp.global_metrics
        assert comp.global_metrics["mean_abs_score_delta"] == pytest.approx(0.0, abs=1e-6)
        assert comp.global_metrics["per_query_rbo"] == pytest.approx(1.0, abs=1e-6)
        assert comp.global_metrics["self_retrieval_at_5"] == pytest.approx(1.0)

    def test_compare_rejects_one_sided_d(self):
        anchor = AnchorSet(inputs=[f"x_{i}" for i in range(20)])
        q, d = anchor.partition()
        monitor = DriftMonitor()
        Z_q = _unit(q.n_samples, 8, 0)
        Z_d = _unit(d.n_samples, 8, 1)
        s0_q = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(Z_q))
        s1_q = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(Z_q))
        s_d = monitor.snapshot(model=None, anchor_set=d, adapter=self._make_adapter(Z_d))
        with pytest.raises(ValueError, match="both be provided"):
            monitor.compare(s0_q, s1_q, d_snapshot_v0=s_d)
