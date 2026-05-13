"""Tests for AnchorSet.partition and composite version hashes (H1)."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.exceptions import AnchorSetMismatchError
from semantic_sentry.probes.anchor_set import AnchorSet


class TestPartitionBasics:
    def test_partition_splits_sizes(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)])
        q, d = a.partition(ratio=0.4, seed=0)
        assert q.n_samples == 8
        assert d.n_samples == 12
        assert len(q) == 8
        assert len(d) == 12

    def test_partition_disjoint_and_complete(self):
        items = [f"x_{i}" for i in range(50)]
        a = AnchorSet(inputs=items)
        q, d = a.partition(ratio=0.5, seed=42)
        assert set(q.inputs).isdisjoint(set(d.inputs))
        assert set(q.inputs) | set(d.inputs) == set(items)

    def test_partition_is_seeded(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        q1, d1 = a.partition(seed=7)
        q2, d2 = a.partition(seed=7)
        assert list(q1.inputs) == list(q2.inputs)
        assert list(d1.inputs) == list(d2.inputs)

    def test_different_seeds_produce_different_splits(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        q1, _ = a.partition(seed=1)
        q2, _ = a.partition(seed=2)
        assert list(q1.inputs) != list(q2.inputs)

    def test_partition_inherits_distribution_tag(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)], distribution_tag="OOD")
        q, d = a.partition()
        assert q.distribution_tag == "OOD"
        assert d.distribution_tag == "OOD"

    def test_partition_carries_labels(self):
        labels = tuple(f"label_{i % 3}" for i in range(20))
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)], labels=labels)
        q, d = a.partition(ratio=0.5, seed=0)
        assert len(q.labels) == q.n_samples
        assert len(d.labels) == d.n_samples
        assert set(q.labels) | set(d.labels) == set(labels)

    def test_too_small_to_partition(self):
        a = AnchorSet(inputs=["a"])
        with pytest.raises(ValueError, match=">=2 samples"):
            a.partition()

    def test_invalid_ratio(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(10)])
        with pytest.raises(ValueError, match="ratio"):
            a.partition(ratio=0.0)
        with pytest.raises(ValueError, match="ratio"):
            a.partition(ratio=1.0)


class TestCompositeVersionHash:
    def test_partition_hashes_differ_from_parent_and_each_other(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        q, d = a.partition(seed=0)
        assert q.version_hash != a.version_hash
        assert d.version_hash != a.version_hash
        assert q.version_hash != d.version_hash

    def test_partition_hash_encodes_parent_role_seed(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        q, d = a.partition(seed=5)
        assert q.version_hash == f"{a.version_hash}:Q:5"
        assert d.version_hash == f"{a.version_hash}:D:5"

    def test_paired_partitions_have_stable_hashes_across_reruns(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(30)])
        q1, d1 = a.partition(seed=42)
        q2, d2 = a.partition(seed=42)
        assert q1.version_hash == q2.version_hash
        assert d1.version_hash == d2.version_hash


class TestPartitionedSnapshotComparison:
    """Q-snapshots can only compare with Q-snapshots; D with D."""

    def _make_adapter(self, rng_seed: int, n: int) -> CustomAdapter:
        Z = np.random.default_rng(rng_seed).standard_normal((n, 16)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z[: len(inputs)]},
            tower_names=["encoder"],
        )

    def test_q_only_compares_with_q(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)])
        q, _ = a.partition(seed=0)
        monitor = DriftMonitor()
        s0 = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(0, q.n_samples))
        s1 = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(1, q.n_samples))
        # Same role, same seed -> should compare without error.
        comp = monitor.compare(s0, s1)
        assert "cka" in comp.global_metrics

    def test_q_vs_d_snapshots_cannot_compare(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)])
        q, d = a.partition(seed=0)
        monitor = DriftMonitor()
        s_q = monitor.snapshot(model=None, anchor_set=q, adapter=self._make_adapter(0, q.n_samples))
        s_d = monitor.snapshot(model=None, anchor_set=d, adapter=self._make_adapter(0, d.n_samples))
        # Different roles -> version_hash differs -> AnchorSetMismatchError.
        with pytest.raises(AnchorSetMismatchError):
            monitor.compare(s_q, s_d)

    def test_different_partition_seeds_cannot_compare(self):
        a = AnchorSet(inputs=[f"x_{i}" for i in range(20)])
        q1, _ = a.partition(seed=1)
        q2, _ = a.partition(seed=2)
        monitor = DriftMonitor()
        s1 = monitor.snapshot(
            model=None, anchor_set=q1, adapter=self._make_adapter(0, q1.n_samples)
        )
        s2 = monitor.snapshot(
            model=None, anchor_set=q2, adapter=self._make_adapter(0, q2.n_samples)
        )
        with pytest.raises(AnchorSetMismatchError):
            monitor.compare(s1, s2)
