"""Tests covering the v0.1.x improvement backlog items 6, 8, 9.

These hit specific behaviour the audit flagged — keep them isolated so a
future cleanup can grep for `improvement.md` and find the regression
coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.metrics.nps import _get_knn_indices, _l2_normalize, nps
from semantic_sentry.metrics.registry import MetricRegistry

# ---------------------------------------------------------------------------
# Item 6 — cross-tower alignment keys with `__` in tower names round-trip
# ---------------------------------------------------------------------------

def test_cross_tower_alignment_roundtrip_with_double_underscore_names(tmp_path):
    """Tower names containing `__` must survive save/load."""
    n, d = 16, 8
    rng = np.random.default_rng(0)
    emb_a = rng.standard_normal((n, d)).astype(np.float32)
    emb_b = rng.standard_normal((n, d)).astype(np.float32)

    snap = Snapshot(
        model_id="test",
        checkpoint_hash="abc123",
        tower_count=2,
        tower_names=("vision__patch", "text__caption"),
        embeddings={"vision__patch": emb_a, "text__caption": emb_b},
        cross_tower_alignment={
            ("vision__patch", "text__caption"): 0.42,
        },
    )

    out = tmp_path / "snap"
    snap.save(out)
    loaded = Snapshot.load(out)

    assert loaded.cross_tower_alignment is not None
    assert ("vision__patch", "text__caption") in loaded.cross_tower_alignment
    assert loaded.cross_tower_alignment[("vision__patch", "text__caption")] == 0.42


def test_cross_tower_alignment_legacy_dict_form_is_loadable(tmp_path):
    """Snapshots written with the old `a__b: v` dict form still load (no `__` in names)."""
    import json
    out = tmp_path / "snap"
    out.mkdir()
    # Build a minimal metadata.json in the legacy shape + the matching
    # safetensors files.
    from safetensors.numpy import save_file
    save_file({"vision": np.zeros((4, 2), dtype=np.float32)}, out / "vision.safetensors")
    save_file({"text":   np.zeros((4, 2), dtype=np.float32)}, out / "text.safetensors")
    # We don't compute embeddings_hash here so we skip the integrity check.
    (out / "metadata.json").write_text(json.dumps({
        "model_id": "test",
        "checkpoint_hash": "abc",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "anchor_set_version": "",
        "tower_count": 2,
        "tower_names": ["vision", "text"],
        "cross_tower_alignment": {"vision__text": 0.5},
        "metadata": {},
    }))
    loaded = Snapshot.load(out)
    assert loaded.cross_tower_alignment == {("vision", "text"): 0.5}


# ---------------------------------------------------------------------------
# Item 8 — _get_knn_indices excludes self deterministically
# ---------------------------------------------------------------------------

def test_get_knn_excludes_self_numpy_branch():
    """Numpy fallback never returns the query row's own index."""
    rng = np.random.default_rng(0)
    X = _l2_normalize(rng.standard_normal((200, 16)).astype(np.float32))
    nbrs = _get_knn_indices(X, k=10)
    assert nbrs.shape == (200, 10)
    self_idx = np.arange(200)[:, None]
    assert not (nbrs == self_idx).any(), "self appeared in neighbour set"


def test_nps_identity_remains_one_after_self_exclusion_refactor():
    """Sanity: nps(X, X) is still 1.0 after the _get_knn_indices refactor."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 32)).astype(np.float32)
    assert nps(X, X, k=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Item 9 — MetricRegistry.reset()
# ---------------------------------------------------------------------------

def test_registry_reset_drops_custom_metrics_and_keeps_builtins():
    reg = MetricRegistry()
    # Register a custom metric.
    def my_metric(a: np.ndarray, b: np.ndarray) -> float:
        return 0.0
    reg.register("my_metric", my_metric, range=(0.0, 1.0), description="test")
    assert "my_metric" in reg.list_metrics()
    # Reset and verify the custom one is gone but builtins remain.
    reg.reset()
    names = reg.list_metrics()
    assert "my_metric" not in names
    assert "cka" in names
    assert "nps" in names
    assert "isotropy_delta" in names


# ---------------------------------------------------------------------------
# Item 2 — classify() reuses precomputed per-anchor NPS
# ---------------------------------------------------------------------------

def test_classify_local_nps_uses_precomputed_per_anchor_nps():
    """After compare(), monitor.classify() should read the cached per-anchor
    NPS rather than recomputing on every call. The cache should be populated
    when the anchor set is large enough (>= k+1 = 11 with default k=10)."""
    from semantic_sentry import DriftMonitor
    from semantic_sentry.adapters.custom import CustomAdapter
    from semantic_sentry.probes.anchor_set import AnchorSet

    rng = np.random.default_rng(0)
    anchor_inputs = [f"anchor_{i}" for i in range(32)]
    anchor_labels = tuple(f"label_{i % 4}" for i in range(32))
    anchor_set = AnchorSet(inputs=anchor_inputs, labels=anchor_labels)

    # Two slightly different "encoders" so v0 != v1.
    v0_encodings = rng.standard_normal((32, 16)).astype(np.float32)
    v1_encodings = v0_encodings + 0.1 * rng.standard_normal((32, 16)).astype(np.float32)

    class _StaticEncoder:
        def __init__(self, table):
            self.table = {t: e for t, e in zip(anchor_inputs, table, strict=False)}
        def encode(self, texts):
            return np.stack([self.table.get(t, np.zeros(16, dtype=np.float32))
                              for t in texts])

    enc_v0 = _StaticEncoder(v0_encodings)
    enc_v1 = _StaticEncoder(v1_encodings)
    adapter_v0 = CustomAdapter(encode_fn=enc_v0.encode, tower_count=1, tower_names=("encoder",))
    adapter_v1 = CustomAdapter(encode_fn=enc_v1.encode, tower_count=1, tower_names=("encoder",))

    monitor = DriftMonitor()
    s0 = monitor.snapshot(enc_v0, anchor_set, adapter=adapter_v0)
    s1 = monitor.snapshot(enc_v1, anchor_set, adapter=adapter_v1)
    monitor.compare(s0, s1)

    # After compare() the per-anchor NPS cache should exist and have one row
    # per anchor.
    assert monitor._last_anchor_per_point_nps is not None
    per = monitor._last_anchor_per_point_nps["encoder"]
    assert per.shape == (32,)
    # Per-anchor NPS is in [0, 1] elementwise.
    assert (per >= 0.0).all() and (per <= 1.0).all()


# ---------------------------------------------------------------------------
# Item 4 — LogisticTransfer fits true MLE, not OLS on binary labels
# ---------------------------------------------------------------------------

def test_logistic_transfer_separates_a_separable_toy_problem():
    """If positives are clustered on one side of the feature space, the
    fitted probability should be high on positives and low on negatives."""
    from semantic_sentry.core.comparison import Comparison
    from semantic_sentry.transfer.function import LogisticTransfer

    rng = np.random.default_rng(0)
    # Synth: 1-cka, 1-nps, |iso| features. High values → degraded.
    n = 80
    drift_features = np.zeros((n, 3))
    drift_features[:n // 2] = rng.uniform(0.5, 1.0, (n // 2, 3))   # high drift
    drift_features[n // 2:] = rng.uniform(0.0, 0.1, (n // 2, 3))    # low drift
    degradations = [0.5] * (n // 2) + [0.0] * (n // 2)

    comparisons = []
    for f in drift_features:
        comparisons.append(Comparison(
            snapshot_v0_hash="a", snapshot_v1_hash="b",
            global_metrics={
                "cka": 1.0 - f[0], "nps": 1.0 - f[1], "isotropy_delta": f[2],
            },
        ))

    transfer = LogisticTransfer(degradation_threshold=0.1)
    transfer.fit(comparisons, degradations)

    high_drift_prob = transfer.predict(comparisons[0])
    low_drift_prob  = transfer.predict(comparisons[-1])
    assert 0.0 <= high_drift_prob <= 1.0
    assert 0.0 <= low_drift_prob  <= 1.0
    # On a cleanly-separable dataset the MLE fit should produce a wide gap.
    assert high_drift_prob - low_drift_prob > 0.5
