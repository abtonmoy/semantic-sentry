"""Tests for rolling baseline (A), live temporal signals (B), async (C)."""

from concurrent.futures import Future

import numpy as np
import pytest
import torch
import torch.nn as nn

from semantic_sentry import AnchorSet, DriftMonitor
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.metrics.cka import linear_cka

N, D = 60, 16


def _anchors():
    return AnchorSet(inputs=[f"t{i}" for i in range(N)], modality="text")


def _adapter(mat):
    arr = mat.astype(np.float32)

    def enc(_inputs):
        return torch.from_numpy(arr)

    return CustomAdapter(encode_fn=enc, tower_count=1, normalize=False)


def _model():
    return nn.Linear(4, 4)


@pytest.fixture
def abc():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N, D))
    B = rng.standard_normal((N, D))
    C = rng.standard_normal((N, D))
    return A, B, C


# --- A: sliding-window baseline ---------------------------------------------

def test_previous_mode_compares_to_prior(abc):
    A, B, C = abc
    monitor = DriftMonitor(baseline_mode="previous")
    assert monitor.track(_model(), _anchors(), adapter=_adapter(A)) is None
    monitor.track(_model(), _anchors(), adapter=_adapter(B))
    cmp3 = monitor.track(_model(), _anchors(), adapter=_adapter(C))
    # previous mode -> third call compares C against B (not A).
    assert cmp3.global_metrics["cka"] == pytest.approx(linear_cka(B, C), abs=1e-4)


def test_fixed_vs_previous_differ(abc):
    A, B, C = abc
    fixed = DriftMonitor(baseline_mode="fixed")
    prev = DriftMonitor(baseline_mode="previous")
    for mon in (fixed, prev):
        mon.track(_model(), _anchors(), adapter=_adapter(A))
        mon.track(_model(), _anchors(), adapter=_adapter(B))
    c_fixed = fixed.track(_model(), _anchors(), adapter=_adapter(C))
    c_prev = prev.track(_model(), _anchors(), adapter=_adapter(C))
    # fixed compares C vs A; previous compares C vs B -> different references.
    assert c_fixed.global_metrics["cka"] == pytest.approx(linear_cka(A, C), abs=1e-4)
    assert c_prev.global_metrics["cka"] == pytest.approx(linear_cka(B, C), abs=1e-4)
    assert abs(c_fixed.global_metrics["cka"] - c_prev.global_metrics["cka"]) > 1e-3


# --- B: live temporal signals -----------------------------------------------

def test_temporal_metadata_present(abc):
    A, B, _ = abc
    monitor = DriftMonitor(track_temporal=True)
    monitor.track(_model(), _anchors(), adapter=_adapter(A))
    cmp = monitor.track(_model(), _anchors(), adapter=_adapter(B))
    assert "temporal" in cmp.metadata
    t = cmp.metadata["temporal"]
    assert set(t) == {"metric", "velocity", "acceleration", "plateau"}
    assert t["metric"] == "cka"


def test_plateau_fires_on_stable_geometry():
    rng = np.random.default_rng(5)
    M = rng.standard_normal((N, D))  # identical embeddings every step -> no drift
    monitor = DriftMonitor(track_temporal=True, plateau_k=3)
    last = None
    for step in range(7):
        last = monitor.track(_model(), _anchors(), adapter=_adapter(M), step=step)
    assert last.metadata["temporal"]["plateau"] is True
    assert last.metadata["temporal"]["velocity"] == pytest.approx(0.0, abs=1e-9)


def test_plateau_false_under_continuous_drift():
    rng = np.random.default_rng(6)
    monitor = DriftMonitor(track_temporal=True, plateau_k=3)
    last = None
    for step in range(7):
        # fresh random embeddings each step -> persistent change, no plateau.
        last = monitor.track(_model(), _anchors(),
                             adapter=_adapter(rng.standard_normal((N, D))), step=step)
    assert last.metadata["temporal"]["plateau"] is False


# --- C: async / non-blocking ------------------------------------------------

def test_async_returns_future_and_drains(abc):
    A, B, _ = abc
    monitor = DriftMonitor(async_mode=True)
    assert monitor.track(_model(), _anchors(), adapter=_adapter(A)) is None
    fut = monitor.track(_model(), _anchors(), adapter=_adapter(B))
    assert isinstance(fut, Future)
    result = fut.result(timeout=10)
    assert isinstance(result, Comparison)
    monitor.drain(timeout=10)
    assert monitor.last_result is result
    monitor.close()


def test_async_matches_sync(abc):
    A, B, _ = abc
    sync = DriftMonitor()
    sync.track(_model(), _anchors(), adapter=_adapter(A))
    cmp_sync = sync.track(_model(), _anchors(), adapter=_adapter(B))

    amon = DriftMonitor(async_mode=True)
    amon.track(_model(), _anchors(), adapter=_adapter(A))
    cmp_async = amon.track(_model(), _anchors(), adapter=_adapter(B)).result(timeout=10)
    amon.close()
    assert cmp_async.global_metrics["cka"] == pytest.approx(
        cmp_sync.global_metrics["cka"], abs=1e-6
    )


def test_async_keep_on_device_returns_future():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((N, D))
    monitor = DriftMonitor(async_mode=True)
    # keep_on_device async path returns a Future too.
    monitor.track(_model(), _anchors(), adapter=_adapter(M), keep_on_device=True)
    fut = monitor.track(_model(), _anchors(), adapter=_adapter(M), keep_on_device=True)
    assert isinstance(fut, Future)
    cmp = fut.result(timeout=10)
    assert cmp.metadata.get("backend") == "torch"
    monitor.close()
