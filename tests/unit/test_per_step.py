"""Tests for per-step tracking: stride, eval-mode probe, async backpressure."""

import threading
import types
from concurrent.futures import Future

import numpy as np
import pytest
import torch
import torch.nn as nn

from semantic_sentry import AnchorSet, DriftMonitor
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.integrations import callbacks as cb_mod
from semantic_sentry.integrations.callbacks import SemanticSentryCallback

N, D = 50, 12


def _anchors():
    return AnchorSet(inputs=[f"t{i}" for i in range(N)], modality="text")


def _adapter(mat):
    arr = mat.astype(np.float32)

    def enc(_inputs):
        return torch.from_numpy(arr)

    return CustomAdapter(encode_fn=enc, tower_count=1, normalize=False)


def _model():
    return nn.Linear(4, 4)


# --- eval-mode probe --------------------------------------------------------

def test_probe_switches_to_eval_and_restores():
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((N, D)).astype(np.float32)
    model = nn.Linear(4, 4)
    model.train()
    seen = []

    def enc(_inputs):
        seen.append(model.training)  # capture mode *during* the encode
        return torch.from_numpy(mat)

    adapter = CustomAdapter(encode_fn=enc, tower_count=1, normalize=False)
    DriftMonitor().track(model, _anchors(), adapter=adapter)

    assert seen[-1] is False           # encoded in eval mode
    assert model.training is True      # train mode restored afterwards


def test_probe_eval_mode_false_keeps_train_mode():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((N, D)).astype(np.float32)
    model = nn.Linear(4, 4)
    model.train()
    seen = []

    def enc(_inputs):
        seen.append(model.training)
        return torch.from_numpy(mat)

    adapter = CustomAdapter(encode_fn=enc, tower_count=1, normalize=False)
    DriftMonitor().track(model, _anchors(), adapter=adapter, probe_eval_mode=False)

    assert seen[-1] is True            # left in train mode


# --- async backpressure (drop-if-busy) --------------------------------------

def test_max_inflight_skips_when_busy():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((N, D))
    B = rng.standard_normal((N, D))
    C = rng.standard_normal((N, D))
    release = threading.Event()

    class BlockingLogger:
        def log_comparison(self, comparison, step=None):
            release.wait(5)  # hold the worker so a job stays in-flight

    monitor = DriftMonitor(async_mode=True, max_inflight=1)
    assert monitor.track(_model(), _anchors(), adapter=_adapter(A)) is None  # baseline
    fut = monitor.track(_model(), _anchors(), adapter=_adapter(B), logger=BlockingLogger())
    assert isinstance(fut, Future)
    assert monitor.is_busy
    # Worker is saturated -> this measurement is dropped (no encode, no queue).
    assert monitor.track(_model(), _anchors(), adapter=_adapter(C)) is None

    release.set()
    monitor.close()
    assert isinstance(fut.result(timeout=5), Comparison)


# --- per-step callback cadence ----------------------------------------------

@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_callback_tracks_on_step_stride():
    rng = np.random.default_rng(3)
    adapter = _adapter(rng.standard_normal((N, D)))
    cb = SemanticSentryCallback(_anchors(), adapter=adapter, every_n_steps=2)
    model = nn.Linear(4, 4)

    def step(n):
        cb.on_step_end(args=None, state=types.SimpleNamespace(global_step=n),
                       control=None, model=model)

    step(1)  # odd -> skipped
    assert cb.monitor._baseline_snapshot is None
    step(2)  # tracked -> establishes baseline
    assert cb.monitor._baseline_snapshot is not None
    assert cb.last_comparison is None
    step(3)  # odd -> skipped
    assert cb.last_comparison is None
    step(4)  # tracked -> first comparison
    assert cb.last_comparison is not None


@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_per_step_mode_disables_eval_hook():
    rng = np.random.default_rng(4)
    adapter = _adapter(rng.standard_normal((N, D)))
    cb = SemanticSentryCallback(_anchors(), adapter=adapter, every_n_steps=2)
    model = nn.Linear(4, 4)
    # on_evaluate must be a no-op in per-step mode (no baseline established).
    cb.on_evaluate(args=None, state=types.SimpleNamespace(global_step=10),
                   control=None, model=model)
    assert cb.monitor._baseline_snapshot is None
