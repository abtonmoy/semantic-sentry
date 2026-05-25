"""Tests for the training-framework callbacks."""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from semantic_sentry import AnchorSet, DriftMonitor
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.integrations import callbacks as cb_mod
from semantic_sentry.integrations.callbacks import (
    SemanticSentryCallback,
    SemanticSentryLightningCallback,
)

N, D = 50, 8


def _anchors():
    return AnchorSet(inputs=[f"t{i}" for i in range(N)], modality="text")


def _adapter():
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((N, D)).astype(np.float32)

    def enc(_inputs):
        return torch.from_numpy(mat)

    return CustomAdapter(encode_fn=enc, tower_count=1)


def _state(step):
    return types.SimpleNamespace(global_step=step)


@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_hf_callback_baseline_then_comparison():
    model = nn.Linear(4, 4)
    cb = SemanticSentryCallback(_anchors(), adapter=_adapter())

    # First evaluate -> baseline, no comparison recorded.
    cb.on_evaluate(args=None, state=_state(0), control=None, model=model)
    assert cb.last_comparison is None

    # Second evaluate -> comparison produced.
    cb.on_evaluate(args=None, state=_state(100), control=None, model=model)
    assert cb.last_comparison is not None
    assert "cka" in cb.last_comparison.global_metrics


@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_hf_callback_on_save_mode_ignores_evaluate():
    model = nn.Linear(4, 4)
    cb = SemanticSentryCallback(_anchors(), adapter=_adapter(), on_event="save")
    cb.on_evaluate(args=None, state=_state(0), control=None, model=model)
    # on_evaluate is a no-op in save mode, so no baseline was set yet.
    assert cb.monitor._baseline_snapshot is None
    cb.on_save(args=None, state=_state(0), control=None, model=model)
    assert cb.monitor._baseline_snapshot is not None


@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_hf_callback_bad_event():
    with pytest.raises(ValueError, match="on_event"):
        SemanticSentryCallback(_anchors(), on_event="bogus")


@pytest.mark.skipif(not cb_mod._HAS_HF, reason="transformers not installed")
def test_hf_callback_stops_on_plateau():
    model = nn.Linear(4, 4)
    adapter = _adapter()  # identical embeddings each call -> geometry plateaus
    monitor = DriftMonitor(track_temporal=True, plateau_k=3)
    cb = SemanticSentryCallback(_anchors(), adapter=adapter, monitor=monitor,
                                stop_on_plateau=True)
    control = types.SimpleNamespace(should_training_stop=False)
    for step in range(7):
        cb.on_evaluate(args=None, state=_state(step), control=control, model=model)
    assert control.should_training_stop is True
    assert cb.last_comparison.metadata["temporal"]["plateau"] is True


def test_lightning_callback_guarded():
    if cb_mod._HAS_LIGHTNING:  # pragma: no cover - depends on env
        SemanticSentryLightningCallback(_anchors(), adapter=_adapter())
    else:
        with pytest.raises(ImportError):
            SemanticSentryLightningCallback(_anchors(), adapter=_adapter())
