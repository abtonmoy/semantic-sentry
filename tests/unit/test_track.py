"""Tests for DriftMonitor.track() / set_baseline() and downstream wiring."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from semantic_sentry import AnchorSet, DriftMonitor
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.comparison import Comparison
from semantic_sentry.evaluation import ClassificationEvaluator
from semantic_sentry.exceptions import AnchorSetMismatchError

N, D = 60, 16


def _anchors(with_labels=False):
    inputs = [f"text {i}" for i in range(N)]
    labels = tuple(f"c{i % 3}" for i in range(N)) if with_labels else ()
    return AnchorSet(inputs=inputs, labels=labels, modality="text")


def _adapter(mat):
    arr = mat.astype(np.float32)

    def enc(_inputs):
        return torch.from_numpy(arr)

    return CustomAdapter(encode_fn=enc, tower_count=1)


def _model():
    return nn.Linear(4, 4)


@pytest.fixture
def matrices():
    rng = np.random.default_rng(1)
    Z0 = rng.standard_normal((N, D))
    Z1 = Z0 + 0.5 * rng.standard_normal((N, D))
    return Z0, Z1


def test_first_track_sets_baseline_returns_none(matrices):
    Z0, _ = matrices
    monitor = DriftMonitor()
    result = monitor.track(_model(), _anchors(), adapter=_adapter(Z0))
    assert result is None
    assert monitor._baseline_snapshot is not None


def test_second_track_returns_comparison(matrices):
    Z0, Z1 = matrices
    monitor = DriftMonitor()
    monitor.track(_model(), _anchors(), adapter=_adapter(Z0))
    cmp = monitor.track(_model(), _anchors(), adapter=_adapter(Z1))
    assert isinstance(cmp, Comparison)
    assert {"cka", "nps", "isotropy_delta"} <= set(cmp.global_metrics)
    # Real drift was injected -> CKA below the identity ceiling.
    assert cmp.global_metrics["cka"] < 0.999


def test_set_baseline_pins_reference(matrices):
    Z0, Z1 = matrices
    monitor = DriftMonitor()
    monitor.set_baseline(_model(), _anchors(), adapter=_adapter(Z0))
    cmp = monitor.track(_model(), _anchors(), adapter=_adapter(Z1))
    assert isinstance(cmp, Comparison)


def test_track_logs_to_logger(matrices):
    Z0, Z1 = matrices

    class RecordingLogger:
        def __init__(self):
            self.calls = []

        def log_comparison(self, comparison, step=None):
            self.calls.append((comparison, step))

    logger = RecordingLogger()
    monitor = DriftMonitor()
    monitor.track(_model(), _anchors(), adapter=_adapter(Z0), logger=logger, step=0)
    monitor.track(_model(), _anchors(), adapter=_adapter(Z1), logger=logger, step=1)
    # Baseline call logs nothing; second call logs once with its step.
    assert len(logger.calls) == 1
    assert logger.calls[0][1] == 1


def test_downstream_evaluators_reported(matrices):
    Z0, Z1 = matrices
    monitor = DriftMonitor()
    monitor.track(_model(), _anchors(with_labels=True), adapter=_adapter(Z0))
    cmp = monitor.track(
        _model(),
        _anchors(with_labels=True),
        adapter=_adapter(Z1),
        evaluators=[ClassificationEvaluator(k=5)],
    )
    assert "downstream" in cmp.metadata
    assert "ClassificationEvaluator" in cmp.metadata["downstream"]
    assert isinstance(cmp.metadata["downstream"]["ClassificationEvaluator"], float)


def test_keep_on_device_path(matrices):
    Z0, Z1 = matrices
    monitor = DriftMonitor()
    assert monitor.track(_model(), _anchors(), adapter=_adapter(Z0),
                         keep_on_device=True) is None
    cmp = monitor.track(_model(), _anchors(), adapter=_adapter(Z1),
                        keep_on_device=True)
    assert isinstance(cmp, Comparison)
    assert cmp.metadata.get("backend") == "torch"
    assert {"cka", "nps", "isotropy_delta"} <= set(cmp.global_metrics)


def test_keep_on_device_matches_numpy_path(matrices):
    """Torch-backend track() should agree with the numpy snapshot path."""
    Z0, Z1 = matrices

    m_np = DriftMonitor()
    m_np.track(_model(), _anchors(), adapter=_adapter(Z0))
    cmp_np = m_np.track(_model(), _anchors(), adapter=_adapter(Z1))

    m_th = DriftMonitor()
    m_th.track(_model(), _anchors(), adapter=_adapter(Z0), keep_on_device=True)
    cmp_th = m_th.track(_model(), _anchors(), adapter=_adapter(Z1), keep_on_device=True)

    assert cmp_th.global_metrics["cka"] == pytest.approx(
        cmp_np.global_metrics["cka"], abs=1e-4
    )
    assert cmp_th.global_metrics["nps"] == pytest.approx(
        cmp_np.global_metrics["nps"], abs=1e-6
    )


def test_keep_on_device_anchor_change_raises(matrices):
    Z0, Z1 = matrices
    monitor = DriftMonitor()
    monitor.track(_model(), _anchors(), adapter=_adapter(Z0), keep_on_device=True)
    other = AnchorSet(inputs=[f"different {i}" for i in range(N)], modality="text")
    with pytest.raises(AnchorSetMismatchError):
        monitor.track(_model(), other, adapter=_adapter(Z1), keep_on_device=True)
