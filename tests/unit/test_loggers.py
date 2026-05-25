"""Tests for WandbLogger / MLflowLogger using injected fake backends."""

import sys
import types

import pytest

from semantic_sentry.core.comparison import Comparison


def _comparison():
    return Comparison(
        snapshot_v0_hash="a",
        snapshot_v1_hash="b",
        global_metrics={"cka": 0.8, "nps": 0.6, "isotropy_delta": -0.01},
        per_tower_metrics={"encoder": {"cka": 0.8}},
        metadata={"downstream": {"RetrievalEvaluator": -0.05}},
    )


# --- wandb ------------------------------------------------------------------

class _FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


def test_wandb_logger_logs_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())
    from semantic_sentry.integrations.wandb_logger import WandbLogger

    run = _FakeRun()
    logger = WandbLogger(run=run)
    logger.log_comparison(_comparison(), step=3)

    payload, step = run.logged[0]
    assert step == 3
    assert payload["drift/cka"] == 0.8
    assert payload["drift/encoder/cka"] == 0.8
    assert payload["drift/downstream/RetrievalEvaluator"] == -0.05
    assert payload["drift/severity"] == 3  # critical (nps 0.6 < 0.70)
    assert run.summary["drift/severity_label"] == "critical"


def test_wandb_logger_does_not_finish_supplied_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())
    from semantic_sentry.integrations.wandb_logger import WandbLogger

    run = _FakeRun()
    WandbLogger(run=run).close()
    assert run.finished is False


def test_wandb_logger_missing_dep(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # forces ImportError
    from semantic_sentry.integrations.wandb_logger import WandbLogger

    with pytest.raises(ImportError, match="wandb"):
        WandbLogger(run=_FakeRun())


# --- mlflow -----------------------------------------------------------------

class _FakeMlflow:
    def __init__(self):
        self.metrics = []
        self.tags = {}
        self.params = {}
        self._active = object()  # pretend a run is already active

    def active_run(self):
        return self._active

    def log_metrics(self, metrics, step=None, run_id=None):
        self.metrics.append((metrics, step, run_id))

    def log_metric(self, key, value, run_id=None):
        self.metrics.append(({key: value}, None, run_id))

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def end_run(self):
        pass


def test_mlflow_logger_logs_metrics(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from semantic_sentry.integrations.mlflow_logger import MLflowLogger

    logger = MLflowLogger()
    logger.log_comparison(_comparison(), step=2)

    metrics, step, _ = fake.metrics[0]
    assert step == 2
    assert metrics["drift/cka"] == 0.8
    assert metrics["drift/downstream/RetrievalEvaluator"] == -0.05
    assert fake.tags["drift/severity"] == "critical"


def test_mlflow_logger_report_splits_numeric_and_param(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from semantic_sentry.integrations.mlflow_logger import MLflowLogger

    MLflowLogger().log_report({"n_steps": 100, "note": "hello"})
    assert any(m[0].get("drift/report/n_steps") == 100.0 for m in fake.metrics)
    assert fake.params["drift/report/note"] == "hello"
