"""Tests for the semantic-sentry CLI (compare / gate / info)."""

import argparse

import numpy as np
import pytest

from semantic_sentry import cli
from semantic_sentry.core.snapshot import Snapshot

N, D = 40, 8


def _snapshot(seed, anchor_version="v1"):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((N, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return Snapshot(
        model_id=f"m{seed}",
        checkpoint_hash=f"hash{seed}",
        anchor_set_version=anchor_version,
        tower_count=1,
        tower_names=("encoder",),
        embeddings={"encoder": emb},
    )


@pytest.fixture
def snap_dirs(tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    _snapshot(0).save(base)
    _snapshot(1).save(cand)  # different embeddings -> real drift
    return str(base), str(cand)


@pytest.fixture
def identical_dirs(tmp_path):
    base = tmp_path / "b"
    cand = tmp_path / "c"
    snap = _snapshot(7)
    snap.save(base)
    snap.save(cand)  # identical -> CKA == 1.0
    return str(base), str(cand)


def test_info(snap_dirs, capsys):
    base, _ = snap_dirs
    rc = cli.main(["info", base, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "anchor_set_version" in out


def test_compare(snap_dirs, capsys):
    base, cand = snap_dirs
    rc = cli.main(["compare", base, cand, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "global_metrics" in out and "cka" in out


def test_gate_pass(identical_dirs, capsys):
    base, cand = identical_dirs
    rc = cli.main(["gate", base, cand, "--fail-under", "cka=0.9,nps=0.9"])
    assert rc == 0
    assert "GATE PASSED" in capsys.readouterr().out


def test_gate_fail(snap_dirs, capsys):
    base, cand = snap_dirs
    # Random independent embeddings -> CKA well below 0.99.
    rc = cli.main(["gate", base, cand, "--fail-under", "cka=0.99"])
    assert rc == 1
    assert "GATE FAILED" in capsys.readouterr().out


def test_gate_requires_threshold(snap_dirs):
    base, cand = snap_dirs
    with pytest.raises(SystemExit):
        cli.main(["gate", base, cand])


def test_gate_missing_metric_fails(snap_dirs, capsys):
    base, cand = snap_dirs
    rc = cli.main(["gate", base, cand, "--fail-under", "nonexistent=0.5"])
    assert rc == 1
    assert "metric missing" in capsys.readouterr().out


def test_parse_thresholds_bad_spec():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_thresholds("cka")  # no '='
