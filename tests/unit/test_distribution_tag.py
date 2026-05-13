"""Tests for AnchorSet.distribution_tag and its threading through Snapshot/Comparison (H2)."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.exceptions import AnchorSetMismatchError
from semantic_sentry.probes.anchor_set import AnchorSet


def _make_anchor(tag: str = "") -> AnchorSet:
    return AnchorSet(
        inputs=[f"x_{i}" for i in range(30)],
        modality="text",
        distribution_tag=tag,
    )


def _make_adapter(rng_seed: int) -> CustomAdapter:
    Z = np.random.default_rng(rng_seed).standard_normal((30, 16)).astype(np.float32)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return CustomAdapter(
        encode_fn=lambda inputs: {"encoder": Z[: len(inputs)]},
        tower_names=["encoder"],
    )


class TestDistributionTag:
    def test_default_tag_is_empty(self):
        a = AnchorSet(inputs=["a", "b"])
        assert a.distribution_tag == ""

    def test_tag_round_trips_through_snapshot_metadata(self):
        anchor = _make_anchor(tag="training-dist")
        monitor = DriftMonitor()
        snap = monitor.snapshot(model=None, anchor_set=anchor, adapter=_make_adapter(0))
        assert snap.metadata.get("distribution_tag") == "training-dist"

    def test_tag_threads_into_comparison_metadata(self):
        anchor = _make_anchor(tag="OOD")
        monitor = DriftMonitor()
        s0 = monitor.snapshot(model=None, anchor_set=anchor, adapter=_make_adapter(0))
        s1 = monitor.snapshot(model=None, anchor_set=anchor, adapter=_make_adapter(1))
        comparison = monitor.compare(s0, s1)
        assert comparison.metadata.get("distribution_tag") == "OOD"

    def test_mismatched_tags_raise(self):
        a_train = _make_anchor(tag="training-dist")
        a_ood = AnchorSet(
            inputs=[f"x_{i}" for i in range(30)],
            modality="text",
            distribution_tag="OOD",
            # Force the same version_hash so the mismatch raises on the tag,
            # not on the underlying inputs (which here happen to be identical).
        )
        # Sanity: by construction these have identical inputs -> identical version_hash.
        assert a_train.version_hash == a_ood.version_hash

        monitor = DriftMonitor()
        s0 = monitor.snapshot(model=None, anchor_set=a_train, adapter=_make_adapter(0))
        s1 = monitor.snapshot(model=None, anchor_set=a_ood, adapter=_make_adapter(1))

        with pytest.raises(AnchorSetMismatchError, match="distribution tag"):
            monitor.compare(s0, s1)

    def test_no_tag_on_either_side_is_legacy_compatible(self):
        anchor = _make_anchor(tag="")  # default
        monitor = DriftMonitor()
        s0 = monitor.snapshot(model=None, anchor_set=anchor, adapter=_make_adapter(0))
        s1 = monitor.snapshot(model=None, anchor_set=anchor, adapter=_make_adapter(1))
        comparison = monitor.compare(s0, s1)
        # No tag is recorded when none was set.
        assert "distribution_tag" not in comparison.metadata


class TestConsoleLoggerExport:
    def test_console_logger_importable_from_package(self):
        from semantic_sentry.integrations import ConsoleLogger, DriftLogger

        assert issubclass(ConsoleLogger, DriftLogger)
