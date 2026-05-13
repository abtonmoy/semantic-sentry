"""Shared pytest fixtures.

The `_reset_metric_registry` fixture clears any custom-registered metric
between tests so that a metric registered in test A does not leak into
test B. Built-in metrics (`cka`, `nps`, `isotropy_delta`) are restored.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_metric_registry():
    """Reset the MetricRegistry singleton between every test.

    Also resets the BehavioralMetricRegistry (lib_enhancement A1-A5) which
    is a separate singleton — without this, a custom behavioral metric
    registered in one test would leak into others.
    """
    from semantic_sentry.metrics.behavioral import BehavioralMetricRegistry
    from semantic_sentry.metrics.registry import MetricRegistry
    from semantic_sentry.metrics.temporal import reset_temporal

    registry = MetricRegistry()
    behavioral = BehavioralMetricRegistry()
    registry.reset()
    behavioral.reset()
    reset_temporal()
    yield
    registry.reset()
    behavioral.reset()
    reset_temporal()
