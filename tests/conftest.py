"""Shared pytest fixtures.

The `_reset_metric_registry` fixture clears any custom-registered metric
between tests so that a metric registered in test A does not leak into
test B. Built-in metrics (`cka`, `nps`, `isotropy_delta`) are restored.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_metric_registry():
    """Reset the MetricRegistry singleton between every test."""
    from semantic_sentry.metrics.registry import MetricRegistry

    registry = MetricRegistry()
    registry.reset()
    yield
    registry.reset()
