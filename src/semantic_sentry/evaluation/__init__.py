"""Downstream task evaluators.

These compute *measured* task performance (and v0->v1 deltas) on a labelled
anchor set — a complement to the label-free geometric metrics. Wire them into
`DriftMonitor.compare(..., anchor_set=, evaluators=)` or
`DriftMonitor.track(..., evaluators=)` to get the downstream delta reported
under `Comparison.metadata["downstream"]`.
"""

from semantic_sentry.evaluation.registry import (
    ClassificationEvaluator,
    Evaluator,
    EvaluatorRegistry,
    RetrievalEvaluator,
)

__all__ = [
    "Evaluator",
    "EvaluatorRegistry",
    "RetrievalEvaluator",
    "ClassificationEvaluator",
]
