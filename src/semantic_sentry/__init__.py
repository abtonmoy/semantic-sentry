"""SemanticSentry: Universal semantic drift detection for any embedding space."""

__version__ = "0.2.0"

from semantic_sentry.core.classification import ClassificationResult, ConfidenceLevel
from semantic_sentry.core.comparison import AlertSeverity, Comparison
from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.evaluation import (
    ClassificationEvaluator,
    Evaluator,
    EvaluatorRegistry,
    RetrievalEvaluator,
)
from semantic_sentry.exceptions import (
    AdapterDetectionError,
    AnchorSetMismatchError,
    CalibrationNotFoundError,
    EmbeddingDimError,
    MetricRegistrationError,
    NoComparisonError,
    SemanticSentryError,
    SnapshotCorruptionError,
    TowerMismatchError,
)
from semantic_sentry.probes.anchor_set import AnchorSet

__all__ = [
    "DriftMonitor",
    "SemanticSentryError",
    "AdapterDetectionError",
    "TowerMismatchError",
    "EmbeddingDimError",
    "AnchorSetMismatchError",
    "MetricRegistrationError",
    "CalibrationNotFoundError",
    "SnapshotCorruptionError",
    "NoComparisonError",
    "Snapshot",
    "Comparison",
    "AlertSeverity",
    "ClassificationResult",
    "ConfidenceLevel",
    "AnchorSet",
    "Evaluator",
    "EvaluatorRegistry",
    "RetrievalEvaluator",
    "ClassificationEvaluator",
]
