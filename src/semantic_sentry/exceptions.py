"""Custom exceptions for SemanticSentry."""


class SemanticSentryError(Exception):
    """Base exception for all SemanticSentry errors."""


class AdapterDetectionError(SemanticSentryError):
    """No adapter matches the provided model class."""


class TowerMismatchError(SemanticSentryError):
    """Base and updated models have different tower counts."""


class EmbeddingDimError(SemanticSentryError):
    """Embedding dimensions differ between checkpoints."""


class AnchorSetMismatchError(SemanticSentryError):
    """Snapshots were captured with different anchor sets."""


class MetricRegistrationError(SemanticSentryError):
    """Custom metric failed determinism or type validation."""


class CalibrationNotFoundError(SemanticSentryError):
    """No calibration profile matches model family."""


class SnapshotCorruptionError(SemanticSentryError):
    """Loaded snapshot fails integrity check."""


class NoComparisonError(SemanticSentryError):
    """classify() called before any compare() has been run."""
