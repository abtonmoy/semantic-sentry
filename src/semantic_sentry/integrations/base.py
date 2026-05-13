"""Base class for drift loggers."""

from abc import ABC, abstractmethod
from typing import Any

from semantic_sentry.core.classification import ClassificationResult
from semantic_sentry.core.comparison import Comparison


class DriftLogger(ABC):
    """Abstract base class for drift logging integrations.

    Implementations can log drift metrics to various backends
    (Weights & Biases, MLflow, webhooks, etc.).
    """

    @abstractmethod
    def log_comparison(self, comparison: Comparison, step: int | None = None) -> None:
        """Log a comparison result.

        Args:
            comparison: Comparison result to log
            step: Optional step number
        """
        pass

    @abstractmethod
    def log_classification(
        self,
        result: ClassificationResult,
        input_id: str | None = None
    ) -> None:
        """Log a classification result.

        Args:
            result: Classification result to log
            input_id: Optional identifier for the input
        """
        pass

    @abstractmethod
    def log_report(self, report: dict[str, Any]) -> None:
        """Log a drift report.

        Args:
            report: Report dictionary
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConsoleLogger(DriftLogger):
    """Simple console logger for debugging."""

    def __init__(self, verbose: bool = True):
        """Initialize console logger.

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose

    def log_comparison(self, comparison: Comparison, step: int | None = None) -> None:
        """Log comparison to console."""
        prefix = f"[Step {step}] " if step is not None else ""
        print(f"{prefix}Drift Comparison:")
        print(f"  Severity: {comparison.severity.value}")
        print("  Global Metrics:")
        for name, value in comparison.global_metrics.items():
            print(f"    {name}: {value:.4f}")

    def log_classification(
        self,
        result: ClassificationResult,
        input_id: str | None = None
    ) -> None:
        """Log classification to console."""
        prefix = f"[{input_id}] " if input_id else ""
        print(f"{prefix}Classification:")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence.value}")
        print(f"  Local NPS: {result.local_nps:.4f}")
        if result.drift_warning:
            print(f"  Warning: {result.drift_warning}")

    def log_report(self, report: dict[str, Any]) -> None:
        """Log report to console."""
        print("=" * 50)
        print("DRIFT REPORT")
        print("=" * 50)
        for key, value in report.items():
            print(f"{key}: {value}")
        print("=" * 50)

    def close(self) -> None:
        """No-op for console logger."""
        pass
