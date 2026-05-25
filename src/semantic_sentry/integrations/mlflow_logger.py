"""MLflow drift logger.

Flattens `Comparison` / `ClassificationResult` into ``mlflow.log_metric(s)``
calls against the active (or a supplied) run. Import is lazy — ``mlflow`` is
only required when you construct the logger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from semantic_sentry.integrations.base import DriftLogger

if TYPE_CHECKING:
    from semantic_sentry.core.classification import ClassificationResult
    from semantic_sentry.core.comparison import Comparison


class MLflowLogger(DriftLogger):
    """Log drift metrics to MLflow.

    Args:
        run_id: Optional explicit run id to log against. If omitted, metrics
            go to the active run; one is started lazily if none is active.
        prefix: Metric-key prefix (default ``"drift"``). MLflow metric keys
            allow ``/``, so nested keys read as ``drift/cka`` etc.
    """

    def __init__(self, run_id: str | None = None, prefix: str = "drift") -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise ImportError(
                "MLflowLogger requires the 'mlflow' extra: pip install "
                "semantic-sentry[mlflow]"
            ) from exc

        self._mlflow = mlflow
        self._prefix = prefix.rstrip("/")
        self._run_id = run_id
        self._owns_run = False
        if run_id is None and mlflow.active_run() is None:
            mlflow.start_run()
            self._run_id = mlflow.active_run().info.run_id
            self._owns_run = True

    def _key(self, *parts: str) -> str:
        return "/".join((self._prefix, *parts))

    def _comparison_metrics(self, comparison: Comparison) -> dict[str, float]:
        metrics: dict[str, float] = {
            self._key(name): float(value)
            for name, value in comparison.global_metrics.items()
        }
        if comparison.per_tower_metrics:
            for tower, tower_metrics in comparison.per_tower_metrics.items():
                for name, value in tower_metrics.items():
                    metrics[self._key(tower, name)] = float(value)
        downstream = comparison.metadata.get("downstream")
        if isinstance(downstream, dict):
            for name, value in downstream.items():
                metrics[self._key("downstream", name)] = float(value)
        return metrics

    def log_comparison(self, comparison: Comparison, step: int | None = None) -> None:
        """Log comparison metrics (with step) + severity as a tag."""
        metrics = self._comparison_metrics(comparison)
        self._mlflow.log_metrics(metrics, step=step, run_id=self._run_id)
        self._mlflow.set_tag(self._key("severity"), comparison.severity.value)

    def log_classification(
        self,
        result: ClassificationResult,
        input_id: str | None = None,
    ) -> None:
        """Log a single classification's local NPS."""
        self._mlflow.log_metrics(
            {self._key("classify", "local_nps"): float(result.local_nps)},
            run_id=self._run_id,
        )

    def log_report(self, report: dict[str, Any]) -> None:
        """Log a report dict — numeric values as metrics, the rest as params."""
        for key, value in report.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._mlflow.log_metric(self._key("report", key), float(value),
                                        run_id=self._run_id)
            else:
                self._mlflow.log_param(self._key("report", key), value)

    def close(self) -> None:
        """End the run only if this logger started it."""
        if self._owns_run:
            self._mlflow.end_run()
