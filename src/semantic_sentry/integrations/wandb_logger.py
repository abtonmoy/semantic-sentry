"""Weights & Biases drift logger.

Flattens `Comparison` / `ClassificationResult` into `wandb.log()` calls so
drift metrics show up next to the loss curves in the run dashboard. Import is
lazy — ``wandb`` is only required when you actually construct the logger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from semantic_sentry.integrations.base import DriftLogger

if TYPE_CHECKING:
    from semantic_sentry.core.classification import ClassificationResult
    from semantic_sentry.core.comparison import Comparison


class WandbLogger(DriftLogger):
    """Log drift metrics to a Weights & Biases run.

    Args:
        run: An existing ``wandb.Run`` (from ``wandb.init()``). If omitted,
            a run is started with ``wandb.init(project=..., **init_kwargs)``.
        project: Project name used only when ``run`` is None.
        prefix: Key prefix for every logged metric (default ``"drift"``).
        init_kwargs: Extra kwargs forwarded to ``wandb.init`` when starting
            a run.
    """

    def __init__(
        self,
        run: Any | None = None,
        project: str | None = None,
        prefix: str = "drift",
        **init_kwargs: Any,
    ) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise ImportError(
                "WandbLogger requires the 'wandb' extra: pip install "
                "semantic-sentry[wandb]"
            ) from exc

        self._wandb = wandb
        self._prefix = prefix.rstrip("/")
        if run is not None:
            self._run = run
            self._owns_run = False
        else:
            self._run = wandb.init(project=project, **init_kwargs)
            self._owns_run = True

    def _key(self, *parts: str) -> str:
        return "/".join((self._prefix, *parts))

    def _comparison_payload(self, comparison: Comparison) -> dict[str, float]:
        payload: dict[str, float] = {
            self._key(name): float(value)
            for name, value in comparison.global_metrics.items()
        }
        if comparison.per_tower_metrics:
            for tower, metrics in comparison.per_tower_metrics.items():
                for name, value in metrics.items():
                    payload[self._key(tower, name)] = float(value)
        downstream = comparison.metadata.get("downstream")
        if isinstance(downstream, dict):
            for name, value in downstream.items():
                payload[self._key("downstream", name)] = float(value)
        return payload

    def log_comparison(self, comparison: Comparison, step: int | None = None) -> None:
        """Log all comparison metrics + a numeric severity ordinal."""
        payload = self._comparison_payload(comparison)
        # Numeric severity so it plots; the string lives in the run summary.
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        payload[self._key("severity")] = order.get(comparison.severity.value, -1)
        self._run.log(payload, step=step)
        self._run.summary[self._key("severity_label")] = comparison.severity.value

    def log_classification(
        self,
        result: ClassificationResult,
        input_id: str | None = None,
    ) -> None:
        """Log a single classification's local NPS + confidence ordinal."""
        order = {"low": 0, "medium": 1, "high": 2}
        payload = {
            self._key("classify", "local_nps"): float(result.local_nps),
            self._key("classify", "confidence"): order.get(result.confidence.value, -1),
        }
        self._run.log(payload)

    def log_report(self, report: dict[str, Any]) -> None:
        """Write a free-form report dict into the run summary."""
        for key, value in report.items():
            self._run.summary[self._key("report", key)] = value

    def close(self) -> None:
        """Finish the run only if this logger started it."""
        if self._owns_run:
            self._run.finish()
