"""Training-framework callbacks that drive `DriftMonitor.track()`.

Drop one into your trainer and drift gets measured against a rolling baseline
at every eval/validation event, with no changes to your training code. Both
callbacks degrade gracefully: if the framework isn't installed, importing the
class raises a clear error rather than failing at module import.

Example (HuggingFace):

    from semantic_sentry import DriftMonitor, AnchorSet
    from semantic_sentry.adapters.huggingface import HuggingFaceAdapter
    from semantic_sentry.integrations.callbacks import SemanticSentryCallback
    from semantic_sentry.integrations.wandb_logger import WandbLogger

    adapter = HuggingFaceAdapter(model, tokenizer)
    cb = SemanticSentryCallback(anchors, adapter=adapter, logger=WandbLogger())
    trainer = Trainer(..., callbacks=[cb])
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from semantic_sentry.core.monitor import DriftMonitor

if TYPE_CHECKING:
    from semantic_sentry.adapters.base import EncoderAdapter
    from semantic_sentry.integrations.base import DriftLogger
    from semantic_sentry.probes.anchor_set import AnchorSet


def _plateau_reached(comparison: Any) -> bool:
    """True if the comparison carries a fired temporal plateau signal."""
    if comparison is None:
        return False
    return bool(comparison.metadata.get("temporal", {}).get("plateau"))


# --- HuggingFace Trainer -----------------------------------------------------

try:
    from transformers import TrainerCallback as _HFTrainerCallback

    _HAS_HF = True
except ImportError:  # pragma: no cover - depends on optional dep
    _HAS_HF = False

    class _HFTrainerCallback:  # type: ignore[no-redef]
        """Placeholder base so the subclass definition below is importable."""


class SemanticSentryCallback(_HFTrainerCallback):
    """`transformers.TrainerCallback` that tracks drift on each evaluation.

    The first evaluation establishes the baseline; subsequent ones compare
    against it. Pass an ``adapter`` that wraps the model being trained
    (HF models need a tokenizer, so they can't be auto-detected).

    Args:
        anchor_set: Fixed anchor set.
        adapter: Encoder adapter for the model under training.
        logger: Optional `DriftLogger` (e.g. `WandbLogger`).
        monitor: Optional pre-built `DriftMonitor` (one is created otherwise).
            Configure live behaviour there — e.g.
            ``DriftMonitor(baseline_mode="previous", track_temporal=True,
            async_mode=True)``.
        evaluators: Optional downstream evaluators (see `DriftMonitor.track`).
        on_event: Which trainer hook fires tracking when ``every_n_steps`` is
            0 — ``"evaluate"`` (default) or ``"save"``.
        every_n_steps: When > 0, track on ``on_step_end`` every N optimizer
            steps instead of at eval/save events — fine-grained, per-step
            cadence. Pair with a monitor built with ``keep_on_device``/
            ``async_mode``/``max_inflight`` and a small anchor set to keep the
            overhead off the critical path. ``1`` means literally every step.
        stop_on_plateau: When True, set ``control.should_training_stop`` once
            the temporal plateau signal fires (requires a monitor built with
            ``track_temporal=True``).
    """

    def __init__(
        self,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None,
        logger: DriftLogger | None = None,
        monitor: DriftMonitor | None = None,
        evaluators: list[Any] | dict[str, Any] | None = None,
        on_event: str = "evaluate",
        every_n_steps: int = 0,
        stop_on_plateau: bool = False,
    ) -> None:
        if not _HAS_HF:
            raise ImportError(
                "SemanticSentryCallback requires 'transformers': "
                "pip install transformers"
            )
        if on_event not in ("evaluate", "save"):
            raise ValueError(f"on_event must be 'evaluate' or 'save', got {on_event!r}")
        if every_n_steps < 0:
            raise ValueError(f"every_n_steps must be >= 0, got {every_n_steps}")
        self.anchor_set = anchor_set
        self.adapter = adapter
        self.logger = logger
        self.monitor = monitor or DriftMonitor()
        self.evaluators = evaluators
        self.on_event = on_event
        self.every_n_steps = every_n_steps
        self.stop_on_plateau = stop_on_plateau
        self.last_comparison: Any | None = None

    def _track(self, state: Any, model: Any) -> bool:
        """Track drift; return True if training should stop (plateau reached)."""
        if model is None:
            return False
        step = getattr(state, "global_step", None)
        result = self.monitor.track(
            model,
            self.anchor_set,
            step=step,
            adapter=self.adapter,
            logger=self.logger,
            evaluators=self.evaluators,
        )
        # In async mode track() returns a Future; the freshest finished
        # comparison lives on the monitor (it may lag by one eval).
        comparison = self.monitor.last_result if isinstance(result, Future) else result
        if comparison is not None:
            self.last_comparison = comparison
        return _plateau_reached(comparison) if self.stop_on_plateau else False

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        """Track drift every ``every_n_steps`` optimizer steps (per-step mode)."""
        if self.every_n_steps > 0:
            step = getattr(state, "global_step", 0) or 0
            if (step % self.every_n_steps == 0
                    and self._track(state, kwargs.get("model"))
                    and control is not None):
                control.should_training_stop = True
        return control

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
        """Track drift after each evaluation phase (when not in per-step mode)."""
        if (self.every_n_steps == 0
                and self.on_event == "evaluate"
                and self._track(state, kwargs.get("model"))
                and control is not None):
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        """Track drift after each checkpoint save (when on_event='save')."""
        if (self.every_n_steps == 0
                and self.on_event == "save"
                and self._track(state, kwargs.get("model"))
                and control is not None):
            control.should_training_stop = True
        return control


# --- PyTorch Lightning -------------------------------------------------------

try:
    from lightning.pytorch.callbacks import (
        Callback as _LightningCallback,  # type: ignore[import-not-found]
    )

    _HAS_LIGHTNING = True
except ImportError:  # pragma: no cover
    try:
        from pytorch_lightning.callbacks import (
            Callback as _LightningCallback,  # type: ignore[import-not-found]
        )

        _HAS_LIGHTNING = True
    except ImportError:
        _HAS_LIGHTNING = False

        class _LightningCallback:  # type: ignore[no-redef]
            """Placeholder base so the subclass definition is importable."""


class SemanticSentryLightningCallback(_LightningCallback):
    """Lightning `Callback` that tracks drift at the end of each validation.

    Args:
        anchor_set: Fixed anchor set.
        adapter: Encoder adapter for the `LightningModule` under training.
        logger: Optional `DriftLogger`.
        monitor: Optional pre-built `DriftMonitor` (configure live behaviour
            there, as for the HF callback).
        evaluators: Optional downstream evaluators.
        every_n_steps: When > 0, track on ``on_train_batch_end`` every N steps
            (per-step cadence) instead of at validation end.
        stop_on_plateau: When True, set ``trainer.should_stop`` once the
            temporal plateau signal fires.
    """

    def __init__(
        self,
        anchor_set: AnchorSet,
        adapter: EncoderAdapter | None = None,
        logger: DriftLogger | None = None,
        monitor: DriftMonitor | None = None,
        evaluators: list[Any] | dict[str, Any] | None = None,
        every_n_steps: int = 0,
        stop_on_plateau: bool = False,
    ) -> None:
        if not _HAS_LIGHTNING:
            raise ImportError(
                "SemanticSentryLightningCallback requires 'lightning' or "
                "'pytorch-lightning'"
            )
        if every_n_steps < 0:
            raise ValueError(f"every_n_steps must be >= 0, got {every_n_steps}")
        self.anchor_set = anchor_set
        self.adapter = adapter
        self.logger = logger
        self.monitor = monitor or DriftMonitor()
        self.evaluators = evaluators
        self.every_n_steps = every_n_steps
        self.stop_on_plateau = stop_on_plateau
        self.last_comparison: Any | None = None

    def _track(self, trainer, pl_module) -> None:
        step = getattr(trainer, "global_step", None)
        result = self.monitor.track(
            pl_module,
            self.anchor_set,
            step=step,
            adapter=self.adapter,
            logger=self.logger,
            evaluators=self.evaluators,
        )
        comparison = self.monitor.last_result if isinstance(result, Future) else result
        if comparison is not None:
            self.last_comparison = comparison
        if self.stop_on_plateau and _plateau_reached(comparison):
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):  # noqa: ANN001
        """Track drift every ``every_n_steps`` steps (per-step mode)."""
        if self.every_n_steps > 0:
            step = getattr(trainer, "global_step", 0) or 0
            if step % self.every_n_steps == 0:
                self._track(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):  # noqa: ANN001
        """Track drift at validation end (when not in per-step mode)."""
        if self.every_n_steps == 0:
            self._track(trainer, pl_module)
