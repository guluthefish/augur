"""Common abstractions for Lightning models used in Augur."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _callable_name(factory: Callable[..., Any] | None) -> str | None:
    """Return a readable name for an optimizer or scheduler factory."""
    if factory is None:
        return None

    return getattr(factory, "__name__", factory.__class__.__name__)


class ModelABC(LightningModule, ABC):
    """Abstract base class for Lightning models.

    Subclasses are expected to implement :meth:`forward` and :meth:`model_step`.
    The default train/validation/test steps delegate to :meth:`model_step`,
    log a stage-specific loss, and optionally log any extra metrics returned by
    the subclass.
    """

    def __init__(
        self: ModelABC,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize shared optimization configuration.

        Parameters
        ----------
        optimizer_factory:
            Callable that builds an optimizer from ``self.parameters()`` and any
            provided ``optimizer_kwargs``. If omitted, subclasses should
            override :meth:`configure_optimizers`.
        optimizer_kwargs:
            Keyword arguments forwarded to ``optimizer_factory``.
        lr_scheduler_factory:
            Optional callable that builds a learning-rate scheduler from the
            optimizer returned by ``optimizer_factory``.
        lr_scheduler_kwargs:
            Keyword arguments forwarded to ``lr_scheduler_factory``.
        lr_scheduler_config:
            Optional Lightning scheduler metadata such as ``monitor``,
            ``interval``, and ``frequency``.
        """
        super().__init__()

        if optimizer_factory is None and optimizer_kwargs:
            raise ValueError(
                "optimizer_kwargs were provided without an optimizer_factory."
            )
        if optimizer_factory is None and lr_scheduler_factory is not None:
            raise ValueError(
                "lr_scheduler_factory requires optimizer_factory when using the "
                "default configure_optimizers implementation."
            )
        if lr_scheduler_factory is None and lr_scheduler_kwargs:
            raise ValueError(
                "lr_scheduler_kwargs were provided without an lr_scheduler_factory."
            )
        if lr_scheduler_factory is None and lr_scheduler_config:
            raise ValueError(
                "lr_scheduler_config was provided without an lr_scheduler_factory."
            )

        self.optimizer_factory = optimizer_factory
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.lr_scheduler_factory = lr_scheduler_factory
        self.lr_scheduler_kwargs = dict(lr_scheduler_kwargs or {})
        self.lr_scheduler_config = dict(lr_scheduler_config or {})

        self.save_hyperparameters(
            {
                "optimizer_factory": _callable_name(optimizer_factory),
                "optimizer_kwargs": self.optimizer_kwargs,
                "lr_scheduler_factory": _callable_name(lr_scheduler_factory),
                "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
                "lr_scheduler_config": self.lr_scheduler_config,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> ModelABC:
        """Construct a model from a configuration dictionary."""
        raise NotImplementedError(
            "Subclasses of ModelABC should implement from_config() to support construction from a configuration dictionary."
        )

    @abstractmethod
    def forward(self: ModelABC, *args: Any, **kwargs: Any) -> Any:
        """Run the model forward pass."""
        raise NotImplementedError(
            "Subclasses of ModelABC must implement forward() to define the model's forward pass."
        )

    @abstractmethod
    def model_step(
        self: ModelABC,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Compute the loss and optional metrics for one batch."""
        raise NotImplementedError(
            "Subclasses of ModelABC must implement model_step() to compute the loss and metrics for a batch."
        )

    def configure_optimizers(self: ModelABC) -> Any:
        """Create the optimizer and optional learning-rate scheduler."""
        if self.optimizer_factory is None:
            raise NotImplementedError(
                "Either provide optimizer_factory to ModelABC or override "
                "configure_optimizers() in the subclass."
            )

        optimizer = self.optimizer_factory(
            self.parameters(),
            **self.optimizer_kwargs,
        )
        if self.lr_scheduler_factory is None:
            return optimizer

        scheduler = self.lr_scheduler_factory(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        if not self.lr_scheduler_config:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.lr_scheduler_config,
            },
        }

    def training_step(  # pylint: disable=arguments-differ
        self: ModelABC, batch: Any, batch_idx: int
    ) -> Tensor:
        """Run one training step and log the resulting metrics."""
        return self._shared_step(batch, batch_idx, stage="train", on_step=True)

    def validation_step(  # pylint: disable=arguments-differ
        self: ModelABC, batch: Any, batch_idx: int
    ) -> Tensor:
        """Run one validation step and log the resulting metrics."""
        return self._shared_step(batch, batch_idx, stage="val", on_step=False)

    def test_step(  # pylint: disable=arguments-differ
        self: ModelABC, batch: Any, batch_idx: int
    ) -> Tensor:
        """Run one test step and log the resulting metrics."""
        return self._shared_step(batch, batch_idx, stage="test", on_step=False)

    def _shared_step(
        self: ModelABC,
        batch: Any,
        batch_idx: int,
        *,
        stage: str,
        on_step: bool,
    ) -> Tensor:
        """Execute a stage-specific step and handle metric logging."""
        loss, metrics = self._parse_step_output(
            self.model_step(batch, batch_idx, stage),
            stage=stage,
        )
        batch_size = self._infer_batch_size(batch)
        sync_dist = self._should_sync_dist_logging()

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=stage != "test",
            logger=True,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        if metrics:
            self.log_dict(
                metrics,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

        return loss

    def _should_sync_dist_logging(self: ModelABC) -> bool:
        """Enable distributed metric reduction only when multiple ranks are active."""
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return False
        return bool(getattr(trainer, "world_size", 1) > 1)

    def _parse_step_output(
        self: ModelABC,
        step_output: Tensor | tuple[Tensor, dict[str, Any]],
        *,
        stage: str,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Normalize model_step output to a loss tensor plus metric dict."""
        if isinstance(step_output, Tensor):
            loss = step_output
            metrics: dict[str, Any] = {}
        else:
            try:
                loss, metrics = step_output
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "model_step() must return either a loss tensor or a "
                    "(loss, metrics) tuple."
                ) from exc

        if not isinstance(loss, Tensor):
            raise TypeError("model_step() must return a torch.Tensor loss.")
        if not isinstance(metrics, dict):
            raise TypeError(
                "The metrics returned by model_step() must be provided as a dict."
            )

        formatted_metrics: dict[str, Any] = {}
        for name, value in metrics.items():
            metric_name = self._format_metric_name(stage, name)
            if value is not None and metric_name != f"{stage}/loss":
                formatted_metrics[metric_name] = value
        return loss, formatted_metrics

    def _format_metric_name(self: ModelABC, stage: str, name: str) -> str:
        """Prefix metric names with the stage when needed."""
        return name if "/" in name else f"{stage}/{name}"

    def _infer_batch_size(self: ModelABC, batch: Any) -> int | None:
        """Best-effort batch-size inference for Lightning logging."""
        if isinstance(batch, Tensor):
            return int(batch.shape[0]) if batch.ndim > 0 else 1
        if isinstance(batch, dict):
            for value in batch.values():
                batch_size = self._infer_batch_size(value)
                if batch_size is not None:
                    return batch_size
            return None
        if isinstance(batch, (tuple, list)):
            for value in batch:
                batch_size = self._infer_batch_size(value)
                if batch_size is not None:
                    return batch_size
        return None
