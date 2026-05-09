"""Lightning module that combines a shared tile encoder with task decoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from VexDR.models.model_abc import ModelABC
from VexDR.models.tile_level.factory import get_module_from_config
from VexDR.models.utils import get_lr_scheduler_from_config, get_optimizer_from_config
from VexDR.utils.config import load_yaml_config
from VexDR.utils.metrics import (
    compute_classification_loss,
    compute_regression_loss,
    compute_semantic_segmentation_loss,
)


def _component_name(module: nn.Module) -> str:
    """Return a stable component name for logging and hyperparameters."""
    return module.__class__.__name__


class TileModel(ModelABC):
    """Lightning model for tile-level multi-task learning.

    The encoder weights are shared across tasks, but each task provides its own
    input image. The common multi-task batch format produced by
    ``TCGATileDataset`` looks like:

    ``{"metadata": {...}, <task_name>: {"image": ..., "target": ...}, ...}``

    In that nested format, each decoder consumes the image stored under its own
    task name, so tasks such as magnification and jigmag can encode different
    views of the same patch record.

    Parameters
    ----------
    encoder
        A shared tile encoder module that produces a common representation for all tasks.
    decoders
        A dict mapping task names to decoder modules that consume the shared encoder output.
    task_weights
        Optional dict mapping task names to positive scalar weights for loss aggregation.
    optimizer_factory
        Optional factory function for constructing an optimizer, e.g. from a config.
    optimizer_kwargs
        Optional dict of keyword arguments to pass to the optimizer factory.
    lr_scheduler_factory
        Optional factory function for constructing a learning rate scheduler, e.g. from a config.
    lr_scheduler_kwargs
        Optional dict of keyword arguments to pass to the learning rate scheduler factory.
    lr_scheduler_config
        Optional dict of additional configuration for the learning rate scheduler,
        such as step intervals.
    task_kwargs
        Optional dict mapping task names to additional keyword arguments for loss computation,
        such as unknown class indices for segmentation tasks.
    """

    def __init__(
        self: TileModel,
        encoder: ModelABC,
        decoders: dict[str, ModelABC],
        task_weights: dict[str, float] | None = None,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        task_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

        if not isinstance(encoder, ModelABC):
            raise TypeError("encoder must be a inherited ModelABC.")
        if not isinstance(decoders, dict):
            raise TypeError("decoders must be a dict of task names to modules.")

        normalized_decoders: dict[str, ModelABC] = {}
        for task_name, decoder in decoders.items():
            if not isinstance(task_name, str) or not task_name:
                raise ValueError(
                    "Each decoder key must be a non-empty task name string."
                )
            if not isinstance(decoder, ModelABC):
                raise TypeError(
                    f"Decoder '{task_name}' must be a inherited ModelABC. "
                    f"Got: {type(decoder)!r}"
                )
            normalized_decoders[task_name] = decoder

        if not task_weights:
            task_weights = {task_name: 1.0 for task_name in normalized_decoders}

        normalized_task_weights = {
            task_name: float(task_weights.get(task_name, 1.0))
            for task_name in normalized_decoders
        }

        # Normalize weights so they sum to 1, but keep relative proportions the same.
        assert all(
            weight > 0.0 for weight in normalized_task_weights.values()
        ), "Task weights must be positive."
        total_weight = sum(normalized_task_weights.values())
        if total_weight > 0:
            normalized_task_weights = {
                task_name: weight / total_weight
                for task_name, weight in normalized_task_weights.items()
            }

        self.encoder = encoder
        self.decoders = nn.ModuleDict(normalized_decoders)
        self.task_weights = normalized_task_weights
        self.task_kwargs = task_kwargs or {}

        self.save_hyperparameters(
            {
                "encoder": _component_name(encoder),
                "decoders": {
                    task_name: _component_name(decoder)
                    for task_name, decoder in self.decoders.items()
                },
                "task_weights": self.task_weights,
                "task_kwargs": self.task_kwargs,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> TileModel:
        """Construct a ``TileModel`` from a configuration dictionary."""
        if not isinstance(config, dict):
            raise TypeError("TileModel config must be provided as a dict.")

        encoder_config = TileModel._resolve_component_config(
            config.get("encoder_config"),
            label="encoder_config",
        )
        encoder = get_module_from_config(encoder_config)

        decoders_config = config.get("decoders_config", None)
        if not isinstance(decoders_config, dict):
            raise TypeError(
                "TileModel configuration field 'decoders_config' must be a dict."
            )

        decoders: dict[str, ModelABC] = {}
        for task_name, decoder_spec in decoders_config.items():
            if decoder_spec is None:
                continue
            decoder_config = TileModel._resolve_component_config(
                decoder_spec,
                label=f"decoders['{task_name}']",
            )
            decoders[task_name] = get_module_from_config(decoder_config)

        (
            default_optimizer_factory,
            default_optimizer_kwargs,
            default_lr_scheduler_factory,
            default_lr_scheduler_kwargs,
            default_lr_scheduler_config,
        ) = TileModel._resolve_optimization_defaults(encoder, decoders)

        if "optimizer" in config:
            optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
                config.get("optimizer", None)
            )
        else:
            optimizer_factory, optimizer_kwargs = (
                default_optimizer_factory,
                default_optimizer_kwargs,
            )

        if "lr_scheduler" in config or "scheduler" in config:
            lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
                get_lr_scheduler_from_config(
                    config.get("lr_scheduler", config.get("scheduler"))
                )
            )
        else:
            (
                lr_scheduler_factory,
                lr_scheduler_kwargs,
                lr_scheduler_config,
            ) = (
                default_lr_scheduler_factory,
                default_lr_scheduler_kwargs,
                default_lr_scheduler_config,
            )

        task_weights = config.get("task_weights", None)
        if task_weights is not None and not isinstance(task_weights, dict):
            raise TypeError("task_weights must be provided as a dict.")

        task_kwargs = config.get("task_kwargs", None)
        if task_kwargs is not None and not isinstance(task_kwargs, dict):
            raise TypeError("task_kwargs must be provided as a dict.")

        return TileModel(
            encoder=encoder,
            decoders=decoders,
            task_weights=task_weights,
            task_kwargs=task_kwargs,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: TileModel,
        batch_or_image: Tensor | dict[str, Any],
    ) -> dict[str, Any]:
        """Run the shared encoder and every task decoder."""
        if isinstance(batch_or_image, Tensor):
            return self._forward_shared_image(batch_or_image)
        if not isinstance(batch_or_image, dict):
            raise TypeError("TileModel expects a torch.Tensor image batch or a dict.")
        if "image" in batch_or_image:
            return self._forward_shared_image(
                self._validate_image_tensor(batch_or_image["image"])
            )
        return self._forward_multitask_batch(batch_or_image)

    def predict_step(  # pylint: disable=arguments-differ
        self: TileModel,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction on a tensor batch or nested multi-task batch."""
        del batch_idx, dataloader_idx
        return self.forward(batch)

    def model_step(
        self: TileModel,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Compute and aggregate one loss per decoder task present in the batch."""
        del batch_idx, stage

        predictions = self.forward(batch)
        total_loss: Tensor | None = None
        metrics: dict[str, Tensor] = {}

        for task_name in self.decoders:
            target = self._extract_task_target(batch, task_name)
            if target is not None and task_name not in predictions:
                raise KeyError(
                    f"Task '{task_name}' has a target in the batch, but no nested "
                    f"image was found under '{task_name}'."
                )

        for task_name, prediction in predictions.items():
            target = self._extract_task_target(batch, task_name)
            if target is None:
                continue

            task_loss = self._compute_task_loss(
                task_name=task_name,
                prediction=prediction,
                target=target,
            )
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = task_loss * task_weight
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )
            metrics[f"{task_name}_loss"] = task_loss.detach()

        if total_loss is None:
            available_tasks = sorted(self.decoders.keys())
            raise KeyError(
                "TileModel.model_step() could not find any task targets in the "
                "batch. Expected either a flat {'target': ...} batch for a "
                f"single decoder or nested task entries for one of: {available_tasks}."
            )

        return total_loss, metrics

    @staticmethod
    def _resolve_component_config(
        spec: str | Path | dict[str, Any] | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        """Load a component config from an inline dict or YAML path."""
        if spec is None:
            raise ValueError(f"TileModel configuration requires '{label}'.")
        if isinstance(spec, (str, Path)):
            return load_yaml_config(spec)
        if isinstance(spec, dict):
            return dict(spec)
        raise TypeError(
            f"TileModel configuration field '{label}' must be a dict or path. "
            f"Got: {type(spec)!r}"
        )

    @staticmethod
    def _resolve_optimization_defaults(
        encoder: ModelABC,
        decoders: dict[str, ModelABC],
    ) -> tuple[
        Callable[..., Optimizer] | None,
        dict[str, Any] | None,
        Callable[..., LRScheduler] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """Reuse optimization settings from the first configured component."""
        for module in (encoder, *decoders.values()):
            optimizer_factory = getattr(module, "optimizer_factory", None)
            if optimizer_factory is None:
                continue

            return (
                optimizer_factory,
                dict(getattr(module, "optimizer_kwargs", {}) or {}),
                getattr(module, "lr_scheduler_factory", None),
                dict(getattr(module, "lr_scheduler_kwargs", {}) or {}),
                dict(getattr(module, "lr_scheduler_config", {}) or {}),
            )

        return None, None, None, None, None

    def _get_last_features_vit(
        self: TileModel, outputs: Tensor | Sequence[Tensor]
    ) -> Tensor:
        if isinstance(outputs, Tensor):
            tokens = outputs
        elif (
            isinstance(outputs, Sequence)
            and len(outputs) > 0
            and isinstance(outputs[-1], Tensor)
        ):
            tokens = outputs[-1]
        else:
            raise TypeError(
                "Expected outputs to be a torch.Tensor or a non-empty sequence of tensors.",
                f"Got: {type(outputs)!r}",
            )

        if tokens.ndim != 3:
            raise ValueError(
                f"Expected ViT token output with shape [B, N, C]. Got: {tuple(tokens.shape)}"
            )

        # tokens: [B, seq_len, hidden_dim]
        num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", None)
        assert (
            num_prefix_tokens is not None
        ), "ViTEncoder must have num_prefix_tokens attribute."
        if num_prefix_tokens >= 1:
            # index 0 is always the CLS token when num_prefix_tokens >= 1
            return tokens[:, 0, :]  # [B, hidden_dim]

        # exclude any prefix tokens (CLS, registers) from the mean pooling of patch tokens
        patch_tokens = tokens[:, num_prefix_tokens:, :]  # [B, num_patches, hidden_dim]
        return patch_tokens.mean(dim=1)  # [B, hidden_dim]

    def _get_last_features(
        self: TileModel, outputs: Tensor | Sequence[Tensor]
    ) -> Tensor:
        """Extract the encoder features for the decoders."""

        match self.encoder.__class__.__name__:
            case "ViTEncoder":
                return self._get_last_features_vit(outputs)
            case "ResNetEncoder" | "UNetEncoder":
                assert (
                    isinstance(outputs, Sequence) and len(outputs) == 5
                ), f"Expected outputs to be a sequence of 5 feature maps from the {self.encoder.__class__.__name__}."
                _, _, _, _, c4 = outputs
                return c4
            case _:
                raise ValueError(
                    f"TileModel does not have a default feature extraction strategy for encoder type '{self.encoder.__class__.__name__}'."
                    "Please provide a custom forward() method or ensure the encoder type matches one of the recognized defaults."
                )

    def _forward_shared_image(self: TileModel, image: Tensor) -> dict[str, Any]:
        """Encode one shared image tensor and fan it out to every decoder."""
        encoded = self.encoder(self._validate_image_tensor(image))
        outputs = {}
        for task_name, decoder in self.decoders.items():
            outputs[task_name] = self._decode_task_output(
                task_name=task_name,
                decoder=decoder,  # type: ignore
                encoded=encoded,
            )
        return outputs

    def _forward_multitask_batch(
        self: TileModel,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Encode one task image per decoder using nested task entries."""
        outputs: dict[str, Any] = {}

        for task_name, decoder in self.decoders.items():
            task_batch = batch.get(task_name)
            if not isinstance(task_batch, dict) or "image" not in task_batch:
                continue

            encoded = self.encoder(self._validate_image_tensor(task_batch["image"]))
            outputs[task_name] = self._decode_task_output(
                task_name=task_name,
                decoder=decoder,  # type: ignore
                encoded=encoded,
            )

        if outputs:
            return outputs

        available_keys = sorted(str(key) for key in batch.keys())
        expected_tasks = sorted(self.decoders.keys())
        raise KeyError(
            "Could not find any task images in the provided batch. Expected nested "
            f"task entries for one of {expected_tasks}. Available top-level keys: {available_keys}."
        )

    def _decode_task_output(
        self: TileModel,
        *,
        task_name: str,
        decoder: ModelABC,
        encoded: Tensor | Sequence[Tensor],
    ) -> Any:
        """Apply the default decoder input mapping for one task."""
        match task_name:
            case "tissue_segmentation" | "hematoxylin":
                return decoder(encoded)
            case "magnification" | "jigmag" | "tumor_classification":
                features = self._get_last_features(encoded)
                return decoder(features)
            case _:
                raise ValueError(
                    f"TileModel does not have a default decoding strategy for task '{task_name}'. "
                    "Please provide a custom forward() method or ensure the task name matches one of the recognized defaults."
                )

    def _extract_task_target(
        self: TileModel, batch: Any, task_name: str
    ) -> Tensor | None:
        """Read one task target from a flat or nested training batch."""
        if isinstance(batch, dict):
            task_batch = batch.get(task_name)
            if isinstance(task_batch, dict):
                target = task_batch.get("target")
                if target is None:
                    return None
                return self._validate_target_tensor(task_name, target)

            if "target" in batch and len(self.decoders) == 1:
                return self._validate_target_tensor(task_name, batch["target"])

        return None

    def _validate_image_tensor(self: TileModel, value: Any) -> Tensor:
        """Ensure the resolved image entry is a tensor."""
        if not isinstance(value, Tensor):
            raise TypeError(
                "TileModel expected the resolved 'image' entry to be a torch.Tensor."
            )
        return value

    def _validate_target_tensor(self: TileModel, task_name: str, value: Any) -> Tensor:
        """Ensure a task target is a tensor."""
        if not isinstance(value, Tensor):
            raise TypeError(
                f"TileModel expected task '{task_name}' target to be a torch.Tensor."
            )
        return value

    def _compute_task_loss(
        self: TileModel,
        *,
        task_name: str,
        prediction: Any,
        target: Tensor,
    ) -> Tensor:
        """Dispatch to a simple default loss based on task name and tensor shape."""
        if not isinstance(prediction, Tensor):
            raise TypeError(
                f"Decoder '{task_name}' must return a torch.Tensor for default "
                "TileModel loss computation."
            )

        match task_name:
            case "hematoxylin":
                return compute_regression_loss(prediction, target)
            case "tissue_segmentation":
                task_kwargs = self.task_kwargs.get(task_name, {})
                unknown_class_index = task_kwargs.get("unknown_class_index", None)
                return compute_semantic_segmentation_loss(
                    prediction, target, unknown_class_index=unknown_class_index
                )
            case "magnification" | "jigmag" | "tumor_classification":
                return compute_classification_loss(
                    prediction, target, unknown_class_index=None
                )
            case _:
                raise ValueError(
                    f"No default loss defined for task '{task_name}'. Please provide "
                    "a custom loss function or ensure the task name matches one of the "
                    "recognized defaults."
                )
