"""Slide-level Multiple Instance Learning (MIL) model for subtyping."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from VexDR.models.model_abc import ModelABC
from VexDR.models.slide_level.attention import Attention, GatedAttention
from VexDR.models.tile_level.tile_model import TileModel
from VexDR.models.utils import (
    get_lr_scheduler_from_config,
    get_optimizer_from_config,
)
from VexDR.utils.config import load_yaml_config
from VexDR.utils.metrics import compute_classification_loss

AggregationMethod = Literal["max", "mean", "attention"]
_SUPPORTED_AGGREGATION_METHODS: tuple[str, ...] = ("max", "mean", "attention")
SUPPORTED_MAIN_TASKS: tuple[str, ...] = ("subtyping",)


def _component_name(module: nn.Module | None) -> str | None:
    """Return a stable component name for logging and hyperparameters."""
    if module is None:
        return None
    return module.__class__.__name__


class _MeanPool(nn.Module):
    """Simple bag-wise mean pooling aggregator."""

    def forward(self, h: Tensor) -> tuple[Tensor, None]:
        """Aggregate a bag of tile features by mean pooling."""
        return h.mean(dim=1), None


class _MaxPool(nn.Module):
    """Simple bag-wise max pooling aggregator."""

    def forward(self, h: Tensor) -> tuple[Tensor, None]:
        """Aggregate a bag of tile features by max pooling."""
        return h.max(dim=1).values, None


class EmbeddingMIL(ModelABC):
    """Lightning model for slide-level embedding-based Multiple Instance Learning.

    Combines an optional shared tile encoder, a bag aggregator, and a fully
    connected decoder head into a single Lightning module solving one
    classification task — currently slide-level subtyping. Given a batch
    with image shape ``(B, K, 3, H, W)`` or pre-computed feature bags of
    shape ``(B, K, D)``, the forward pass:

    1. Encodes every tile independently via the shared encoder, if provided.
    2. Aggregates the ``K`` tile features into a single slide representation
       using the method specified by ``aggregation_method``.
    3. Produces ``num_classes`` logits via the fully connected decoder head.

    The main-task target is read from ``batch["target"]`` as a ``(B,)`` long
    tensor of class indices and the loss is cross-entropy. Any extra
    pretext-task entries in the batch are ignored.

    Parameters
    ----------
    aggregation_method
        How tile features are pooled into a slide representation. One of
        ``"max"``, ``"mean"``, or ``"attention"``. When ``"attention"`` is
        selected and ``gated=True``, :class:`GatedAttention` is used in place
        of plain :class:`Attention`.
    encoder
        Optional shared tile encoder. If ``None``, inputs are expected to be
        pre-computed feature bags of shape ``(B, K, D)``.
    main_task
        Name of the main task. Currently only ``"subtyping"`` is supported.
    hidden_dims
        Hidden layer sizes in the decoder MLP.
    output_dim
        Number of subtype classes the head produces.
    dropout
        Dropout rate applied between decoder MLP layers.
    attn_kwargs
        Optional keyword arguments for attention-based aggregators. Ignored
        when ``aggregation_method != "attention"``. Recognised keys:

        - ``input_dim`` (int or None): tile feature dim ``D``. ``None`` triggers
          lazy init on the first forward pass.
        - ``hidden_dim`` (int): attention hidden dim ``L``.
        - ``num_heads`` (int): number of independent attention distributions.
        - ``dropout`` (float): dropout after the attention non-linearity.
        - ``gated`` (bool): use gated attention (``tanh(V h) * sigmoid(U h)``).
    task_kwargs
        Optional mapping from task name to per-task keyword arguments used in
        loss computation. Recognized keys per task:

        - ``unknown_class_index`` (int or None): class index to ignore in the
          cross-entropy loss. Defaults to ``0`` for the subtyping main task
          (matches ``TCGASlideDataset.UNKNOWN_SUBTYPE_CLASS``). Set to
          ``None`` to count every class.
    freeze_encoder
        If ``True`` (default), set ``requires_grad=False`` on every encoder
        parameter and keep the encoder in ``eval()`` mode even when the
        parent is flipped to ``train()``. Has no effect when ``encoder`` is
        ``None``.
    optimizer_factory
        An optional factory function for constructing the optimizer.
    optimizer_kwargs
        Keyword arguments forwarded to ``optimizer_factory``.
    lr_scheduler_factory
        Optional callable that builds a learning-rate scheduler from the
        optimizer returned by ``optimizer_factory``.
    lr_scheduler_kwargs
        Keyword arguments forwarded to ``lr_scheduler_factory``.
    lr_scheduler_config
        Optional configuration for the learning-rate scheduler.
    """

    def __init__(
        self: EmbeddingMIL,
        aggregation_method: AggregationMethod = "attention",
        encoder: ModelABC | None = None,
        main_task: str = "subtyping",
        *,
        enc_dim: int | None,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        attn_kwargs: dict[str, Any] | None = None,
        task_kwargs: dict[str, Any] | None = None,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

        if aggregation_method not in _SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                f"aggregation_method must be one of {_SUPPORTED_AGGREGATION_METHODS}. "
                f"Got: {aggregation_method!r}"
            )
        if encoder is not None and not isinstance(encoder, ModelABC):
            raise TypeError("encoder must be an inherited ModelABC or None.")

        if not isinstance(main_task, str) or not main_task:
            raise ValueError("main_task must be a non-empty string.")
        if main_task not in SUPPORTED_MAIN_TASKS:
            raise ValueError(
                f"Unsupported main_task: {main_task!r}. "
                f"Must be one of {SUPPORTED_MAIN_TASKS}."
            )
        if task_kwargs is not None and not isinstance(task_kwargs, dict):
            raise TypeError(
                "task_kwargs must be a dict of task name to per-task options."
            )
        task_kwargs = dict(task_kwargs or {})
        main_task_kwargs = task_kwargs.get(main_task, {})
        if not isinstance(main_task_kwargs, dict):
            raise TypeError(f"task_kwargs['{main_task}'] must be a dict if provided.")
        unknown_class_index = main_task_kwargs.get("unknown_class_index", 0)
        if unknown_class_index is not None and (
            not isinstance(unknown_class_index, int) or unknown_class_index < 0
        ):
            raise ValueError(
                "task_kwargs['{main_task}']['unknown_class_index'] must be a "
                f"non-negative integer or None. Got: {unknown_class_index}"
            )

        attn_kwargs = attn_kwargs or {}
        attn_input_dim = attn_kwargs.get("input_dim")
        attn_hidden_dim = attn_kwargs.get("hidden_dim", 128)
        attn_num_heads = attn_kwargs.get("num_heads", 1)
        attn_dropout = attn_kwargs.get("dropout", 0.0)
        attn_gated = attn_kwargs.get("gated", False)

        if aggregation_method == "attention":
            if attn_input_dim is not None and (
                not isinstance(attn_input_dim, int) or attn_input_dim <= 0
            ):
                raise ValueError(
                    f"attn_kwargs['input_dim'] must be a positive integer or None. "
                    f"Got: {attn_input_dim}"
                )
            if not isinstance(attn_hidden_dim, int) or attn_hidden_dim <= 0:
                raise ValueError(
                    f"attn_kwargs['hidden_dim'] must be a positive integer. "
                    f"Got: {attn_hidden_dim}"
                )
            if not isinstance(attn_num_heads, int) or attn_num_heads <= 0:
                raise ValueError(
                    f"attn_kwargs['num_heads'] must be a positive integer. "
                    f"Got: {attn_num_heads}"
                )
            if (
                not isinstance(attn_dropout, (int, float))
                or not 0.0 <= attn_dropout < 1.0
            ):
                raise ValueError(
                    f"attn_kwargs['dropout'] must be a float in the range [0.0, 1.0). "
                    f"Got: {attn_dropout}"
                )
            if not isinstance(attn_gated, bool):
                raise ValueError(
                    f"attn_kwargs['gated'] must be a boolean. Got: {attn_gated}"
                )

        self.aggregation_method: AggregationMethod = aggregation_method
        self.attn_gated = (
            bool(attn_gated) if aggregation_method == "attention" else False
        )
        self.has_encoder = encoder is not None
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.freeze_encoder = bool(freeze_encoder) and self.has_encoder
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        if not isinstance(encoder_chunk_size, int):
            raise TypeError("encoder_chunk_size must be an integer.")
        self.encoder_chunk_size = encoder_chunk_size
        self.aggregator = self._build_aggregator(
            method=aggregation_method,
            gated=self.attn_gated,
            input_dim=attn_input_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=attn_num_heads,
            dropout=attn_dropout,
        )

        self.main_task = main_task
        self.task_kwargs: dict[str, Any] = task_kwargs
        self.unknown_class_index = unknown_class_index
        self.decoder = self._build_decoder(
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.save_hyperparameters(
            {
                "encoder": _component_name(encoder),
                "aggregation_method": self.aggregation_method,
                "gated": self.attn_gated,
                "attn_kwargs": attn_kwargs,
                "aggregator": _component_name(self.aggregator),
                "main_task": self.main_task,
                "hidden_dims": hidden_dims,
                "output_dim": output_dim,
                "dropout": dropout,
                "task_kwargs": self.task_kwargs,
                "freeze_encoder": self.freeze_encoder,
                "encoder_chunk_size": self.encoder_chunk_size,
            }
        )

    def train(self: EmbeddingMIL, mode: bool = True) -> EmbeddingMIL:
        """Keep a frozen encoder in eval mode even when the parent is training."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @staticmethod
    def from_config(config: dict[str, Any]) -> EmbeddingMIL:
        """Instantiate an EmbeddingMIL from a configuration dictionary."""
        if not isinstance(config, dict):
            raise TypeError("EmbeddingMIL config must be provided as a dict.")

        tile_model_spec = config.get("tile_model_config", None)

        if tile_model_spec is not None:
            tile_model_cfg = EmbeddingMIL._resolve_component_config(
                tile_model_spec,
                label="tile_model_config",
            )
            tile_model_params = tile_model_cfg.get("params", tile_model_cfg)
            tile_model = TileModel.from_config(tile_model_params)
            if not isinstance(tile_model, TileModel):
                raise TypeError(
                    "The 'tile_model' component of the config must be a TileModel instance."
                )

            checkpoint_path = tile_model_cfg.get("checkpoint_path", None)
            assert isinstance(
                checkpoint_path, str
            ), "tile_model_config checkpoint_path must be a string."
            assert os.path.isfile(
                checkpoint_path
            ), f"tile_model_config checkpoint_path does not exist: {checkpoint_path}"

            # Supports both raw state_dicts and Lightning-style checkpoints.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            tile_model.load_state_dict(state_dict, strict=False)

            encoder = tile_model.encoder
        else:
            encoder = None

        aggregation_method = config.get("aggregation_method", "attention")
        assert aggregation_method in _SUPPORTED_AGGREGATION_METHODS, (
            f"aggregation_method must be one of {_SUPPORTED_AGGREGATION_METHODS}. "
            f"Got: {aggregation_method!r}"
        )

        main_task = config.get("main_task", "subtyping")
        assert (
            isinstance(main_task, str) and main_task
        ), "main_task must be a non-empty string"
        assert main_task in SUPPORTED_MAIN_TASKS, (
            f"Unsupported main_task: {main_task!r}. "
            f"Must be one of {SUPPORTED_MAIN_TASKS}."
        )

        enc_dim = config.get("enc_dim", None)
        if enc_dim is not None:
            assert (
                isinstance(enc_dim, int) and enc_dim > 0
            ), "enc_dim must be a positive integer when provided"

        hidden_dims = config.get("hidden_dims", None)
        assert hidden_dims is not None, "hidden_dims must be specified in the config"
        assert isinstance(hidden_dims, list) and all(
            isinstance(hd, int) and hd > 0 for hd in hidden_dims
        ), "hidden_dims must be a list of positive integers"

        output_dim = config.get("output_dim", None)
        assert output_dim is not None, "output_dim must be specified in the config"
        assert (
            isinstance(output_dim, int) and output_dim > 0
        ), "output_dim must be a positive integer"

        dropout = config.get("dropout", 0.0)
        assert (
            isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0
        ), "dropout must be a float in the range [0.0, 1.0)"

        attn_kwargs = config.get("attn_kwargs", None)
        if attn_kwargs is not None and not isinstance(attn_kwargs, dict):
            raise TypeError("attn_kwargs must be a dict if provided.")

        task_kwargs = config.get("task_kwargs", None)
        if task_kwargs is not None and not isinstance(task_kwargs, dict):
            raise TypeError("task_kwargs must be a dict if provided.")

        freeze_encoder = config.get("freeze_encoder", True)
        if not isinstance(freeze_encoder, bool):
            raise TypeError("freeze_encoder must be a boolean if provided.")

        encoder_chunk_size = config.get("encoder_chunk_size", 64)
        if not isinstance(encoder_chunk_size, int):
            raise TypeError("encoder_chunk_size must be an int if provided.")

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return EmbeddingMIL(
            aggregation_method=aggregation_method,
            encoder=encoder,
            main_task=main_task,
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            attn_kwargs=attn_kwargs,
            task_kwargs=task_kwargs,
            freeze_encoder=freeze_encoder,
            encoder_chunk_size=encoder_chunk_size,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    @staticmethod
    def _resolve_component_config(
        spec: str | Path | dict[str, Any] | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        """Load a component config from an inline dict or YAML path."""
        if spec is None:
            raise ValueError(f"EmbeddingMIL configuration requires '{label}'.")
        if isinstance(spec, (str, Path)):
            return load_yaml_config(spec)
        if isinstance(spec, dict):
            return dict(spec)
        raise TypeError(
            f"EmbeddingMIL configuration field '{label}' must be a dict or path. "
            f"Got: {type(spec)!r}"
        )

    def forward(  # pylint: disable=arguments-differ
        self: EmbeddingMIL,
        batch_or_image: Tensor | dict[str, Any],
    ) -> dict[str, Any]:
        """Encode tiles, aggregate the bag, and run the decoder."""
        if isinstance(batch_or_image, dict):
            if "image" not in batch_or_image:
                raise KeyError(
                    "EmbeddingMIL expects a dict batch with an 'image' entry or a tensor."
                )
            image = batch_or_image["image"]
        else:
            image = batch_or_image
        if not isinstance(image, Tensor):
            raise TypeError("EmbeddingMIL expected 'image' to be a torch.Tensor.")

        features = self._encode_bag(image)
        aggregated, attention_weights = self._aggregate(features)
        prediction = self.decoder(aggregated)

        return {
            self.main_task: prediction,
            "_aggregated": aggregated,
            "_attention_weights": attention_weights,
        }

    def predict_step(  # pylint: disable=arguments-differ
        self: EmbeddingMIL,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction on a slide-level batch."""
        del batch_idx, dataloader_idx
        return self.forward(batch)

    def model_step(
        self: EmbeddingMIL,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Compute the cross-entropy subtyping loss for one slide-level batch."""
        del batch_idx, stage

        if not isinstance(batch, dict):
            raise TypeError("EmbeddingMIL.model_step() expects a dict batch.")
        target = batch.get("target")
        if target is None:
            raise KeyError(
                "EmbeddingMIL.model_step() could not find a main-task target in "
                "the batch. Expected batch['target']."
            )
        if not isinstance(target, Tensor):
            raise TypeError(
                f"EmbeddingMIL expected main-task target to be a torch.Tensor. "
                f"Got: {type(target)!r}"
            )

        predictions = self.forward(batch)
        prediction = predictions[self.main_task]
        loss = compute_classification_loss(
            prediction,
            target,
            unknown_class_index=self.unknown_class_index,
        )
        return loss, {f"{self.main_task}_loss": loss.detach()}

    @staticmethod
    def _build_aggregator(
        *,
        method: AggregationMethod,
        gated: bool,
        input_dim: int | None,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> nn.Module:
        """Instantiate the aggregator module for the requested method."""
        match method:
            case "max":
                return _MaxPool()
            case "mean":
                return _MeanPool()
            case "attention":
                aggregator_cls = GatedAttention if gated else Attention
                return aggregator_cls(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            case _:
                raise ValueError(f"Unsupported aggregation_method: {method!r}")

    @staticmethod
    def _build_decoder(
        *,
        enc_dim: int | None,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build the MLP decoder head, lazily inferring the input width if needed."""
        if enc_dim is not None and (not isinstance(enc_dim, int) or enc_dim <= 0):
            raise ValueError(
                f"enc_dim must be a positive integer or None. Got: {enc_dim}"
            )
        if not isinstance(hidden_dims, list) or any(
            not isinstance(hidden_dim, int) or hidden_dim <= 0
            for hidden_dim in hidden_dims
        ):
            raise ValueError("hidden_dims must be a list of positive integers.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(
                f"output_dim must be a positive integer. Got: {output_dim}"
            )
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
            )

        first_out_dim = hidden_dims[0] if hidden_dims else output_dim
        layers: list[nn.Module] = [
            (
                nn.LazyLinear(first_out_dim)
                if enc_dim is None
                else nn.Linear(enc_dim, first_out_dim)
            )
        ]
        if not hidden_dims:
            return nn.Sequential(*layers)

        prev_dim = first_out_dim
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _encode_bag(self: EmbeddingMIL, image: Tensor) -> Tensor:
        """Encode a tile bag to per-instance features of shape ``(B, K, D)``."""
        if image.ndim == 3:
            return image
        if image.ndim != 5:
            raise ValueError(
                "Expected image shape (B, K, 3, H, W) or (B, K, D). "
                f"Got: {tuple(image.shape)}"
            )
        if not self.has_encoder:
            raise ValueError(
                "EmbeddingMIL received raw tiles but no encoder was configured. "
                "Either supply an encoder or pre-encode inputs to (B, K, D) features."
            )

        batch_size, num_tiles = image.shape[:2]
        flat = image.flatten(0, 1)
        chunk_size = self.encoder_chunk_size
        # Chunk the encoder forward so layer-1 activations don't blow up GPU
        # memory on whole-slide bags. Frozen encoders can run under no_grad.
        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with ctx:
            if chunk_size <= 0 or flat.shape[0] <= chunk_size:
                features = self._get_last_features(self.encoder(flat))
            else:
                feature_chunks = [
                    self._get_last_features(self.encoder(flat[start : start + chunk_size]))
                    for start in range(0, flat.shape[0], chunk_size)
                ]
                features = torch.cat(feature_chunks, dim=0)
        return features.view(batch_size, num_tiles, -1)

    def _aggregate(
        self: EmbeddingMIL, features: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Run the aggregator and normalize the return to ``(aggregated, weights)``."""
        output = self.aggregator(features)
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], Tensor):
                raise TypeError(
                    "Aggregator tuple outputs must start with an aggregated tensor."
                )
            aggregated = output[0]
            attention_weights = output[1] if len(output) >= 2 else None
            return aggregated, attention_weights
        if not isinstance(output, Tensor):
            raise TypeError(
                "Aggregator must return a tensor or a (tensor, weights) tuple."
            )
        return output, None

    def _get_last_features(
        self: EmbeddingMIL, outputs: Tensor | Sequence[Tensor]
    ) -> Tensor:
        """Reduce encoder outputs to a flat per-instance feature tensor ``(N, D)``."""
        if not self.has_encoder and isinstance(outputs, Tensor) and outputs.ndim == 2:
            return outputs

        encoder_name = self.encoder.__class__.__name__
        match encoder_name:
            case "ViTEncoder":
                return self._get_last_features_vit(outputs)
            case "ResNetEncoder" | "UNetEncoder":
                assert isinstance(outputs, Sequence) and len(outputs) == 5, (
                    f"Expected outputs to be a sequence of 5 feature maps from the "
                    f"{encoder_name}."
                )
                _, _, _, _, c4 = outputs
                return c4.flatten(start_dim=2).mean(dim=-1)
            case _:
                if isinstance(outputs, Tensor) and outputs.ndim == 2:
                    return outputs
                raise ValueError(
                    f"EmbeddingMIL does not have a default feature extraction "
                    f"strategy for encoder type '{encoder_name}'. Ensure the "
                    "encoder returns a 2-D tensor of shape (N, D)."
                )

    def _get_last_features_vit(
        self: EmbeddingMIL, outputs: Tensor | Sequence[Tensor]
    ) -> Tensor:
        """Extract a pooled feature vector from ViT encoder outputs."""
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
                "Expected ViT outputs to be a Tensor or non-empty Tensor sequence. "
                f"Got: {type(outputs)!r}"
            )

        if tokens.ndim != 3:
            raise ValueError(
                f"Expected ViT token output with shape (N, seq_len, C). "
                f"Got: {tuple(tokens.shape)}"
            )

        num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", None)
        assert (
            num_prefix_tokens is not None
        ), "ViTEncoder must expose a num_prefix_tokens attribute."
        if num_prefix_tokens >= 1:
            return tokens[:, 0, :]
        patch_tokens = tokens[:, num_prefix_tokens:, :]
        return patch_tokens.mean(dim=1)
