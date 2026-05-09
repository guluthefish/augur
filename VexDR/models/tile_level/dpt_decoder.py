"""Lightning-compatible DPT-style decoder for tile-level dense prediction."""

from __future__ import annotations

from math import isqrt
from typing import Any, Callable, Sequence

from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from VexDR.models.model_abc import ModelABC
from VexDR.models.utils import get_lr_scheduler_from_config, get_optimizer_from_config


def _normalize_pair(
    value: int | Sequence[int] | None,
    *,
    name: str,
    allow_none: bool = False,
) -> tuple[int, int] | None:
    """Normalize scalar-or-pair config values to an ``(h, w)`` tuple."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be provided.")

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive. Got: {value}")
        return (value, value)

    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be an int or a 2-item sequence. Got: {value}")

    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive. Got: {value}")
    return (height, width)


def _align_corners_or_none(upsample_mode: str, align_corners: bool) -> bool | None:
    """Return a valid ``align_corners`` argument for ``F.interpolate``."""
    return align_corners if upsample_mode != "nearest" else None


class _ConvBNReLU(nn.Module):
    """A small convolutional refinement block used throughout the decoder."""

    def __init__(
        self: _ConvBNReLU,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self: _ConvBNReLU, x: Tensor) -> Tensor:
        """Apply the refinement block."""
        return self.block(x)


class _ResidualConvUnit(nn.Module):
    """A lightweight residual refinement block inspired by DPT fusion units."""

    def __init__(
        self: _ResidualConvUnit, channels: int, *, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.block1 = _ConvBNReLU(channels, channels, dropout=dropout)
        self.block2 = _ConvBNReLU(channels, channels, dropout=dropout)

    def forward(self: _ResidualConvUnit, x: Tensor) -> Tensor:
        """Refine a feature map while preserving its residual signal."""
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        return x + residual


class _FeatureFusionBlock(nn.Module):
    """Fuse a coarse decoder state with a finer skip feature map."""

    def __init__(
        self: _FeatureFusionBlock,
        channels: int,
        *,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.skip_refine = _ResidualConvUnit(channels, dropout=dropout)
        self.out_refine = _ResidualConvUnit(channels, dropout=dropout)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self: _FeatureFusionBlock,
        x: Tensor,
        skip: Tensor | None = None,
    ) -> Tensor:
        """Upsample the decoder state and merge it with an optional skip tensor."""
        if skip is not None:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode=self.upsample_mode,
                align_corners=_align_corners_or_none(
                    self.upsample_mode, self.align_corners
                ),
            )
            x = x + self.skip_refine(skip)

        x = self.out_refine(x)
        return self.out_proj(x)


class _ReassembleBlock(nn.Module):
    """Project ViT token grids to a decoder feature map at a target scale."""

    def __init__(
        self: _ReassembleBlock,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine = _ConvBNReLU(out_channels, out_channels, dropout=dropout)

    def forward(
        self: _ReassembleBlock,
        x: Tensor,
        target_size: tuple[int, int],
    ) -> Tensor:
        """Project and resize one spatial feature map to the requested scale."""
        x = self.proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode=self.upsample_mode,
                align_corners=_align_corners_or_none(
                    self.upsample_mode, self.align_corners
                ),
            )
        return self.refine(x)


class DPTDecoder(ModelABC):
    """DPT-style decoder for dense prediction from ViT token features.

    The decoder accepts:

    - a single ViT token tensor of shape ``(B, N, C)``
    - a sequence of token tensors, typically intermediate ViT layer outputs
    - a dict containing ``"tokens"`` or ``"features"`` plus optional metadata

    When only one token tensor is provided, it is reused across the four DPT
    reassembly branches so the decoder remains compatible with encoder wrappers
    that currently expose only the final token sequence.

    Parameters
    ----------
    output_channels
        The number of channels in the output logits. This should match the number
        of classes for classification tasks, or 1 for regression tasks.
    embed_dim
        The embedding dimension of the input token features.
        This should match the output dimension of the ViT encoder.
    feature_channels
        The number of channels in the intermediate feature maps produced by the decoder.
    head_channels
        The number of channels in the final decoder head before projecting to the output channels.
    patch_size
        The spatial size of the image patches corresponding to the input tokens.
        This is used to determine the output resolution when it cannot be inferred from the input metadata.
    grid_size
        The spatial size of the token grid corresponding to the input tokens.
        This is used to determine the spatial arrangement of the input tokens when it cannot be inferred from the token count.
    num_prefix_tokens
        The number of prefix tokens in the input sequence, such as class tokens or distillation tokens.
        These tokens are ignored when reassembling the spatial feature maps.
    dropout
        The dropout probability used in the convolutional blocks throughout the decoder.
    upsample_mode
        The interpolation mode used when resizing feature maps. Passed to ``F.interpolate``.
    align_corners
        The ``align_corners`` argument used when resizing feature maps. Passed to ``F.interpolate``.
        Ignored when ``upsample_mode`` is "nearest".
    optimizer_factory
        An optional factory function for constructing the optimizer.
    optimizer_kwargs
        Keyword arguments forwarded to ``optimizer_factory``.
    lr_scheduler_factory
        Optional callable that builds a learning-rate scheduler from the optimizer
        returned by ``optimizer_factory``.
    lr_scheduler_kwargs
        Keyword arguments forwarded to ``lr_scheduler_factory``.
    lr_scheduler_config
        Optional configuration for the learning-rate scheduler.
    """

    def __init__(
        self: DPTDecoder,
        output_channels: int,
        embed_dim: int,
        *,
        feature_channels: int = 256,
        head_channels: int = 128,
        patch_size: int | Sequence[int] = 16,
        grid_size: int | Sequence[int] | None = None,
        num_prefix_tokens: int = 1,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
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

        if not isinstance(output_channels, int) or output_channels <= 0:
            raise ValueError(
                f"output_channels must be a positive integer. Got: {output_channels}"
            )
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer. Got: {embed_dim}")
        if not isinstance(feature_channels, int) or feature_channels <= 0:
            raise ValueError(
                f"feature_channels must be a positive integer. Got: {feature_channels}"
            )
        if not isinstance(head_channels, int) or head_channels <= 0:
            raise ValueError(
                f"head_channels must be a positive integer. Got: {head_channels}"
            )
        if not isinstance(num_prefix_tokens, int) or num_prefix_tokens < 0:
            raise ValueError(
                "num_prefix_tokens must be a non-negative integer. "
                f"Got: {num_prefix_tokens}"
            )
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
            )
        if not isinstance(upsample_mode, str):
            raise ValueError(f"upsample_mode must be a string. Got: {upsample_mode}")
        if not isinstance(align_corners, bool):
            raise ValueError(f"align_corners must be a boolean. Got: {align_corners}")

        self.output_channels = output_channels
        self.embed_dim = embed_dim
        self.feature_channels = feature_channels
        self.head_channels = head_channels
        self.patch_size = _normalize_pair(patch_size, name="patch_size")
        self.grid_size = _normalize_pair(grid_size, name="grid_size", allow_none=True)
        self.num_prefix_tokens = num_prefix_tokens
        self.dropout = float(dropout)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        self.reassemble_blocks = nn.ModuleList(
            [
                _ReassembleBlock(
                    embed_dim,
                    feature_channels,
                    dropout=self.dropout,
                    upsample_mode=upsample_mode,
                    align_corners=align_corners,
                )
                for _ in range(4)
            ]
        )
        self.lowest_refine = _ResidualConvUnit(
            feature_channels,
            dropout=self.dropout,
        )
        self.fuse_mid = _FeatureFusionBlock(
            feature_channels,
            dropout=self.dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.fuse_high = _FeatureFusionBlock(
            feature_channels,
            dropout=self.dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.fuse_top = _FeatureFusionBlock(
            feature_channels,
            dropout=self.dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.head = nn.Sequential(
            _ConvBNReLU(feature_channels, head_channels, dropout=self.dropout),
            nn.Conv2d(head_channels, output_channels, kernel_size=1),
        )

        self.save_hyperparameters(
            {
                "output_channels": output_channels,
                "embed_dim": embed_dim,
                "feature_channels": feature_channels,
                "head_channels": head_channels,
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "num_prefix_tokens": num_prefix_tokens,
                "dropout": self.dropout,
                "upsample_mode": upsample_mode,
                "align_corners": align_corners,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> DPTDecoder:
        """Construct a ``DPTDecoder`` from a configuration dictionary."""
        output_channels = config.get("output_channels", None)
        embed_dim = config.get("embed_dim", None)
        if output_channels is None:
            raise ValueError(
                "DPTDecoder requires an 'output_channels' field in the config."
            )
        if embed_dim is None:
            raise ValueError("DPTDecoder requires an 'embed_dim' field in the config.")

        feature_channels = config.get("feature_channels", 256)
        head_channels = config.get("head_channels", 128)
        patch_size = config.get("patch_size", 16)
        grid_size = config.get("grid_size", None)
        num_prefix_tokens = config.get("num_prefix_tokens", 1)
        dropout = config.get("dropout", 0.0)
        upsample_mode = config.get("upsample_mode", "bilinear")
        align_corners = config.get("align_corners", False)

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return DPTDecoder(
            output_channels=int(output_channels),
            embed_dim=int(embed_dim),
            feature_channels=int(feature_channels),
            head_channels=int(head_channels),
            patch_size=patch_size,
            grid_size=grid_size,
            num_prefix_tokens=int(num_prefix_tokens),
            dropout=float(dropout),
            upsample_mode=upsample_mode,
            align_corners=align_corners,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: DPTDecoder,
        encoded: Tensor | Sequence[Tensor] | dict[str, Any],
    ) -> Tensor:
        """Decode ViT token features into full-resolution dense logits."""
        features, grid_size, output_size = self._extract_features(encoded)
        reassembled_sizes = self._get_reassembled_sizes(grid_size)

        reassembled = [
            block(feature, target_size)
            for block, feature, target_size in zip(
                self.reassemble_blocks, features, reassembled_sizes
            )
        ]

        path = self.lowest_refine(reassembled[3])
        path = self.fuse_mid(path, reassembled[2])
        path = self.fuse_high(path, reassembled[1])
        path = self.fuse_top(path, reassembled[0])

        logits = self.head(path)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode=self.upsample_mode,
                align_corners=_align_corners_or_none(
                    self.upsample_mode, self.align_corners
                ),
            )
        return logits

    def predict_step(  # pylint: disable=arguments-differ
        self: DPTDecoder,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """Run decoder inference on token batches or token metadata dicts."""
        del batch_idx, dataloader_idx
        return self.forward(batch)

    def model_step(
        self: DPTDecoder,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "DPTDecoder is a decoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def _extract_features(
        self: DPTDecoder,
        encoded: Tensor | Sequence[Tensor] | dict[str, Any],
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """Normalize supported decoder inputs into four spatial feature maps."""
        num_prefix_tokens = self.num_prefix_tokens
        grid_size = self.grid_size
        output_size: tuple[int, int] | None = None

        if isinstance(encoded, dict):
            if "features" in encoded:
                raw_features = encoded["features"]
            elif "tokens" in encoded:
                raw_features = encoded["tokens"]
            else:
                raise KeyError(
                    "DPTDecoder dict inputs must contain 'tokens' or 'features'."
                )

            if "num_prefix_tokens" in encoded:
                num_prefix_tokens = int(encoded["num_prefix_tokens"])
            if "grid_size" in encoded:
                grid_size = _normalize_pair(
                    encoded["grid_size"],
                    name="grid_size",
                    allow_none=True,
                )
            if "output_size" in encoded:
                output_size = _normalize_pair(
                    encoded["output_size"],
                    name="output_size",
                )
        else:
            raw_features = encoded

        feature_sequence = self._normalize_feature_sequence(raw_features)
        reference_grid = self._infer_grid_size(
            feature_sequence[0],
            grid_size=grid_size,
            num_prefix_tokens=num_prefix_tokens,
        )

        spatial_features = [
            self._to_spatial_feature(
                feature,
                grid_size=reference_grid,
                num_prefix_tokens=num_prefix_tokens,
            )
            for feature in feature_sequence
        ]

        if output_size is None:
            if self.patch_size is None:
                raise ValueError(
                    "DPTDecoder cannot determine output_size without patch_size or "
                    "explicit output_size in the input dict."
                )
            output_size = (
                reference_grid[0] * self.patch_size[0],
                reference_grid[1] * self.patch_size[1],
            )

        return spatial_features, reference_grid, output_size

    def _normalize_feature_sequence(
        self: DPTDecoder,
        features: Tensor | Sequence[Tensor],
    ) -> list[Tensor]:
        """Normalize decoder inputs to exactly four feature tensors."""
        if isinstance(features, Tensor):
            normalized = [features]
        elif isinstance(features, Sequence):
            normalized = list(features)
        else:
            raise TypeError(
                "DPTDecoder expects a torch.Tensor, a sequence of tensors, or a "
                "dict containing 'tokens' or 'features'."
            )

        if not normalized:
            raise ValueError("DPTDecoder received an empty feature sequence.")
        if not all(isinstance(feature, Tensor) for feature in normalized):
            raise TypeError("All DPTDecoder features must be torch.Tensor objects.")

        if len(normalized) > 4:
            normalized = normalized[-4:]
        while len(normalized) < 4:
            normalized.append(normalized[-1])

        return normalized

    def _infer_grid_size(
        self: DPTDecoder,
        feature: Tensor,
        *,
        grid_size: tuple[int, int] | None,
        num_prefix_tokens: int,
    ) -> tuple[int, int]:
        """Infer the base token grid from the first feature tensor."""
        if feature.ndim == 4:
            return (int(feature.shape[-2]), int(feature.shape[-1]))

        if feature.ndim != 3:
            raise TypeError(
                "DPTDecoder token features must have shape (B, N, C) or "
                f"(B, C, H, W). Got tensor shape: {tuple(feature.shape)}"
            )

        if grid_size is not None:
            token_count = grid_size[0] * grid_size[1]
            if feature.shape[1] not in (token_count, token_count + num_prefix_tokens):
                raise ValueError(
                    "Provided grid_size is incompatible with the token count. "
                    f"Expected {token_count} or {token_count + num_prefix_tokens} "
                    f"tokens, got {feature.shape[1]}."
                )
            return grid_size

        for prefix_tokens in (num_prefix_tokens, 0):
            patch_tokens = feature.shape[1] - prefix_tokens
            if patch_tokens <= 0:
                continue
            side = isqrt(patch_tokens)
            if side * side == patch_tokens:
                return (side, side)

        raise ValueError(
            "DPTDecoder could not infer a square patch grid from the token "
            f"count {feature.shape[1]}. Provide grid_size explicitly."
        )

    def _to_spatial_feature(
        self: DPTDecoder,
        feature: Tensor,
        *,
        grid_size: tuple[int, int],
        num_prefix_tokens: int,
    ) -> Tensor:
        """Convert a token sequence or spatial feature map to ``(B, C, H, W)``."""
        if feature.ndim == 4:
            return feature

        if feature.ndim != 3:
            raise TypeError(
                "DPTDecoder token features must have shape (B, N, C) or "
                f"(B, C, H, W). Got tensor shape: {tuple(feature.shape)}"
            )

        height, width = grid_size
        token_count = height * width
        sequence_length = int(feature.shape[1])
        if sequence_length == token_count:
            tokens = feature
        elif sequence_length == token_count + num_prefix_tokens:
            tokens = feature[:, num_prefix_tokens:, :]
        else:
            raise ValueError(
                "Token count does not match the inferred grid size. "
                f"Expected {token_count} or {token_count + num_prefix_tokens} "
                f"tokens, got {sequence_length}."
            )

        batch_size = int(tokens.shape[0])
        channels = int(tokens.shape[2])
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

    def _get_reassembled_sizes(
        self: DPTDecoder,
        grid_size: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return the target spatial sizes for the four DPT branches."""
        base_h, base_w = grid_size
        return [
            (base_h * 4, base_w * 4),
            (base_h * 2, base_w * 2),
            (base_h, base_w),
            (max(base_h // 2, 1), max(base_w // 2, 1)),
        ]
