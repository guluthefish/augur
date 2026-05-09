"""Lightning-compatible U-Net decoder for tile-level dense prediction."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from VexDR.models.model_abc import ModelABC
from VexDR.models.utils import get_lr_scheduler_from_config, get_optimizer_from_config


class _DoubleConv(nn.Module):
    """Two Conv-BN-ReLU blocks used throughout the decoder."""

    def __init__(
        self: _DoubleConv,
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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self: _DoubleConv, x: Tensor) -> Tensor:
        """Apply the double-convolution block."""
        return self.block(x)


class _DecoderStage(nn.Module):
    """Upsample the decoder state, merge a skip connection, and refine it."""

    def __init__(
        self: _DecoderStage,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.block = _DoubleConv(
            in_channels + skip_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self: _DecoderStage, x: Tensor, skip: Tensor) -> Tensor:
        """Fuse an upsampled decoder state with a same-scale skip tensor."""
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode=self.upsample_mode,
            align_corners=(
                self.align_corners if self.upsample_mode != "nearest" else None
            ),
        )
        return self.block(torch.cat((x, skip), dim=1))


class UNetDecoder(ModelABC):
    """
    U-Net decoder exposed as a Lightning module.

    Expects multi-scale feature maps from a ResNet-like encoder and decodes them
    into full-resolution logits for dense prediction tasks.

    Parameters
    ----------
    output_channels
        The number of output channels for the final prediction head.
    encoder_channels
        The number of channels in each encoder feature map, ordered as (c0, c1, c2, c3, c4).
    decoder_channels
        The number of channels in each decoder stage, ordered as (d3, d2, d1, d0).
    dropout
        The dropout probability to apply after each convolutional block.
    upsample_mode
        The interpolation mode to use for upsampling (e.g., "bilinear" or "nearest").
    align_corners
        Whether to align corners when upsampling with a mode that supports it.
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
        self: UNetDecoder,
        output_channels: int,
        encoder_channels: Sequence[int] = (64, 64, 128, 256, 512),
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
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

        if len(encoder_channels) != 5:
            raise ValueError(
                "encoder_channels must contain five channel counts: c0 through c4."
            )
        if len(decoder_channels) != 4:
            raise ValueError(
                "decoder_channels must contain four channel counts for the decoder."
            )

        self.output_channels = output_channels
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.decoder_channels = tuple(int(channel) for channel in decoder_channels)
        self.dropout = dropout
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        c0_channels, c1_channels, c2_channels, c3_channels, c4_channels = (
            self.encoder_channels
        )
        d3_channels, d2_channels, d1_channels, d0_channels = self.decoder_channels

        self.decode3 = _DecoderStage(
            in_channels=c4_channels,
            skip_channels=c3_channels,
            out_channels=d3_channels,
            dropout=dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.decode2 = _DecoderStage(
            in_channels=d3_channels,
            skip_channels=c2_channels,
            out_channels=d2_channels,
            dropout=dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.decode1 = _DecoderStage(
            in_channels=d2_channels,
            skip_channels=c1_channels,
            out_channels=d1_channels,
            dropout=dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        self.fuse0 = _DoubleConv(
            in_channels=d1_channels + c0_channels,
            out_channels=d0_channels,
            dropout=dropout,
        )
        if output_channels is not None:
            self.head = nn.Conv2d(d0_channels, output_channels, kernel_size=1)
        else:
            self.head = nn.Identity()

        self.save_hyperparameters(
            {
                "output_channels": output_channels,
                "encoder_channels": self.encoder_channels,
                "decoder_channels": self.decoder_channels,
                "dropout": dropout,
                "upsample_mode": upsample_mode,
                "align_corners": align_corners,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> UNetDecoder:
        """Construct a UNetDecoder from a configuration dictionary."""

        output_channels = config.get("output_channels", None)
        if output_channels is None:
            raise ValueError(
                "UNetDecoder requires an 'output_channels' field in the config."
            )
        if not isinstance(output_channels, int) or output_channels <= 0:
            raise ValueError(
                f"output_channels must be a positive integer. Got: {output_channels}"
            )
        encoder_channels = config.get("encoder_channels", (64, 64, 128, 256, 512))
        if not isinstance(encoder_channels, Sequence) or len(encoder_channels) != 5:
            raise ValueError(
                f"encoder_channels must be a sequence of five channel counts. Got: {encoder_channels}"
            )
        decoder_channels = config.get("decoder_channels", (256, 128, 64, 64))
        if not isinstance(decoder_channels, Sequence) or len(decoder_channels) != 4:
            raise ValueError(
                f"decoder_channels must be a sequence of four channel counts. Got: {decoder_channels}"
            )
        dropout = config.get("dropout", 0.0)
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
            )
        upsample_mode = config.get("upsample_mode", "bilinear")
        if not isinstance(upsample_mode, str):
            raise ValueError(
                f"upsample_mode must be a string (e.g., 'bilinear' or 'nearest'). Got: {upsample_mode}"
            )
        align_corners = config.get("align_corners", False)
        if not isinstance(align_corners, bool):
            raise ValueError(f"align_corners must be a boolean. Got: {align_corners}")

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return UNetDecoder(
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: UNetDecoder,
        features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Sequence[Tensor],
    ) -> Tensor:
        """Decode encoder feature maps into full-resolution logits."""
        c0, c1, c2, c3, c4 = self._normalize_feature_maps(features)

        d3 = self.decode3(c4, c3)
        d2 = self.decode2(d3, c2)
        d1 = self.decode1(d2, c1)
        d0 = self.fuse0(torch.cat((d1, c0), dim=1))

        logits = self.head(d0)
        output_size = (c0.shape[-2] * 4, c0.shape[-1] * 4)
        return F.interpolate(
            logits,
            size=output_size,
            mode=self.upsample_mode,
            align_corners=(
                self.align_corners if self.upsample_mode != "nearest" else None
            ),
        )

    def predict_step(  # pylint: disable=arguments-differ
        self: UNetDecoder,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """Run decoder inference on feature-map batches."""
        del batch_idx, dataloader_idx
        return self.forward(self._extract_feature_maps(batch))

    def model_step(
        self: UNetDecoder,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "UNetDecoder is a decoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def _extract_feature_maps(
        self: UNetDecoder, batch: Any
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Normalize predict batches to a 5-tensor feature-map tuple."""
        if isinstance(batch, dict):
            if "features" in batch:
                return self._normalize_feature_maps(batch["features"])
            if all(key in batch for key in ("c0", "c1", "c2", "c3", "c4")):
                return self._normalize_feature_maps(
                    (
                        batch["c0"],
                        batch["c1"],
                        batch["c2"],
                        batch["c3"],
                        batch["c4"],
                    )
                )
            raise KeyError(
                "Expected a 'features' entry or explicit 'c0'...'c4' tensors in the batch."
            )

        return self._normalize_feature_maps(batch)

    def _normalize_feature_maps(
        self: UNetDecoder,
        features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Validate and normalize decoder inputs to a fixed 5-tensor tuple."""
        if not isinstance(features, Sequence):
            raise TypeError(
                "Expected decoder inputs to be a sequence of five feature maps."
            )
        if len(features) != 5:
            raise ValueError(
                f"Expected five feature maps ordered as (c0, c1, c2, c3, c4), got {len(features)}."
            )

        normalized = tuple(features)
        if not all(isinstance(feature, Tensor) for feature in normalized):
            raise TypeError("All decoder feature maps must be torch.Tensor objects.")

        c0, c1, c2, c3, c4 = normalized
        return c0, c1, c2, c3, c4
