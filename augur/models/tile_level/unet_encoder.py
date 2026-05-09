"""Lightning-compatible U-Net-style encoder for tile-level feature extraction."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from augur.models.model_abc import ModelABC
from augur.models.utils import (
    get_lr_scheduler_from_config,
    get_optimizer_from_config,
)


class _ConvBlock(nn.Module):
    """Two Conv-BN-ReLU blocks used throughout the encoder."""

    def __init__(
        self: _ConvBlock,
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

    def forward(self: _ConvBlock, x: Tensor) -> Tensor:
        """Apply the convolutional refinement block."""
        return self.block(x)


class UNetEncoder(ModelABC):
    """U-Net-style encoder exposed as a Lightning module.

    The encoder returns five multi-scale feature maps ordered as ``(c0, c1, c2, c3, c4)``.
    The default pyramid matches the spatial layout: ``c0`` and ``c1`` share quarter-resolution,
    followed by successive downsampling to 8th, 16th, and 32nd resolution.

    Parameters
    ----------
    input_channels
        The number of channels in the input image tensor.
    feature_channels
        A sequence of five channel counts for the output feature maps (c0 through c4).
    dropout
        An optional dropout probability for the convolutional blocks.
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
        self: UNetEncoder,
        input_channels: int = 3,
        feature_channels: Sequence[int] = (64, 64, 128, 256, 512),
        dropout: float = 0.0,
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

        if len(feature_channels) != 5:
            raise ValueError(
                "feature_channels must contain five channel counts: c0 through c4."
            )

        self.input_channels = input_channels
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.dropout = dropout

        c0_channels, c1_channels, c2_channels, c3_channels, c4_channels = (
            self.feature_channels
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                c0_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(c0_channels),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage0 = _ConvBlock(c0_channels, c0_channels, dropout=dropout)
        self.stage1 = _ConvBlock(c0_channels, c1_channels, dropout=dropout)
        self.stage2 = _ConvBlock(c1_channels, c2_channels, dropout=dropout)
        self.stage3 = _ConvBlock(c2_channels, c3_channels, dropout=dropout)
        self.stage4 = _ConvBlock(c3_channels, c4_channels, dropout=dropout)

        self.save_hyperparameters(
            {
                "input_channels": input_channels,
                "feature_channels": self.feature_channels,
                "dropout": dropout,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> UNetEncoder:
        """Construct a UNetEncoder from a configuration dictionary."""

        input_channels = config.get("input_channels", 3)
        if not isinstance(input_channels, int) or input_channels <= 0:
            raise ValueError(
                f"input_channels must be a positive integer. Got: {input_channels}"
            )
        feature_channels = config.get("feature_channels", (64, 64, 128, 256, 512))
        if not isinstance(feature_channels, Sequence) or len(feature_channels) != 5:
            raise ValueError(
                f"feature_channels must be a sequence of five channel counts. Got: {feature_channels}"
            )
        dropout = config.get("dropout", 0.0)
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
            )

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return UNetEncoder(
            input_channels=input_channels,
            feature_channels=feature_channels,
            dropout=dropout,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: UNetEncoder, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return the encoder feature pyramid for an image batch."""
        x = self.stem(x)
        c0 = self.stage0(self.stem_pool(x))
        c1 = self.stage1(c0)
        c2 = self.stage2(self.downsample(c1))
        c3 = self.stage3(self.downsample(c2))
        c4 = self.stage4(self.downsample(c3))

        return c0, c1, c2, c3, c4

    def predict_step(  # pylint: disable=arguments-differ
        self: UNetEncoder,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run encoder inference on a tensor batch or ``{'image': tensor}`` batch."""
        del batch_idx, dataloader_idx
        return self.forward(self._extract_image_tensor(batch))

    def model_step(
        self: UNetEncoder,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "UNetEncoder is an encoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def _extract_image_tensor(self: UNetEncoder, batch: Any) -> Tensor:
        """Normalize predict batches to an image tensor."""
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, dict):
            if "image" not in batch:
                raise KeyError("Expected batch dicts to contain an 'image' tensor.")
            image = batch["image"]
            if not isinstance(image, Tensor):
                raise TypeError(
                    "The 'image' entry in the batch dict must be a torch.Tensor."
                )
            return image

        raise TypeError(
            "Expected a torch.Tensor batch or a dict containing an 'image' tensor."
        )
