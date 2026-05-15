"""Lightning-compatible ResNet encoder for tile-level feature extraction."""

from __future__ import annotations

from typing import Any, Callable

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.models import (
    ResNet101_Weights,
    ResNet152_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from augur.models.model_abc import ModelABC
from augur.models.utils import get_lr_scheduler_from_config, get_optimizer_from_config

_PRETRAINED_SPECS: dict[
    str,
    tuple[
        type[BasicBlock] | type[Bottleneck],
        list[int],
        Callable[..., ResNet],
        Any,
    ],
] = {
    "resnet18": (
        BasicBlock,
        [2, 2, 2, 2],
        resnet18,
        ResNet18_Weights.DEFAULT,
    ),
    "resnet34": (
        BasicBlock,
        [3, 4, 6, 3],
        resnet34,
        ResNet34_Weights.DEFAULT,
    ),
    "resnet50": (
        Bottleneck,
        [3, 4, 6, 3],
        resnet50,
        ResNet50_Weights.DEFAULT,
    ),
    "resnet101": (
        Bottleneck,
        [3, 4, 23, 3],
        resnet101,
        ResNet101_Weights.DEFAULT,
    ),
    "resnet152": (
        Bottleneck,
        [3, 8, 36, 3],
        resnet152,
        ResNet152_Weights.DEFAULT,
    ),
}


def _load_pretrained_state_dict(
    pretrained: str | None,
    block: type[BasicBlock] | type[Bottleneck],
    layers: list[int],
    logger: Any | None = None,
) -> dict[str, Tensor] | None:
    """Load a torchvision state dict when a recognized preset is requested."""
    if pretrained is None:
        return None

    spec = _PRETRAINED_SPECS.get(pretrained)
    if spec is None:
        if logger is not None:
            logger.warning(
                "Pretrained model '%s' not recognized. No pretrained weights will be loaded.",
                pretrained,
            )
        return None

    expected_block, expected_layers, builder, weights = spec
    if block is not expected_block:
        if logger is not None:
            logger.error(
                "%s must use %d, got %s.",
                pretrained,
                expected_block.__name__,
                block.__name__,
            )
        raise ValueError(
            f"{pretrained} must use {expected_block.__name__}, got {block.__name__}."
        )
    if list(layers) != expected_layers:
        if logger is not None:
            logger.error("%s must use layers ")
        raise ValueError(
            f"{pretrained} must use layers {expected_layers}, got {list(layers)}."
        )

    return builder(weights=weights).state_dict()


class _ResNetFeatureExtractor(ResNet):
    """Torchvision ResNet that returns intermediate encoder feature maps."""

    def forward(
        self: _ResNetFeatureExtractor, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return multi-scale feature maps from each ResNet stage."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = self.maxpool(x)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c0, c1, c2, c3, c4


class ResNetEncoder(ModelABC):
    """ResNet encoder exposed as a Lightning module.

    Parameters
    ----------
    block
        The ResNet block type to use (e.g., BasicBlock or Bottleneck).
    layers
        The number of blocks in each ResNet stage (e.g., [2, 2, 2, 2] for ResNet-18).
    pretrained
        An optional name of a pretrained ResNet variant to load from torchvision.
    logger
        An optional logger for warning messages about pretrained loading.
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
        self: ResNetEncoder,
        block: type[BasicBlock] | type[Bottleneck],
        layers: list[int],
        pretrained: str | None = None,
        logger: Any | None = None,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

        self.block = block
        self.layers = list(layers)
        self.pretrained = pretrained
        self.feature_channels = (
            64,
            64 * block.expansion,
            128 * block.expansion,
            256 * block.expansion,
            512 * block.expansion,
        )

        self.backbone = _ResNetFeatureExtractor(block, self.layers, **kwargs)
        state_dict = _load_pretrained_state_dict(
            pretrained=pretrained,
            block=block,
            layers=self.layers,
            logger=logger,
        )
        if state_dict is not None:
            self.backbone.load_state_dict(state_dict, strict=False)
        # The encoder is used only for multi-scale feature extraction, so the
        # classification head would otherwise stay trainable but never receive
        # gradients under DDP.
        self.backbone.fc = nn.Identity()  # type: ignore[assignment]

        self.save_hyperparameters(
            {
                "block": block.__name__,
                "layers": self.layers,
                "pretrained": pretrained,
                "feature_channels": self.feature_channels,
            }
        )

    @property
    def conv1(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stem for compatibility."""
        return self.backbone.conv1

    @property
    def bn1(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stem for compatibility."""
        return self.backbone.bn1

    @property
    def relu(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stem for compatibility."""
        return self.backbone.relu

    @property
    def maxpool(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stem for compatibility."""
        return self.backbone.maxpool

    @property
    def layer1(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stage for compatibility."""
        return self.backbone.layer1

    @property
    def layer2(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stage for compatibility."""
        return self.backbone.layer2

    @property
    def layer3(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stage for compatibility."""
        return self.backbone.layer3

    @property
    def layer4(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet stage for compatibility."""
        return self.backbone.layer4

    @property
    def avgpool(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet head for compatibility."""
        return self.backbone.avgpool

    @property
    def fc(self: ResNetEncoder) -> Any:
        """Expose the wrapped ResNet head for compatibility."""
        return self.backbone.fc

    @staticmethod
    def from_config(config: dict[str, Any]) -> ResNetEncoder:
        """Construct a ResNetEncoder from a configuration dictionary."""

        block_name = config.get("block_name", None)
        if block_name is None:
            raise ValueError("ResNetEncoder config must contain a 'block_name' key.")
        block = {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}.get(
            block_name, None
        )
        if block is None:
            raise ValueError(
                f"ResNetEncoder block_name '{block_name}' is not recognized. "
                "Expected 'BasicBlock' or 'Bottleneck'."
            )

        layers = config.get("layers", None)
        if layers is None:
            raise ValueError("ResNetEncoder config must contain a 'layers' key.")
        if not isinstance(layers, list) or not all(isinstance(n, int) for n in layers):
            raise ValueError("ResNetEncoder 'layers' must be a list of integers.")

        pretrained = config.get("pretrained", None)
        if pretrained is not None and not isinstance(pretrained, str):
            raise ValueError("ResNetEncoder 'pretrained' must be a string or None.")

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return ResNetEncoder(
            block=block,
            layers=layers,
            pretrained=pretrained,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: ResNetEncoder, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return multi-scale feature maps for the input image batch."""
        return self.backbone(x)

    def predict_step(  # pylint: disable=arguments-differ
        self: ResNetEncoder,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run encoder inference on a tensor batch or ``{'image': tensor}`` batch."""
        del batch_idx, dataloader_idx
        return self.forward(self._extract_image_tensor(batch))

    def model_step(
        self: ResNetEncoder,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "ResNetEncoder is an encoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def _extract_image_tensor(self: ResNetEncoder, batch: Any) -> Tensor:
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
