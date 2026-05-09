"""Factory for creating tile-level modules from configuration."""

from typing import Any

from augur.models.model_abc import ModelABC
from augur.models.tile_level.classifier import Classifier
from augur.models.tile_level.dpt_decoder import DPTDecoder
from augur.models.tile_level.resnet_encoder import ResNetEncoder
from augur.models.tile_level.unet_decoder import UNetDecoder
from augur.models.tile_level.unet_encoder import UNetEncoder
from augur.models.tile_level.vit_encoder import ViTEncoder


def get_module_from_config(config: dict[str, Any]) -> ModelABC:
    """Factory function to create tile-level modules based on config."""

    module_name = config.get("name")
    match module_name:
        case "ResNetEncoder":
            return ResNetEncoder.from_config(config.get("params", {}))
        case "UNetEncoder":
            return UNetEncoder.from_config(config.get("params", {}))
        case "UNetDecoder":
            return UNetDecoder.from_config(config.get("params", {}))
        case "DPTDecoder":
            return DPTDecoder.from_config(config.get("params", {}))
        case "ViTEncoder":
            return ViTEncoder.from_config(config.get("params", {}))
        case "Classifier":
            return Classifier.from_config(config.get("params", {}))
        case _:
            raise ValueError(f"Unsupported tile-level module name: {module_name}")
