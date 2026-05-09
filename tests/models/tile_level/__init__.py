"""Unit tests for tile-level models."""

from .dpt_decoder import test_DPTDecoder
from .resnet_encoder import test_ResNetEncoder
from .tile_model import test_TileModel
from .unet_decoder import test_UNetDecoder
from .unet_encoder import test_UNetEncoder
from .vit_encoder import test_ViTEncoder


def test_all_models_tile_level():
    """Run all tile-level model tests."""
    print(" TILE-LEVEL MODEL UNIT TESTS ".center(40, "*"))
    test_ResNetEncoder()
    print("".center(40, "-"))
    test_ViTEncoder()
    print("".center(40, "-"))
    test_DPTDecoder()
    print("".center(40, "-"))
    test_UNetEncoder()
    print("".center(40, "-"))
    test_UNetDecoder()
    print("".center(40, "-"))
    test_TileModel()
    print("".center(40, "*"))
