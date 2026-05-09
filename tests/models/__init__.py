""" "Unit tests for Augur models."""

from .model_abc import test_all_models_model_abc
from .slide_level import test_all_models_slide_level
from .tile_level import test_all_models_tile_level
from .utils import test_all_models_utils


def test_all_models():
    """Run all model unit tests."""
    print(" MODEL UNIT TESTS ".center(40, "="))
    test_all_models_model_abc()
    print("".center(40, "-"))
    test_all_models_utils()
    print("".center(40, "-"))
    test_all_models_tile_level()
    print("".center(40, "-"))
    test_all_models_slide_level()
    print("".center(40, "="))
