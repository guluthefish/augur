"""Unit tests for slide-level models."""

from tests.models.slide_level.dual_clam import test_DualCLAM

from .attention import test_Attention
from .mil import test_EmbeddingMIL


def test_all_models_slide_level() -> None:
    """Run all slide-level model tests."""
    print(" SLIDE-LEVEL MODEL UNIT TESTS ".center(40, "*"))
    test_Attention()
    print("".center(40, "-"))
    test_EmbeddingMIL()
    print("".center(40, "-"))
    test_DualCLAM()
    print("".center(40, "*"))
