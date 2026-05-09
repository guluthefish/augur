"""Unit tests for VexDR utilities."""

from .config import test_all_utils_config
from .logger import test_all_utils_logger


def test_all_utils():
    """Run all tests in VexDR/utils."""
    print(" UTILS UNIT TESTS ".center(40, "="))
    test_all_utils_config()
    test_all_utils_logger()
    print("".center(40, "="))
