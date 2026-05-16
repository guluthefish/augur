"""This module serves as the entry point for running all dataset-related tests."""

from .hematoxylin import test_all_datasets_hematoxylin
from .magnification import test_all_datasetes_magnification
from .utils import test_all_datasets_utils
from .jigmag import test_all_datasets_jigmag
from .tcga_tile_dataset import test_all_datasets_tcga
from .tcga_slide_dataset import test_all_datasets_slide
from .tcga_feature_dataset import test_all_datasets_feature
from .factory import test_all_datasets_factory


def test_all_datasets():
    """Run all dataset-related tests."""
    print(" DATASET UNIT TESTS ".center(40, "="))
    test_all_datasets_hematoxylin()
    print("".center(40, "-"))
    test_all_datasetes_magnification()
    print("".center(40, "-"))
    test_all_datasets_utils()
    print("".center(40, "-"))
    test_all_datasets_jigmag()
    print("".center(40, "-"))
    test_all_datasets_tcga()
    print("".center(40, "-"))
    test_all_datasets_slide()
    print("".center(40, "-"))
    test_all_datasets_feature()
    print("".center(40, "-"))
    test_all_datasets_factory()
    print("".center(40, "="))
