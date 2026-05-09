"""Master test script to run unit tests for every module."""

from argparse import ArgumentParser

from tests.models.slide_level import test_all_models_slide_level

from .datasets import test_all_datasets
from .models import test_all_models
from .models.tile_level import test_all_models_tile_level
from .utils import test_all_utils


if __name__ == "__main__":
    parser = ArgumentParser(description="Run all unit tests for VexDR.")
    parser.add_argument(
        "--datasets",
        action="store_true",
        default=False,
        help="Run only dataset tests.",
    )
    parser.add_argument(
        "--tile-level-models",
        action="store_true",
        default=False,
        help="Run only tile-level model tests.",
    )
    parser.add_argument(
        "--slide-level-models",
        action="store_true",
        default=False,
        help="Run only slide-level model tests.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        default=False,
        help="Run only model tests.",
    )
    parser.add_argument(
        "--utils",
        action="store_true",
        default=False,
        help="Run only utility tests.",
    )

    args = parser.parse_args()
    if args.datasets:
        test_all_datasets()
    if args.tile_level_models:
        test_all_models_tile_level()
    if args.slide_level_models:
        test_all_models_slide_level()
    if args.models:
        test_all_models()
    if args.utils:
        test_all_utils()

    if (
        not args.datasets
        and not args.tile_level_models
        and not args.slide_level_models
        and not args.models
        and not args.utils
    ):
        test_all_datasets()
        test_all_models()
        test_all_utils()
