"""Unit tests for the jigmag task functions."""

import os
import random

import numpy as np
from openslide import OpenSlide

from VexDR.datasets.jigmag import build_jigmag_permutations, process_jigmag_task
from VexDR.datasets.utils import (
    load_slide_records,
    tile_record_center_l0,
    resolve_manifest_path,
    sample_tile_records,
)


def _test_build_jigmag_permutations():
    """Test that the correct number of permutations are generated."""
    print("Testing build_jigmag_permutations...")
    num_tiles = 4
    permutations = build_jigmag_permutations(num_tiles)
    expected_num_permutations = 24  # 4! = 4 * 3 * 2 * 1
    assert len(permutations) == expected_num_permutations, (
        f"Expected {expected_num_permutations} permutations. "
        f"Got: {len(permutations)}"
    )
    print("[OK] build_jigmag_permutations test passed.")


def _test_process_jigmag_task():
    """Test that the process_jigmag_task function runs without errors."""
    print("Testing process_jigmag_task...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    slide = OpenSlide(slide_records[0].slide_path)
    tile_record = sample_tile_records(
        [slide_records[0]],
        output_size=512,
        base_mpp=0.25,
        context_mpp=1.0,
        tiles_per_slide=1,
        min_tissue_fraction=0.5,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        seed=42,
    )[0]
    center_x, center_y = tile_record_center_l0(slide, tile_record)

    target_mpps = [0.25, 0.5, 1.0, 2.0]

    result = process_jigmag_task(
        slide=slide,
        center_x=center_x,
        center_y=center_y,
        output_size=512,
        target_mpps=target_mpps,
        rng=random.Random(42),
    )

    assert "image" in result
    assert "target" in result
    assert "permutation" in result
    assert "mpps" in result
    assert isinstance(result["image"], np.ndarray)
    assert isinstance(result["target"], int)
    assert isinstance(result["permutation"], tuple)
    assert isinstance(result["mpps"], tuple)

    # Check that the mpps in the result match the input target_mpps
    assert sorted(result["mpps"]) == sorted(
        target_mpps
    ), f"Expected mpps {target_mpps}. Got: {result['mpps']}"
    # Check the shape of the image
    assert result["image"].shape == (
        512,
        512,
        3,
    ), f"Expected image shape (512, 512, 3). Got: {result['image'].shape}"
    # Check that the target permutation index is valid
    perm = result["permutation"]
    all_perms = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (1, 2, 0, 3),
        (1, 2, 3, 0),
        (1, 3, 0, 2),
        (1, 3, 2, 0),
        (2, 0, 1, 3),
        (2, 0, 3, 1),
        (2, 1, 0, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 0, 1, 2),
        (3, 0, 2, 1),
        (3, 1, 0, 2),
        (3, 1, 2, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    ]
    assert perm in all_perms, f"Unexpected permutation: {perm}"
    expected_index = all_perms.index(perm)
    assert result["target"] == expected_index, (
        f"Expected target index {expected_index} for permutation {perm}. "
        f"Got: {result['target']}"
    )
    print("[OK] process_jigmag_task test passed.")


def test_all_datasets_jigmag():
    """Run all jigmag task tests."""
    print("Running jigmag task tests...")
    _test_build_jigmag_permutations()
    _test_process_jigmag_task()
    print("All jigmag task tests passed!")


if __name__ == "__main__":
    test_all_datasets_jigmag()
