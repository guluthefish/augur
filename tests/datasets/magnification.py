"""Unit tests for the magnification task functions."""

import os
import random

import numpy as np
from openslide import OpenSlide

from VexDR.datasets.magnification import process_magnification_task
from VexDR.datasets.utils import (
    load_slide_records,
    tile_record_center_l0,
    resolve_manifest_path,
    sample_tile_records,
)


def _test_process_magnification_task():
    """Test that process_magnification_task returns expected keys and types."""
    print("Testing process_magnification_task...")

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

    result = process_magnification_task(
        slide=slide,
        center_x=center_x,
        center_y=center_y,
        output_size=512,
        target_mpps=target_mpps,
        rng=random.Random(42),
    )

    assert "image" in result
    assert "target" in result
    assert "target_mpp" in result
    assert "mpp_candidates" in result
    assert isinstance(result["image"], np.ndarray)
    assert isinstance(result["target"], int)
    assert isinstance(result["target_mpp"], float)
    assert isinstance(result["mpp_candidates"], tuple)

    # Check that the target_mpp is one of the candidates
    assert result["target_mpp"] in target_mpps

    # Check the shape of the image
    assert result["image"].shape == (512, 512, 3)

    print("[OK] test_process_magnification_task test passed.")


def test_all_datasetes_magnification():
    """Run all magnification task tests."""
    print("Running magnification task tests...")
    _test_process_magnification_task()
    print("All magnification task tests passed!")


if __name__ == "__main__":
    test_all_datasetes_magnification()
