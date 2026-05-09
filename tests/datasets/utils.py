"""Unit test for utility functions in VexDR datasets."""

import os


import numpy as np
from openslide import OpenSlide
import pandas as pd
from PIL import Image

import torch

from VexDR.datasets.utils import (
    as_image_tensor,
    as_mask_tensor,
    compute_tissue_mask,
    derive_bcss_slide_name,
    enumerate_slide_tile_centers,
    load_slide_records,
    load_tissue_mask_label,
    tile_record_center_l0,
    read_tile_at_mpp,
    read_tile_from_record,
    resolve_manifest_path,
    resolve_slide_main_label_path,
    resolve_slide_pretext_label_path,
    sample_tile_record_in_roi_bounds,
    sample_tile_records,
    scaled_thumbnail_size,
    split_slide_records,
)


def _test_resolve_manifest_path():
    """Test that the manifest path is resolved correctly."""
    print("Testing _resolve_manifest_path...")
    root_dir = "data/TCGA-BRCA-test"

    # Test with a provided manifest path
    manifest_path = "data/TCGA-BRCA-test/manifests/downloaded/gdc_manifest.2026-03-17_16-01-09.moved.txt"

    resolved_path = resolve_manifest_path(root_dir, manifest_path)
    assert (
        resolved_path == manifest_path
    ), f"Resolved manifest path does not match provided path. Expected: {manifest_path}. Got: {resolved_path}"

    # Test with a manifest path that does not exist
    non_existent_path = (
        "data/TCGA-BRCA-test/manifests/downloaded/non_existent_manifest.txt"
    )
    try:
        resolve_manifest_path(root_dir, non_existent_path)
        assert False, "Expected FileNotFoundError for non-existent manifest path"
    except FileNotFoundError:
        pass  # Expected exception

    # Test with no manifest path provided (should look up in atlas)
    resolved_path = resolve_manifest_path(root_dir, None)
    assert os.path.exists(
        resolved_path
    ), f"Resolved manifest path does not exist. Got: {resolved_path}"

    print("[OK] _resolve_manifest_path test passed!")


def _test_load_slide_records():
    """Test that slide records are loaded correctly from the manifest."""
    print("Testing _load_slide_records...")
    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)
    # Load slide records using the manifest
    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )
    # Check that slide records is a list
    assert isinstance(slide_records, list), "Slide records is not a list."
    # Check the number of slide records matches the number of entries in the manifest
    manifest_df = pd.read_table(manifest_path, index_col=0, dtype=str)
    manifest_df = manifest_df[manifest_df["data_type"] == "Slide Image"]
    assert len(slide_records) == len(
        manifest_df
    ), f"Number of slide records does not match number of entries in manifest. Expected {len(manifest_df)}, got {len(slide_records)}."
    print("[OK] _load_slide_records test passed!")


def _test_split_slide_records():
    """Test that slide records are split correctly into train/val/test."""
    print("Testing _split_slide_records...")
    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)
    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )
    # Check that we have some slide records
    assert len(slide_records) > 0, "No slide records loaded to split."
    # Split the records into train/val/test
    # num_slides = len(slide_records)
    # num_train = int(round(0.7 * num_slides))
    # num_val = int(round(0.15 * num_slides))
    n_submitters = len(set(record.submitter_id for record in slide_records))
    n_train = int(round(0.7 * n_submitters))
    n_val = int(round(0.15 * n_submitters))

    records_dict = split_slide_records(
        slide_records,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=1 - 0.7 - 0.15,
        seed=42,
    )
    train_records = records_dict["train"]
    val_records = records_dict["val"]
    test_records = records_dict["test"]

    # Check that the number of records in each split matches the expected numbers
    train_submitters = set(record.submitter_id for record in train_records)
    val_submitters = set(record.submitter_id for record in val_records)
    test_submitters = set(record.submitter_id for record in test_records)

    assert (
        len(train_submitters) == n_train
    ), f"Expected {n_train} unique submitters in train set, got {len(train_submitters)}."
    assert (
        len(val_submitters) == n_val
    ), f"Expected {n_val} unique submitters in validation set, got {len(val_submitters)}."
    assert (
        len(test_submitters) == n_submitters - n_train - n_val
    ), f"Expected {n_submitters - n_train - n_val} unique submitters in test set, got {len(test_submitters)}."

    assert set(train_records).isdisjoint(
        set(val_records)
    ), "Train and validation records are not mutually exclusive."
    assert set(train_records).isdisjoint(
        set(test_records)
    ), "Train and test records are not mutually exclusive."
    assert set(val_records).isdisjoint(
        set(test_records)
    ), "Validation and test records are not mutually exclusive."
    print("[OK] _split_slide_records test passed!")


def _test_scaled_thumbnail_size():
    """Test that the scaled thumbnail size is calculated correctly."""
    print("Testing scaled_thumbnail_size...")
    # Test with a square image
    width, height = 1000, 1000
    max_size = 500
    expected_size = (500, 500)
    assert (
        scaled_thumbnail_size(width, height, max_size) == expected_size
    ), f"Expected {expected_size} for square image. Got: {scaled_thumbnail_size(width, height, max_size)}"

    # Test with a landscape image
    width, height = 2000, 1000
    max_size = 500
    expected_size = (500, 250)
    assert (
        scaled_thumbnail_size(width, height, max_size) == expected_size
    ), f"Expected {expected_size} for landscape image. Got: {scaled_thumbnail_size(width, height, max_size)}"

    # Test with a portrait image
    width, height = 1000, 2000
    max_size = 500
    expected_size = (250, 500)
    assert (
        scaled_thumbnail_size(width, height, max_size) == expected_size
    ), f"Expected {expected_size} for portrait image. Got: {scaled_thumbnail_size(width, height, max_size)}"

    # Test with an image smaller than the max size (should not upscale)
    width, height = 400, 300
    max_size = 500
    expected_size = (400, 300)
    assert (
        scaled_thumbnail_size(width, height, max_size) == expected_size
    ), f"Expected {expected_size} for small image. Got: {scaled_thumbnail_size(width, height, max_size)}"

    print("[OK] scaled_thumbnail_size test passed!")


def _test_compute_tissue_mask():
    """Test that the tissue mask is computed correctly from the thumbnail."""
    print("Testing compute_tissue_mask...")
    # Create a simple thumbnail with a bright background and a darker tissue region
    thumbnail = np.array(
        [
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
            [[255, 255, 255], [100, 150, 200], [255, 255, 255]],
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    white_threshold = 0.9
    expected_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    tissue_mask = compute_tissue_mask(
        thumbnail=thumbnail, white_threshold=white_threshold
    )
    assert np.array_equal(
        tissue_mask, expected_mask
    ), f"Expected tissue mask does not match computed mask. Expected:\n{expected_mask}\nGot:\n{tissue_mask}"

    print("[OK] compute_tissue_mask test passed!")


def _test_derive_bcss_slide_name():
    """Test that BCSS slide names are derived correctly from slide filenames."""
    print("Testing derive_bcss_slide_name...")
    slide_path = "data/TCGA-BRCA-test/ordered_data/TCGA-A1-A0SK/images/66fd32f2-3f81-4914-9394-4795414893bd/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297.svs"

    # Check with submitter_id
    slide_name = derive_bcss_slide_name(slide_path, "TCGA-A1-A0SK")
    assert (
        slide_name == "TCGA-A1-A0SK-DX1"
    ), f"Expected TCGA-A1-A0SK-DX1. Got: {slide_name}"

    # Check without submitter_id
    slide_name = derive_bcss_slide_name(slide_path, None)
    assert (
        slide_name == "TCGA-A1-A0SK-DX1"
    ), f"Expected TCGA-A1-A0SK-DX1. Got: {slide_name}"

    print("[OK] derive_bcss_slide_name test passed!")


def _test_sample_tile_record_in_roi_bounds():
    """Test that sampled tile records have centers within the ROI bounds."""
    print("Testing sample_tile_record_in_roi_bounds...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    slide_name = "TCGA-A1-A0SK-DX1"
    slide_record = None
    for s in slide_records:
        if derive_bcss_slide_name(s.slide_path) == slide_name:
            slide_record = s
            break
    assert (
        slide_record is not None
    ), f"No slide record found for slide_name {slide_name}"

    roi_name = derive_bcss_slide_name(slide_record.slide_path)
    xmin, ymin, xmax, ymax = 45749, 25055, 49789, 27991
    tile_record = sample_tile_record_in_roi_bounds(
        slide_record=slide_record,
        roi_name=roi_name,
        roi_xmin=xmin,
        roi_ymin=ymin,
        roi_xmax=xmax,
        roi_ymax=ymax,
        output_size=512,
        base_mpp=0.25,
        seed=42,
    )

    center_x, center_y = tile_record_center_l0(
        OpenSlide(slide_record.slide_path), tile_record
    )

    assert (
        xmin <= center_x < xmax
    ), f"Sampled tile center_x {center_x} is out of ROI bounds [{xmin}, {xmax})."
    assert (
        ymin <= center_y < ymax
    ), f"Sampled tile center_y {center_y} is out of ROI bounds [{ymin}, {ymax})."

    print("[OK] sample_tile_record_in_roi_bounds test passed!")


def _test_load_tissue_mask():
    """Test that BCSS tissue masks are padded with label 0 outside the ROI."""
    print("Testing load_tissue_mask...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    slide_name = "TCGA-A1-A0SK-DX1"
    slide_record = None
    for s in slide_records:
        if derive_bcss_slide_name(s.slide_path) == slide_name:
            slide_record = s
            break
    assert (
        slide_record is not None
    ), f"No slide record found for slide_name {slide_name}"
    slide = OpenSlide(slide_record.slide_path)

    roi_name = derive_bcss_slide_name(slide_record.slide_path)
    xmin, ymin, xmax, ymax = 45749, 25055, 49789, 27991
    tile_record = sample_tile_record_in_roi_bounds(
        slide_record=slide_record,
        roi_name=roi_name,
        roi_xmin=xmin,
        roi_ymin=ymin,
        roi_xmax=xmax,
        roi_ymax=ymax,
        output_size=512,
        base_mpp=0.25,
        seed=42,
    )

    center_x, center_y = tile_record_center_l0(slide, tile_record)
    tile_image = read_tile_at_mpp(
        slide,
        center_x=center_x,
        center_y=center_y,
        output_size=512,
        target_mpp=0.25,
    )
    tile_mask = load_tissue_mask_label(
        root_dir=root_dir,
        tile_record=tile_record,
        slide=slide,
        base_mpp=0.25,
        output_size=512,
    )
    assert tile_mask.shape == (
        512,
        512,
    ), f"Expected tile mask shape (512, 512). Got: {tile_mask.shape}"

    # Optional: Plot the image and the mask provided by BCSS dataset to visually confirm that the sampled tile is within the tissue region (requires matplotlib)
    image_path = "data/TCGA-BRCA-test/labels/tissues/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"
    mask_path = "data/TCGA-BRCA-test/labels/tissues/masks/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    roi_width_l0 = max(int(tile_record.roi_xmax) - int(tile_record.roi_xmin), 1)  # type: ignore
    roi_height_l0 = max(int(tile_record.roi_ymax) - int(tile_record.roi_ymin), 1)  # type: ignore
    tile_extent_l0 = tile_record.size * float(
        slide.level_downsamples[tile_record.level]
    )

    def crop_to_tile_region(pil_image: Image.Image) -> Image.Image:
        scale_x = pil_image.width / roi_width_l0
        scale_y = pil_image.height / roi_height_l0
        left = int(round((tile_record.x - int(tile_record.roi_xmin)) * scale_x))  # type: ignore
        top = int(round((tile_record.y - int(tile_record.roi_ymin)) * scale_y))  # type: ignore
        width = max(int(round(tile_extent_l0 * scale_x)), 1)
        height = max(int(round(tile_extent_l0 * scale_y)), 1)
        return pil_image.crop((left, top, left + width, top + height))

    image = crop_to_tile_region(image)
    mask = crop_to_tile_region(mask)

    # import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # _, ax = plt.subplots(1, 4, figsize=(20, 5))
    # ax[0].imshow(image)
    # ax[0].set_title("BCSS Image")
    # ax[0].axis("off")
    # ax[1].imshow(mask)
    # ax[1].set_title("BCSS Mask")
    # ax[1].axis("off")
    # ax[2].imshow(tile_image)
    # ax[2].set_title("Sampled Tile Image")
    # ax[2].axis("off")
    # ax[3].imshow(tile_mask, cmap="tab20")
    # ax[3].set_title("Loaded Tissue Mask")
    # ax[3].axis("off")
    # plt.suptitle("load_tissue_mask Test Visualization")
    # plt.axis("off")
    # plt.show()

    print("[OK] load_tissue_mask test passed!")


def _test_as_image_tensor():
    """Test that the as_image_tensor function converts a thumbnail to a tensor correctly."""
    print("Testing as_image_tensor...")
    # Create a simple thumbnail with known values
    thumbnail = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )  # CHW format
    thumbnail = thumbnail.transpose(1, 2, 0)  # Convert to HWC
    thumbnail = thumbnail.astype(np.float32) / 255.0
    expected_tensor = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    image_tensor = as_image_tensor(thumbnail)
    assert torch.allclose(
        image_tensor, expected_tensor
    ), f"Expected image tensor does not match computed tensor. Expected:\n{expected_tensor}\nGot:\n{image_tensor}"

    print("[OK] as_image_tensor test passed!")


def _test_as_mask_tensor():
    """Test that the as_mask_tensor function converts a tissue mask to a tensor correctly."""
    print("Testing as_mask_tensor...")
    # Create a simple tissue mask with known values
    tissue_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    expected_tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask_tensor = as_mask_tensor(tissue_mask)
    assert torch.allclose(
        mask_tensor, expected_tensor
    ), f"Expected mask tensor does not match computed tensor. Expected:\n{expected_tensor}\nGot:\n{mask_tensor}"

    print("[OK] as_mask_tensor test passed!")


def _test_sample_tile_records():
    """Test that tile records are sampled correctly from a slide record."""
    print("Testing sample_tile_records...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)
    # Load slide records using the manifest
    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    # Sample tile records
    tile_records = sample_tile_records(
        slide_records,
        output_size=512,
        base_mpp=0.25,
        context_mpp=1.0,
        tiles_per_slide=10,
        min_tissue_fraction=0.5,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        seed=42,
    )

    # Check that tile records is a list
    assert isinstance(tile_records, list), "Tile records is not a list."
    # Check length of tile records matches expected number (10 tiles per slide)
    expected_num_tiles = len(slide_records) * 10
    assert (
        len(tile_records) <= expected_num_tiles
    ), f"Expected at most {expected_num_tiles} tile records, got {len(tile_records)}."
    print("[OK] sample_tile_records test passed!")


def _test_read_tile_at_mpp():
    """Test that a tile can be read correctly at the specified MPP."""
    print("Testing read_tile_at_mpp...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    # Read a tile from the first tile record at target MPP of 0.5
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
    tile_image = read_tile_at_mpp(
        slide,
        center_x=tile_record.x,
        center_y=tile_record.y,
        output_size=512,
        target_mpp=0.5,
    )

    # Check that the tile image has the expected dtype and shape
    assert (
        tile_image.dtype == np.uint8
    ), f"Expected tile dtype uint8. Got: {tile_image.dtype}"
    assert tile_image.shape == (
        512,
        512,
        3,
    ), f"Expected tile shape (512, 512, 3). Got: {tile_image.shape}"

    # Optional: Plot the tile image to visually inspect it (requires matplotlib)
    # import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # plt.imshow(tile_image)
    # plt.title("read_tile_at_mpp Extracted Tile Image")
    # plt.axis("off")
    # plt.show()

    print("[OK] read_tile_at_mpp test passed.")


def _test_read_tile_from_record():
    """Test that a tile can be read correctly from a tile record."""
    print("Testing read_tile_from_record...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    # Read a tile from the first tile record
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
    tile_image = read_tile_from_record(slide, tile_record)

    # Check that the tile image has the expected dtype
    assert (
        tile_image.dtype == np.uint8
    ), f"Expected tile dtype uint8. Got: {tile_image.dtype}"

    # Optional: Plot the tile image to visually inspect it (requires matplotlib)
    # import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # plt.imshow(tile_image)
    # plt.title("read_tile_from_record Extracted Tile Image")
    # plt.axis("off")
    # plt.show()

    print("[OK] read_tile_from_record test passed.")


def _test_tile_record_center_l0():
    """Test that the center coordinates of a tile record correspond to the expected level 0 coordinates."""
    print("Testing tile_record_center_l0...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    # Sample a tile record
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

    # Calculate expected level 0 coordinates based on the tile record's x, y, and mpp
    slide = OpenSlide(slide_records[0].slide_path)
    center_x, center_y = tile_record_center_l0(slide, tile_record)

    downsample_factor = slide.level_downsamples[tile_record.level]

    extent_l0 = tile_record.size * downsample_factor
    expected_center_x = int(round(tile_record.x + extent_l0 / 2.0))
    expected_center_y = int(round(tile_record.y + extent_l0 / 2.0))

    assert (
        center_x == expected_center_x
    ), f"Expected center_x {expected_center_x}, got {center_x}."
    assert (
        center_y == expected_center_y
    ), f"Expected center_y {expected_center_y}, got {center_y}."

    print("[OK] tile_record_center_l0 test passed.")


def _test_resolve_slide_main_label_path():
    """Test that the slide-level main-task label path resolves via the atlas."""
    print("Testing resolve_slide_main_label_path...")
    root_dir = "data/TCGA-BRCA-test"

    # Provided explicit path that exists is returned as-is.
    atlas_path = os.path.join(root_dir, "atlases", "slide_main_atlas.txt")
    resolved_explicit = resolve_slide_main_label_path(root_dir, "subtyping", atlas_path)
    assert (
        resolved_explicit == atlas_path
    ), f"Expected explicit path to be returned. Got: {resolved_explicit}"

    # Missing explicit path raises FileNotFoundError.
    try:
        resolve_slide_main_label_path(
            root_dir, "subtyping", "data/TCGA-BRCA-test/labels/does_not_exist.txt"
        )
        assert False, "Expected FileNotFoundError for non-existent main labels path."
    except FileNotFoundError:
        pass

    # Atlas-driven resolution returns an existing file for the supported task.
    resolved_via_atlas = resolve_slide_main_label_path(root_dir, "subtyping", None)
    assert os.path.exists(resolved_via_atlas), (
        "Atlas-resolved subtyping labels path should exist. "
        f"Got: {resolved_via_atlas}"
    )

    print("[OK] resolve_slide_main_label_path test passed.")


def _test_resolve_slide_pretext_label_path():
    """Test that the slide-level pretext-task label path resolves via the atlas."""
    print("Testing resolve_slide_pretext_label_path...")
    root_dir = "data/TCGA-BRCA-test"

    # Provided explicit path that exists is returned as-is.
    atlas_path = os.path.join(root_dir, "atlases", "slide_pretext_atlas.txt")
    resolved_explicit = resolve_slide_pretext_label_path(
        root_dir, "sbs_regression", atlas_path
    )
    assert (
        resolved_explicit == atlas_path
    ), f"Expected explicit path to be returned. Got: {resolved_explicit}"

    # Missing explicit path raises FileNotFoundError.
    try:
        resolve_slide_pretext_label_path(
            root_dir,
            "sbs_regression",
            "data/TCGA-BRCA-test/labels/does_not_exist.txt",
        )
        assert False, "Expected FileNotFoundError for non-existent pretext labels path."
    except FileNotFoundError:
        pass

    # Atlas-driven resolution returns an existing file for each supported task.
    for pretext_task in (
        "sbs_regression",
        "sbs_thresholded_multilabel",
        "sbs_ranked_multilabel",
    ):
        resolved_via_atlas = resolve_slide_pretext_label_path(
            root_dir, pretext_task, None
        )
        assert os.path.exists(resolved_via_atlas), (
            f"Atlas-resolved pretext labels path for '{pretext_task}' should exist. "
            f"Got: {resolved_via_atlas}"
        )

    print("[OK] resolve_slide_pretext_label_path test passed.")


def _test_enumerate_slide_tile_centers():
    """Test that ``stride`` controls the spacing of enumerated tile centers."""
    print("Testing enumerate_slide_tile_centers...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)
    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )
    assert slide_records, "Test manifest should contain at least one slide."

    slide_record = slide_records[0]
    output_size = 512

    # Default stride (= output_size) yields non-overlapping tiles.
    centers_default = enumerate_slide_tile_centers(
        slide_record,
        output_size=output_size,
        context_mpp=0.25,
        min_tissue_fraction=0.25,
        thumbnail_max_size=1024,
        white_threshold=0.8,
    )
    assert isinstance(
        centers_default, list
    ), "enumerate_slide_tile_centers should return a list."
    assert centers_default, "At least one tissue tile should be enumerated."
    for center_x, center_y in centers_default:
        assert isinstance(center_x, int) and isinstance(
            center_y, int
        ), "Each candidate must be an (int, int) center."

    # Halving the stride should yield strictly more candidates (denser sampling).
    centers_dense = enumerate_slide_tile_centers(
        slide_record,
        output_size=output_size,
        context_mpp=0.25,
        min_tissue_fraction=0.25,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        stride=output_size // 2,
    )
    assert len(centers_dense) >= len(centers_default), (
        "Halving the stride should not decrease the candidate count. "
        f"Got default={len(centers_default)}, dense={len(centers_dense)}."
    )

    # Doubling the stride should yield no more candidates than the default.
    centers_sparse = enumerate_slide_tile_centers(
        slide_record,
        output_size=output_size,
        context_mpp=0.25,
        min_tissue_fraction=0.25,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        stride=output_size * 2,
    )
    assert len(centers_sparse) <= len(centers_default), (
        "Doubling the stride should not increase the candidate count. "
        f"Got default={len(centers_default)}, sparse={len(centers_sparse)}."
    )

    # Invalid stride is rejected.
    try:
        enumerate_slide_tile_centers(
            slide_record,
            output_size=output_size,
            context_mpp=0.25,
            min_tissue_fraction=0.25,
            thumbnail_max_size=1024,
            white_threshold=0.8,
            stride=0,
        )
        assert False, "Expected ValueError for non-positive stride."
    except ValueError:
        pass

    print("[OK] enumerate_slide_tile_centers test passed.")


def test_all_datasets_utils():
    """Run all tests for VexDR.datasets.utils."""
    print("Running dataset utility tests...")
    _test_resolve_manifest_path()
    _test_resolve_slide_main_label_path()
    _test_resolve_slide_pretext_label_path()
    _test_load_slide_records()
    _test_split_slide_records()
    _test_scaled_thumbnail_size()
    _test_compute_tissue_mask()
    _test_derive_bcss_slide_name()
    _test_sample_tile_record_in_roi_bounds()
    _test_load_tissue_mask()
    _test_as_image_tensor()
    _test_as_mask_tensor()
    _test_sample_tile_records()
    _test_read_tile_at_mpp()
    _test_read_tile_from_record()
    _test_tile_record_center_l0()
    _test_enumerate_slide_tile_centers()

    print("All dataset utility tests passed!")


if __name__ == "__main__":
    test_all_datasets_utils()
