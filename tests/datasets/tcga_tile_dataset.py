"""Unit tests for the TCGATileDataset module."""

import os

import torch

from augur.datasets.tcga_tile_dataset import TCGATileDataset, _TileDataset
from augur.datasets.utils import (
    derive_bcss_slide_name,
    load_slide_records,
    resolve_manifest_path,
    sample_tile_record_in_roi_bounds,
    sample_tile_records,
)


def _test_TileDataset():
    """Test that _TileDataset can be instantiated and returns expected keys."""

    print("Testing _TileDataset...")

    root_dir = "data/TCGA-BRCA-test"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    test_slide_name = "TCGA-A1-A0SK-DX1"
    test_slide_record = None
    for s in slide_records:
        if derive_bcss_slide_name(s.slide_path) == test_slide_name:
            test_slide_record = s
            break
    assert test_slide_record is not None, (
        f"No slide record found for slide_name {test_slide_name}"
    )

    roi_name = derive_bcss_slide_name(test_slide_record.slide_id)
    xmin, ymin, xmax, ymax = 45749, 25055, 49789, 27991
    test_tile_record = sample_tile_record_in_roi_bounds(
        slide_record=test_slide_record,
        roi_name=roi_name,
        roi_xmin=xmin,
        roi_ymin=ymin,
        roi_xmax=xmax,
        roi_ymax=ymax,
        output_size=512,
        base_mpp=0.25,
        seed=42,
    )

    tile_records = [test_tile_record]

    tile_records.extend(
        sample_tile_records(
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
    )

    tile_dataset = _TileDataset(
        root_dir=root_dir,
        records=tile_records,
        tasks=["tissue_segmentation", "magnification", "hematoxylin", "jigmag"],
        tile_size=512,
        image_size=128,
        base_mpp=0.25,
        magnification_mpps=[0.25, 0.5, 1.0, 2.0],
        jigmag_mpps=[0.25, 0.5, 1.0, 2.0],
        random_seed=42,
        tissue_segmentation_n_classes=22,
    )

    assert len(tile_dataset) == len(tile_records), (
        f"Expected dataset length {len(tile_records)}. Got: {len(tile_dataset)}"
    )

    sample_item = tile_dataset[0]

    expected_keys = {
        "metadata",
        "tissue_segmentation",
        "magnification",
        "hematoxylin",
        "jigmag",
    }

    assert set(sample_item.keys()) == expected_keys, (
        f"Expected keys {expected_keys}. Got: {set(sample_item.keys())}"
    )

    # Check that metadata contains expected keys and values
    metadata = sample_item["metadata"]
    expected_keys = {
        "slide_id",
        "submitter_id",
        "x",
        "y",
        "level",
        "size",
        "center_x",
        "center_y",
        "base_mpp",
    }

    extra_expected_keys = {
        "slide_id",
        "submitter_id",
        "x",
        "y",
        "level",
        "size",
        "center_x",
        "center_y",
        "base_mpp",
        "roi_name",
        "roi_xmin",
        "roi_ymin",
        "roi_xmax",
        "roi_ymax",
    }

    assert (
        set(metadata.keys()) == expected_keys
        or set(metadata.keys()) == extra_expected_keys
    ), (
        f"Expected metadata keys {expected_keys} or {extra_expected_keys}. Got: {set(metadata.keys())}"
    )
    assert metadata["slide_id"] == tile_records[0].slide_id, (
        f"Expected slide_id {tile_records[0].slide_id}. Got: {metadata['slide_id']}"
    )
    assert metadata["submitter_id"] == tile_records[0].submitter_id, (
        f"Expected submitter_id {tile_records[0].submitter_id}. Got: {metadata['submitter_id']}"
    )
    assert metadata["x"] == tile_records[0].x, (
        f"Expected x {tile_records[0].x}. Got: {metadata['x']}"
    )
    assert metadata["y"] == tile_records[0].y, (
        f"Expected y {tile_records[0].y}. Got: {metadata['y']}"
    )
    assert metadata["level"] == tile_records[0].level, (
        f"Expected level {tile_records[0].level}. Got: {metadata['level']}"
    )
    assert metadata["size"] == tile_records[0].size, (
        f"Expected size {tile_records[0].size}. Got: {metadata['size']}."
    )
    assert metadata["base_mpp"] == 0.25, (
        f"Expected base_mpp 0.25. Got: {metadata['base_mpp']}"
    )

    # Check that tissue segmentation task returns expected keys and types
    tissue_item = sample_item["tissue_segmentation"]
    expected_keys = {"image", "target"}
    assert set(tissue_item.keys()) == expected_keys, (
        f"Expected tissue_segmentation keys {expected_keys}. Got: {set(tissue_item.keys())}"
    )
    assert isinstance(tissue_item["image"], torch.Tensor), (
        f"Tissue segmentation image should be a torch tensor. Got: {type(tissue_item['image'])}"
    )
    assert tissue_item["image"].shape == (
        3,
        tile_dataset.image_size,
        tile_dataset.image_size,
    ), (
        f"Tissue segmentation image should have shape (3, {tile_dataset.image_size}, {tile_dataset.image_size}). Got: {tissue_item['image'].shape}"
    )
    assert torch.all((tissue_item["image"] >= 0) & (tissue_item["image"] <= 1)), (
        "Tissue segmentation image values should be in the range [0, 1]"
    )
    assert isinstance(tissue_item["target"], torch.Tensor), (
        f"Tissue segmentation target should be a torch tensor. Got: {type(tissue_item['target'])}"
    )
    assert tissue_item["target"].shape == (
        22,
        tile_dataset.image_size,
        tile_dataset.image_size,
    ), (
        f"Tissue segmentation target should have shape (22, {tile_dataset.image_size}, {tile_dataset.image_size}). Got: {tissue_item['target'].shape}"
    )
    assert torch.all((tissue_item["target"] >= 0) & (tissue_item["target"] <= 1)), (
        "Tissue segmentation target should be a binary tensor with values 0 or 1."
    )

    # Check that magnification task returns expected keys and types
    magnification_item = sample_item["magnification"]
    expected_keys = {"image", "target", "target_mpp", "mpp_candidates"}
    assert set(magnification_item.keys()) == expected_keys, (
        f"Expected magnification keys {expected_keys}. Got: {set(magnification_item.keys())}"
    )
    assert isinstance(magnification_item["image"], torch.Tensor), (
        f"Magnification image should be a torch tensor. Got: {type(magnification_item['image'])}"
    )
    assert torch.all(
        (magnification_item["image"] >= 0) & (magnification_item["image"] <= 1)
    ), "Magnification image values should be in the range [0, 1]"
    assert magnification_item["image"].shape == (
        3,
        tile_dataset.image_size,
        tile_dataset.image_size,
    ), (
        f"Magnification image should have shape (3, {tile_dataset.image_size}, {tile_dataset.image_size}). Got: {magnification_item['image'].shape}"
    )
    assert isinstance(magnification_item["target"], torch.Tensor), (
        f"Magnification target should be a torch tensor. Got: {type(magnification_item['target'])}"
    )
    assert magnification_item["target"].shape == (
        len(magnification_item["mpp_candidates"]),
    ), (
        f"Magnification target should have shape {(len(magnification_item['mpp_candidates']),)}. Got: {magnification_item['target'].shape}"
    )
    assert (
        torch.all(
            (magnification_item["target"] == 0) | (magnification_item["target"] == 1)
        )
        and torch.sum(magnification_item["target"]) == 1
    ), (
        "Magnification target should be a one-hot tensor with exactly one element equal to 1 and the rest equal to 0."
    )
    assert isinstance(magnification_item["target_mpp"], torch.Tensor), (
        f"Magnification target_mpp should be a torch tensor. Got: {type(magnification_item['target_mpp'])}"
    )
    assert magnification_item["target_mpp"] in magnification_item["mpp_candidates"], (
        "Magnification target_mpp should be one of the candidates."
    )
    assert isinstance(magnification_item["mpp_candidates"], torch.Tensor), (
        f"Magnification mpp_candidates should be a torch tensor. Got: {type(magnification_item['mpp_candidates'])}"
    )

    # Check that hematoxylin task returns expected keys and types
    hematoxylin_item = sample_item["hematoxylin"]
    expected_keys = {"image", "target"}
    assert set(hematoxylin_item.keys()) == expected_keys, (
        f"Expected hematoxylin keys {expected_keys}. Got: {set(hematoxylin_item.keys())}"
    )
    assert isinstance(hematoxylin_item["image"], torch.Tensor), (
        f"Hematoxylin image should be a torch tensor. Got: {type(hematoxylin_item['image'])}"
    )
    assert hematoxylin_item["image"].shape == (
        3,
        tile_dataset.image_size,
        tile_dataset.image_size,
    ), (
        f"Hematoxylin image should have shape (3, {tile_dataset.image_size}, {tile_dataset.image_size}). Got: {hematoxylin_item['image'].shape}"
    )
    assert torch.all(
        (hematoxylin_item["image"] >= 0) & (hematoxylin_item["image"] <= 1)
    ), "Hematoxylin image values should be in the range [0, 1]"
    assert (
        isinstance(hematoxylin_item["target"], torch.Tensor)
        and hematoxylin_item["target"].shape[1:] == hematoxylin_item["image"].shape[1:]
    ), (
        f"Hematoxylin target should be a torch tensor with the same height and width as the image."
        f"Got: {type(hematoxylin_item['target'])} with shape {hematoxylin_item['target'].shape}"
    )
    assert torch.all(
        (hematoxylin_item["target"] >= 0) & (hematoxylin_item["target"] <= 1)
    ), "Hematoxylin target values should be in the range [0, 1]"

    # Check that jigmag task returns expected keys and types
    jigmag_item = sample_item["jigmag"]
    expected_keys = {"image", "target", "permutation", "mpps"}
    assert set(jigmag_item.keys()) == expected_keys, (
        f"Expected jigmag keys {expected_keys}. Got: {set(jigmag_item.keys())}"
    )
    assert isinstance(jigmag_item["image"], torch.Tensor), (
        f"JigMag image should be a torch tensor. Got: {type(jigmag_item['image'])}"
    )
    assert jigmag_item["image"].shape == (
        3,
        tile_dataset.image_size,
        tile_dataset.image_size,
    ), (
        f"JigMag image should have shape (3, {tile_dataset.image_size}, {tile_dataset.image_size}). Got: {jigmag_item['image'].shape}"
    )
    assert torch.all((jigmag_item["image"] >= 0) & (jigmag_item["image"] <= 1)), (
        "JigMag image values should be in the range [0, 1]"
    )
    assert isinstance(jigmag_item["target"], torch.Tensor), (
        f"JigMag target should be a torch tensor. Got: {type(jigmag_item['target'])}"
    )
    assert jigmag_item["target"].shape == (4 * 3 * 2 * 1,), (
        f"JigMag target should have shape {(4 * 3 * 2 * 1,)}. Got: {jigmag_item['target'].shape}"
    )
    assert (
        torch.all((jigmag_item["target"] == 0) | (jigmag_item["target"] == 1))
        and torch.sum(jigmag_item["target"]) == 1
    ), (
        "JigMag target should be a one-hot tensor with exactly one element equal to 1 and the rest equal to 0."
    )
    assert isinstance(jigmag_item["permutation"], torch.Tensor), (
        f"JigMag permutation should be a torch tensor. Got: {type(jigmag_item['permutation'])}"
    )
    assert jigmag_item["permutation"].shape == (4,), (
        f"JigMag permutation should have shape (4,). Got: {jigmag_item['permutation'].shape}"
    )
    assert set(jigmag_item["permutation"].tolist()) == {
        0,
        1,
        2,
        3,
    }, "JigMag permutation should be a rearrangement of (0, 1, 2, 3)."
    assert isinstance(jigmag_item["mpps"], torch.Tensor), (
        f"JigMag mpps should be a torch tensor. Got: {type(jigmag_item['mpps'])}"
    )

    # Optional: Plot the images for visual inspection (requires matplotlib)
    # import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # _, axes = plt.subplots(1, 6, figsize=(30, 5))
    # axes[0].imshow(magnification_item["image"].permute(1, 2, 0))
    # axes[0].set_title(f'Magnification Target MPP: {magnification_item["target_mpp"]}')
    # axes[1].imshow(hematoxylin_item["image"].permute(1, 2, 0))
    # axes[1].set_title("Hematoxylin Image")
    # axes[2].imshow(hematoxylin_item["target"].permute(1, 2, 0), cmap="gray")
    # axes[2].set_title("Hematoxylin Target")
    # axes[3].imshow(jigmag_item["image"].permute(1, 2, 0))
    # axes[3].set_title(f'JigMag Permutation: {jigmag_item["permutation"]}')
    # axes[4].imshow(tissue_item["image"].permute(1, 2, 0))
    # axes[4].set_title("Tissue Image")
    # axes[5].imshow(tissue_item["target"].permute(1, 2, 0), cmap="tab20")
    # axes[5].set_title("Tissue Target")
    # plt.suptitle(
    #     f"TCGATileDataset Sample Item Visualization: Submitter {sample_item['metadata']['submitter_id']} - Slide {sample_item['metadata']['slide_id']}"
    # )
    # plt.axis("off")
    # plt.show()

    print("[OK] TCGATileDataset test passed.")


def _test_TCGATileDataset():
    """Test that TCGATileDataset can be instantiated and returns expected keys."""
    print("Testing TCGATileDataset...")
    root_dir = "data/TCGA-BRCA-test"
    tasks = ["magnification", "hematoxylin", "jigmag", "tissue_segmentation"]

    dataset = TCGATileDataset(
        root_dir=root_dir,
        tasks=tasks,
        magnification_mpps=[0.25, 0.5, 1.0, 2.0],
        jigmag_mpps=[0.25, 0.5, 1.0, 2.0],
    )

    dataset.prepare_data()
    dataset.setup()

    # Check that train, val, and test dataloaders can be created and return expected keys
    for dataloader in [
        dataset.train_dataloader(),
        dataset.val_dataloader(),
        dataset.test_dataloader(),
    ]:
        sample_batch = next(iter(dataloader))
        expected_keys = {
            "metadata",
            "magnification",
            "hematoxylin",
            "jigmag",
            "tissue_segmentation",
        }
        assert set(sample_batch.keys()) == expected_keys, (
            f"Expected batch keys {expected_keys}. Got: {set(sample_batch.keys())}"
        )

    # Optional: Print a sample batch for visual inspection
    print("Sample batch from TCGATileDataset:")

    # print("Magnification task:")
    # print(
    #     f"{sample_batch['magnification']['image'].shape=}\n{sample_batch['magnification']['target'].shape=}"
    # )
    # print(
    #     f"{sample_batch['magnification']['target_mpp'].shape=}\n{sample_batch['magnification']['mpp_candidates'].shape=}"
    # )

    # print("Hematoxylin task:")
    # print(
    #     f"{sample_batch['hematoxylin']['image'].shape=}\n{sample_batch['hematoxylin']['target'].shape=}"
    # )
    # print(
    #     f"{sample_batch['hematoxylin']['target'].min()=}\n{sample_batch['hematoxylin']['target'].max()=}"
    # )

    # print("Tissue segmentation task:")
    # print(
    #     f"{sample_batch['jigmag']['image'].shape=}\n{sample_batch['jigmag']['target'].shape=}"
    # )
    # print(
    #     f"{sample_batch['jigmag']['permutation'].shape=}\n{sample_batch['jigmag']['mpps'].shape=}"
    # )

    # print("Tissue segmentation task:")
    # print(
    #     f"{sample_batch['tissue_segmentation']['image'].shape=}\n{sample_batch['tissue_segmentation']['target'].shape=}"
    # )

    print("[OK] TCGATileDataset test passed.")


def test_all_datasets_tcga():
    """Run all TCGA tile dataset tests."""
    print("Running all TCGA tile dataset tests...")
    _test_TileDataset()
    _test_TCGATileDataset()
    print("All TCGA tile dataset tests passed!")


if __name__ == "__main__":
    test_all_datasets_tcga()
