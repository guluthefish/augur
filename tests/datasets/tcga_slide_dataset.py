"""Unit tests for the TCGASlideDataset module."""

import torch
import yaml

from augur.datasets.cancer_subtyping import UNKNOWN_SUBTYPE_CLASS
from augur.datasets.factory import get_dataset_from_config
from augur.datasets.tcga_slide_dataset import (
    TCGASlideDataset,
    _SlideDataset,
)


def _test_SlideDataset() -> None:
    """Test that _SlideDataset yields a per-slide tile stack and a subtyping label."""
    print("Testing _SlideDataset ...")

    root_dir = "data/TCGA-BRCA-test"
    portion_per_sample = 0.5

    datamodule = TCGASlideDataset(
        root_dir=root_dir,
        main_task="subtyping",
        portion_per_sample=portion_per_sample,
        stride=512,
        tile_size=512,
        image_size=128,
        base_mpp=0.25,
        min_tissue_fraction=0.25,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        batch_size=2,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_dataset = datamodule.train_dataset
    assert isinstance(train_dataset, _SlideDataset), (
        f"Expected _SlideDataset, got: {type(train_dataset)}"
    )
    assert len(train_dataset) > 0, "Train split should contain at least one slide."

    # Subtyping main-task label table is loaded into the datamodule.
    assert datamodule.main_label_names[0] == UNKNOWN_SUBTYPE_CLASS, (
        "Subtyping class index 0 must always be the unknown class."
    )
    assert datamodule.num_main_labels >= 2, (
        "Need at least one real subtype + the unknown class. "
        f"Got: {datamodule.num_main_labels}."
    )

    sample = train_dataset[0]
    expected_keys = {"image", "target", "metadata"}
    assert set(sample.keys()) == expected_keys, (
        f"Expected sample keys {expected_keys}. Got: {set(sample.keys())}"
    )

    image = sample["image"]
    assert isinstance(image, torch.Tensor), (
        f"image must be a tensor. Got: {type(image)}"
    )
    K = image.shape[0]
    candidate_count = len(
        train_dataset.centers_by_slide_id[train_dataset.slide_records[0].slide_id]
    )
    expected_K = max(1, int(candidate_count * portion_per_sample))
    expected_K = min(expected_K, candidate_count)
    assert K == expected_K, (
        f"Expected K = max(1, floor(T * portion)) = {expected_K} "
        f"for T={candidate_count} candidates. Got K={K}."
    )
    assert image.shape == (
        K,
        3,
        datamodule.image_size,
        datamodule.image_size,
    ), f"image should have shape (K, 3, H, W). Got: {image.shape}"
    assert torch.all((image >= 0) & (image <= 1)), (
        "image values should be in the range [0, 1]."
    )

    target = sample["target"]
    assert isinstance(target, torch.Tensor), (
        f"target must be a tensor. Got: {type(target)}"
    )
    assert target.ndim == 0, (
        "Subtyping main-task target should be a scalar long. "
        f"Got shape: {target.shape}."
    )
    assert target.dtype == torch.long, (
        f"Subtyping target should be long. Got: {target.dtype}"
    )
    assert 0 <= int(target.item()) < datamodule.num_main_labels, (
        "Subtyping class index out of range. "
        f"Got: {int(target.item())} / {datamodule.num_main_labels}."
    )

    metadata = sample["metadata"]
    expected_metadata_keys = {
        "slide_id",
        "submitter_id",
        "base_mpp",
        "task",
        "tile_centers",
        "tile_xy",
        "tile_level",
        "tile_size",
    }
    assert set(metadata.keys()) == expected_metadata_keys, (
        f"Expected metadata keys {expected_metadata_keys}. Got: {set(metadata.keys())}"
    )
    assert metadata["task"] == "subtyping"
    assert metadata["tile_centers"].shape == (K, 2)
    assert metadata["tile_xy"].shape == (K, 2)
    assert metadata["tile_level"].shape == (K,)
    assert metadata["tile_size"].shape == (K,)

    # Two calls on the same slide should sample different tiles (random sampling).
    sample_again = train_dataset[0]
    if candidate_count >= 2 * K:
        assert not torch.equal(
            sample["metadata"]["tile_centers"], sample_again["metadata"]["tile_centers"]
        ), "Random sampling should yield different tile centers across calls."

    datamodule.teardown()
    print("[OK] _SlideDataset test passed.")


def _test_TCGASlideDataset() -> None:
    """Test that TCGASlideDataset builds dataloaders with the expected batch shape."""
    print("Testing TCGASlideDataset ...")

    config_path = "configs/slide_dataset-TCGA-BRCA-test.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    datamodule = get_dataset_from_config(config)
    assert isinstance(datamodule, TCGASlideDataset), (
        f"Expected TCGASlideDataset. Got: {type(datamodule)}"
    )
    assert datamodule.main_task == "subtyping"
    assert datamodule.pretext_tasks, (
        "Test config should configure at least one SBS pretext task. "
        f"Got: {datamodule.pretext_tasks}."
    )

    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.main_label_names[0] == UNKNOWN_SUBTYPE_CLASS, (
        "Subtyping class index 0 must always be the unknown class."
    )
    assert datamodule.num_main_labels >= 2

    main_submitter_labels = datamodule._main_submitter_labels
    assert main_submitter_labels is not None, "Main labels should be loaded."
    assert "TCGA-A1-A0SK" in main_submitter_labels, (
        "Expected the test submitter TCGA-A1-A0SK in the main label table."
    )

    # Each pretext SBS task exposes a vector-valued label table.
    for pretext_task in datamodule.pretext_tasks:
        names = datamodule.pretext_label_names[pretext_task]
        assert len(names) >= 1, (
            f"Pretext task '{pretext_task}' should expose at least one label column."
        )
        assert datamodule.num_pretext_labels[pretext_task] == len(names)
        assert "TCGA-A1-A0SK" in datamodule._pretext_submitter_labels[pretext_task], (
            f"Pretext '{pretext_task}' should cover submitter TCGA-A1-A0SK."
        )

    expected_batch_keys = {
        "image",
        "mask",
        "target",
        "metadata",
        *datamodule.pretext_tasks,
    }

    for dataloader, batch_size in (
        (datamodule.train_dataloader(), datamodule.batch_size),
        (datamodule.val_dataloader(), datamodule.val_batch_size),
        (datamodule.test_dataloader(), datamodule.test_batch_size),
    ):
        batch = next(iter(dataloader))
        assert set(batch.keys()) == expected_batch_keys, (
            f"Unexpected batch keys: {set(batch.keys())}"
        )

        image = batch["image"]
        B, K = image.shape[0], image.shape[1]
        assert image.shape == (
            B,
            K,
            3,
            datamodule.image_size,
            datamodule.image_size,
        ), f"Unexpected image batch shape: {image.shape}"
        assert B <= batch_size, (
            f"Batch dim should be <= batch_size={batch_size}. Got: {B}."
        )
        assert K >= 1, "Each batch should contain at least one tile per bag."

        mask = batch["mask"]
        assert mask.shape == (B, K), f"Unexpected mask shape: {mask.shape}"
        assert mask.dtype == torch.bool
        assert mask.any(dim=-1).all(), "Every bag should have at least one valid tile."

        target = batch["target"]
        assert target.shape == (B,), (
            f"Subtyping target should be a (B,) long tensor. Got: {target.shape}."
        )
        assert target.dtype == torch.long
        assert torch.all((target >= 0) & (target < datamodule.num_main_labels)), (
            "Subtyping class indices must lie in [0, num_main_labels)."
        )

        for pretext_task in datamodule.pretext_tasks:
            pretext_target = batch[pretext_task]["target"]
            assert isinstance(pretext_target, torch.Tensor), (
                f"Pretext '{pretext_task}' target must be a tensor."
            )
            assert pretext_target.shape == (
                B,
                datamodule.num_pretext_labels[pretext_task],
            ), (
                f"Pretext '{pretext_task}' target should have shape "
                f"({B}, {datamodule.num_pretext_labels[pretext_task]}). "
                f"Got: {pretext_target.shape}."
            )
            assert pretext_target.dtype == torch.float32, (
                f"Pretext '{pretext_task}' target should be float32."
            )
            assert torch.isfinite(pretext_target).all(), (
                f"Pretext '{pretext_task}' target contains non-finite values."
            )

    datamodule.teardown()
    print("[OK] TCGASlideDataset test passed.")


def test_all_datasets_slide() -> None:
    """Run all TCGASlideDataset tests."""
    print("Running all  TCGA slide dataset tests ...")
    _test_SlideDataset()
    _test_TCGASlideDataset()
    print("All TCGA slide dataset tests passed!")


if __name__ == "__main__":
    test_all_datasets_slide()
