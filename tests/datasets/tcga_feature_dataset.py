"""Unit tests for the TCGAFeatureDataset module.

Uses the real ``data/TCGA-BRCA-test`` cohort and the cached features under
``data/TCGA-BRCA-test/features/resnet50-full`` produced by
``scripts/model_training/precompute_tile_features.py``.
"""

import os

import torch
import yaml

from augur.datasets.factory import get_dataset_from_config
from augur.datasets.tcga_feature_dataset import (
    TCGAFeatureDataset,
    _FeatureBagDataset,
)
from augur.datasets.cancer_subtyping import UNKNOWN_SUBTYPE_CLASS
from augur.utils.config import load_dataset_config

# Shared test fixtures: real test cohort + its precomputed feature cache.
_TEST_ROOT_DIR = "data/TCGA-BRCA-test"
_TEST_FEATURES_DIR = "data/TCGA-BRCA-test/features/resnet50-full"


def _test_FeatureBagDataset() -> None:
    """_FeatureBagDataset should yield a per-slide (K, D) bag and a subtyping label."""
    print("Testing _FeatureBagDataset ...")

    assert os.path.isdir(_TEST_FEATURES_DIR), (
        f"Test feature cache missing at {_TEST_FEATURES_DIR}. "
        "Run scripts/model_training/precompute_tile_features.py against the "
        "tcga-brca-test dataset base first."
    )

    portion_per_sample = 0.5

    datamodule = TCGAFeatureDataset(
        root_dir=_TEST_ROOT_DIR,
        features_dir=_TEST_FEATURES_DIR,
        main_task="subtyping",
        portion_per_sample=portion_per_sample,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        batch_size=2,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_dataset = datamodule.train_dataset
    assert isinstance(
        train_dataset, _FeatureBagDataset
    ), f"Expected _FeatureBagDataset, got: {type(train_dataset)}"
    assert len(train_dataset) > 0, "Train split should contain at least one slide."

    # Subtyping main-task label table is loaded into the datamodule.
    assert datamodule._main_label_names is not None  # pylint: disable=protected-access
    assert (
        datamodule._main_label_names[0]  # pylint: disable=protected-access
        == UNKNOWN_SUBTYPE_CLASS
    ), "Subtyping class index 0 must always be the unknown class."
    assert datamodule.num_main_labels >= 2, (
        "Need at least one real subtype + the unknown class. "
        f"Got: {datamodule.num_main_labels}."
    )

    # Cache discovery + sanity check populated the inferred encoder properties.
    assert (
        datamodule.encoder_name == "ResNetEncoder"
    ), f"Inferred encoder_name should match the cache. Got: {datamodule.encoder_name}"
    assert (
        datamodule.enc_dim == 2048
    ), f"ResNet50 cache should have enc_dim=2048. Got: {datamodule.enc_dim}"

    sample = train_dataset[0]
    expected_keys = {"image", "target", "metadata"}
    assert (
        set(sample.keys()) == expected_keys
    ), f"Expected sample keys {expected_keys}. Got: {set(sample.keys())}"

    image = sample["image"]
    assert isinstance(
        image, torch.Tensor
    ), f"image must be a tensor. Got: {type(image)}"
    assert (
        image.dtype == torch.float32
    ), f"Feature bags should be float32. Got: {image.dtype}"
    assert image.ndim == 2, (
        "Feature bag should be 2D (K, D) so the aggregator's pre-encoded "
        f"path is taken. Got ndim={image.ndim}, shape={image.shape}."
    )
    K, D = image.shape
    assert (
        D == datamodule.enc_dim
    ), f"Feature dim should match datamodule.enc_dim={datamodule.enc_dim}. Got: {D}"

    # K must obey: max(1, floor(K_total * portion)) capped at K_total.
    k_total = int(sample["metadata"]["k_total"])
    expected_K = max(1, int(k_total * portion_per_sample))
    expected_K = min(expected_K, k_total)
    assert K == expected_K, (
        f"Expected K = max(1, floor(K_total * portion)) = {expected_K} "
        f"for K_total={k_total}. Got K={K}."
    )
    assert torch.isfinite(image).all(), "Cached features contain non-finite values."

    target = sample["target"]
    assert isinstance(target, torch.Tensor)
    assert target.ndim == 0, (
        "Subtyping main-task target should be a scalar long. "
        f"Got shape: {target.shape}."
    )
    assert target.dtype == torch.long
    assert 0 <= int(target.item()) < datamodule.num_main_labels, (
        "Subtyping class index out of range. "
        f"Got: {int(target.item())} / {datamodule.num_main_labels}."
    )

    metadata = sample["metadata"]
    expected_metadata_keys = {
        "slide_id",
        "submitter_id",
        "task",
        "encoder_name",
        "enc_dim",
        "base_mpp",
        "tile_size",
        "image_size",
        "tile_centers",
        "selected_indices",
        "k_total",
    }
    assert set(metadata.keys()) == expected_metadata_keys, (
        f"Expected metadata keys {expected_metadata_keys}. "
        f"Got: {set(metadata.keys())}"
    )
    assert metadata["task"] == "subtyping"
    assert metadata["encoder_name"] == "ResNetEncoder"
    assert int(metadata["enc_dim"]) == D
    assert metadata["tile_centers"].shape == (K, 2)
    assert metadata["tile_centers"].dtype == torch.long
    assert metadata["selected_indices"].shape == (K,)
    assert metadata["selected_indices"].dtype == torch.long
    assert 0 <= int(metadata["selected_indices"].min()) < k_total
    assert int(metadata["selected_indices"].max()) < k_total
    # Indices are sampled without replacement.
    assert (
        metadata["selected_indices"].unique().numel() == K
    ), "selected_indices should be unique (sampling without replacement)."

    # Two calls on the same slide should sample different tiles (random sampling).
    sample_again = train_dataset[0]
    if k_total >= 2 * K:
        assert not torch.equal(
            sample["metadata"]["selected_indices"],
            sample_again["metadata"]["selected_indices"],
        ), "Random sampling should yield different indices across calls."

    print("[OK] _FeatureBagDataset test passed.")


def _test_TCGAFeatureDataset() -> None:
    """TCGAFeatureDataset should build dataloaders with the expected batch shape."""
    print("Testing TCGAFeatureDataset ...")

    config = load_dataset_config(
        "configs/dataset",
        base="tcga-brca-test",
        main_task="subtyping",
        subtasks=["sbs_regression"],
        flavor="feature",
        encoder="resnet50",
        pretext="full",
    )

    datamodule = get_dataset_from_config(config)
    assert isinstance(
        datamodule, TCGAFeatureDataset
    ), f"Expected TCGAFeatureDataset. Got: {type(datamodule)}"
    assert datamodule.main_task == "subtyping"
    assert datamodule.subtasks, (
        "Test config should configure at least one SBS subtask. "
        f"Got: {datamodule.subtasks}."
    )

    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule._main_label_names is not None  # pylint: disable=protected-access
    assert (
        datamodule._main_label_names[0]  # pylint: disable=protected-access
        == UNKNOWN_SUBTYPE_CLASS
    ), "Subtyping class index 0 must always be the unknown class."
    assert datamodule.num_main_labels >= 2

    # Cache validation should have populated inferred encoder/enc_dim.
    assert (
        datamodule.encoder_name == "ResNetEncoder"
    ), f"Inferred encoder_name should match the cache. Got: {datamodule.encoder_name}"
    assert (
        datamodule.enc_dim == 2048
    ), f"ResNet50 cache should have enc_dim=2048. Got: {datamodule.enc_dim}"

    main_submitter_labels = (
        datamodule._main_submitter_labels  # pylint: disable=protected-access
    )
    assert main_submitter_labels is not None, "Main labels should be loaded."

    # Each subtask exposes a vector-valued label table.
    for subtask in datamodule.subtasks:
        names = datamodule._subtask_label_names[  # pylint: disable=protected-access
            subtask
        ]
        assert (
            len(names) >= 1
        ), f"Subtask task '{subtask}' should expose at least one label column."
        assert datamodule.num_subtask_labels[subtask] == len(names)

    expected_batch_keys = {
        "image",
        "mask",
        "target",
        "metadata",
        *datamodule.subtasks,
    }

    for dataloader, batch_size in (
        (datamodule.train_dataloader(), datamodule.batch_size),
        (datamodule.val_dataloader(), datamodule.val_batch_size),
        (datamodule.test_dataloader(), datamodule.test_batch_size),
    ):
        batch = next(iter(dataloader))
        assert (
            set(batch.keys()) == expected_batch_keys
        ), f"Unexpected batch keys: {set(batch.keys())}"

        image = batch["image"]
        # After pad_bag_collate the bag becomes (B, K_max, D) — exactly the
        # shape the aggregator's _encode_bag takes via its `image.ndim == 3`
        # pre-encoded short-circuit.
        assert image.ndim == 3, (
            "Collated feature bag should be 3D (B, K_max, D) to match the "
            f"aggregator's pre-encoded path. Got: {image.shape}"
        )
        B, K, D = image.shape
        assert D == datamodule.enc_dim, (
            f"Feature dim should match datamodule.enc_dim={datamodule.enc_dim}. "
            f"Got: {D}"
        )
        assert (
            B <= batch_size
        ), f"Batch dim should be <= batch_size={batch_size}. Got: {B}."
        assert K >= 1, "Each batch should contain at least one tile per bag."
        assert (
            image.dtype == torch.float32
        ), f"Feature batches should be float32. Got: {image.dtype}"
        assert torch.isfinite(image).all(), "Batch contains non-finite feature values."

        mask = batch["mask"]
        assert mask.shape == (B, K), f"Unexpected mask shape: {mask.shape}"
        assert mask.dtype == torch.bool
        assert mask.any(
            dim=-1
        ).all(), "Every bag should have at least one valid tile after padding."

        target = batch["target"]
        assert target.shape == (
            B,
        ), f"Subtyping target should be a (B,) long tensor. Got: {target.shape}."
        assert target.dtype == torch.long
        assert torch.all(
            (target >= 0) & (target < datamodule.num_main_labels)
        ), "Subtyping class indices must lie in [0, num_main_labels)."

        for subtask in datamodule.subtasks:
            subtask_target = batch[subtask]["target"]
            assert isinstance(
                subtask_target, torch.Tensor
            ), f"Subtask '{subtask}' target must be a tensor."
            assert subtask_target.shape == (
                B,
                datamodule.num_subtask_labels[subtask],
            ), (
                f"Subtask '{subtask}' target should have shape "
                f"({B}, {datamodule.num_subtask_labels[subtask]}). "
                f"Got: {subtask_target.shape}."
            )
            assert subtask_target.dtype == torch.float32
            assert torch.isfinite(
                subtask_target
            ).all(), f"Subtask '{subtask}' target contains non-finite values."

    print("[OK] TCGAFeatureDataset test passed.")


def _test_encoder_mismatch_rejected() -> None:
    """Declaring a wrong encoder/enc_dim should fail fast at setup."""
    print("Testing TCGAFeatureDataset cache-encoder validation ...")

    # Wrong enc_dim — cache is 2048, declare 1536 (gigapath dim) and expect failure.
    datamodule = TCGAFeatureDataset(
        root_dir=_TEST_ROOT_DIR,
        features_dir=_TEST_FEATURES_DIR,
        main_task="subtyping",
        portion_per_sample=0.5,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        batch_size=2,
        enc_dim=1536,
    )
    datamodule.prepare_data()
    try:
        datamodule.setup(stage="fit")
    except ValueError as exc:
        assert "enc_dim" in str(
            exc
        ), f"Expected enc_dim mismatch error message. Got: {exc}"
    else:
        raise AssertionError("Expected ValueError for declared enc_dim mismatch.")

    # Wrong encoder name — cache is ResNetEncoder, declare ViTEncoder.
    datamodule = TCGAFeatureDataset(
        root_dir=_TEST_ROOT_DIR,
        features_dir=_TEST_FEATURES_DIR,
        main_task="subtyping",
        portion_per_sample=0.5,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        batch_size=2,
        expected_encoder_name="ViTEncoder",
    )
    datamodule.prepare_data()
    try:
        datamodule.setup(stage="fit")
    except ValueError as exc:
        assert "encoder_name" in str(
            exc
        ), f"Expected encoder_name mismatch error message. Got: {exc}"
    else:
        raise AssertionError("Expected ValueError for declared encoder_name mismatch.")

    print("[OK] TCGAFeatureDataset cache-encoder validation test passed.")


def test_all_datasets_feature() -> None:
    """Run all TCGAFeatureDataset tests."""
    print("Running all TCGA feature dataset tests ...")
    _test_FeatureBagDataset()
    _test_TCGAFeatureDataset()
    _test_encoder_mismatch_rejected()
    print("All TCGA feature dataset tests passed!")


if __name__ == "__main__":
    test_all_datasets_feature()
