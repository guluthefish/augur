"""Unit tests for the DualCLAM slide-level MIL model."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from augur.datasets.tcga_slide_dataset import TCGASlideDataset
from augur.models.model_abc import ModelABC
from augur.models.slide_level.attention import Attention, GatedAttention
from augur.models.slide_level.dual_clam import (
    DualCLAM,
    DualCLAM_MB,
    DualCLAM_SB,
)


def _assert_raises(
    expected_exception: type[BaseException],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Assert that a callable raises the expected exception type."""
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, got {type(exc).__name__}."
        ) from exc
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised.")


class _TinyTileEncoder(ModelABC):
    """Fast deterministic encoder shared with the EmbeddingMIL test suite."""

    feature_dim = 6

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def from_config(config: dict[str, Any]) -> _TinyTileEncoder:
        del config
        return _TinyTileEncoder()

    def forward(  # pylint: disable=arguments-differ
        self, image: torch.Tensor
    ) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                f"_TinyTileEncoder expected image shape (N, 3, H, W). Got: {image.shape}"
            )
        flat = image.float().flatten(start_dim=2)
        channel_means = flat.mean(dim=-1)
        channel_stds = flat.std(dim=-1, unbiased=False)
        return torch.cat((channel_means, channel_stds), dim=1)

    def model_step(
        self,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        del batch, batch_idx, stage
        raise NotImplementedError("_TinyTileEncoder is only used for inference tests.")


def _load_real_slide_batch(
    *, pretext_tasks: list[str] | None = None
) -> tuple[TCGASlideDataset, dict[str, Any]]:
    """Load one small real TCGA slide batch for DualCLAM tests."""
    datamodule = TCGASlideDataset(
        root_dir="data/TCGA-BRCA-test",
        main_task="subtyping",
        pretext_tasks=pretext_tasks,
        portion_per_sample=0.1,
        stride=512,
        tile_size=512,
        image_size=64,
        base_mpp=0.25,
        min_tissue_fraction=0.25,
        thumbnail_max_size=1024,
        white_threshold=0.8,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        max_slides=2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        shuffle_train=False,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="predict")
    batch = next(iter(datamodule.predict_dataloader()))
    return datamodule, batch


def _build_dual_clam(
    *,
    pretext_tasks: list[str] | None,
    output_dims: dict[str, int],
    multi_branch: bool = False,
    gated: bool = True,
    inst_weight: float = 0.0,
    out_of_class: bool = False,
) -> DualCLAM:
    """Construct a tiny DualCLAM Lightning model for real-data tests."""
    encoder = _TinyTileEncoder()
    return DualCLAM(
        encoder=encoder,
        main_task="subtyping",
        pretext_tasks=pretext_tasks,
        enc_dim=encoder.feature_dim,
        hidden_dims=[8],
        output_dims=output_dims,
        dropout=0.0,
        attn_kwargs={
            "gated": gated,
            "hidden_dim": 4,
            "dropout": 0.0,
            "multi_branch": multi_branch,
        },
        cluster_kwargs={
            "k_sample": 2,
            "inst_weight": inst_weight,
            "out_of_class": out_of_class,
        },
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )


def _test_init() -> None:
    """DualCLAM should construct SB/MB backbones with coherent shapes."""
    print("Testing DualCLAM initialization...")

    num_subtype_classes = 5

    # Single-branch, no pretext.
    model_sb = _build_dual_clam(
        pretext_tasks=None,
        output_dims={"subtyping": num_subtype_classes},
    )
    assert isinstance(
        model_sb, ModelABC
    ), f"DualCLAM must inherit ModelABC. Got: {type(model_sb)}."
    assert isinstance(
        model_sb.backbone, DualCLAM_SB
    ), f"Expected DualCLAM_SB backbone. Got: {type(model_sb.backbone)}."
    assert (
        model_sb.backbone.num_heads == 1
    ), f"SB must use num_heads=1. Got: {model_sb.backbone.num_heads}."
    assert model_sb.backbone.num_main_branches == 1
    assert (
        not model_sb.pretext_tasks
    ), f"No-pretext model must have empty pretext_tasks. Got: {model_sb.pretext_tasks}."
    assert isinstance(
        model_sb.backbone.attention_net, GatedAttention
    ), "Default attention should be gated."
    assert set(model_sb.backbone.heads.keys()) == {"subtyping"}, (
        "No-pretext model should only have the main head. "
        f"Got: {set(model_sb.backbone.heads.keys())}."
    )
    # Instance classifiers are anchored on the main classification task.
    assert isinstance(model_sb.backbone.instance_classifiers, nn.ModuleList)
    assert len(model_sb.backbone.instance_classifiers) == num_subtype_classes, (
        "SB should register one instance classifier per main-task class. "
        f"Got: {len(model_sb.backbone.instance_classifiers)}."
    )

    # Single-branch with pretext: adds a pretext head but keeps one shared branch.
    model_sb_pretext = _build_dual_clam(
        pretext_tasks=["sbs_regression"],
        output_dims={"subtyping": num_subtype_classes, "sbs_regression": 96},
    )
    assert isinstance(
        model_sb_pretext.backbone, DualCLAM_SB
    ), "multi_branch=False should keep the SB backbone even with pretext tasks."
    assert model_sb_pretext.backbone.num_heads == 1
    assert set(model_sb_pretext.backbone.heads.keys()) == {
        "subtyping",
        "sbs_regression",
    }
    # In SB, every task slices branches [0, 1).
    assert model_sb_pretext.backbone.branch_layout == {
        "subtyping": (0, 1),
        "sbs_regression": (0, 1),
    }

    # Multi-branch: num_heads = num_main_classes + len(pretext_tasks).
    model_mb = _build_dual_clam(
        pretext_tasks=["sbs_regression"],
        output_dims={"subtyping": num_subtype_classes, "sbs_regression": 96},
        multi_branch=True,
        gated=False,
    )
    assert isinstance(
        model_mb.backbone, DualCLAM_MB
    ), f"Expected DualCLAM_MB backbone. Got: {type(model_mb.backbone)}."
    expected_num_heads = num_subtype_classes + 1
    assert model_mb.backbone.num_heads == expected_num_heads, (
        f"MB num_heads must equal num_subtype_classes + len(pretext_tasks) = "
        f"{expected_num_heads}. Got: {model_mb.backbone.num_heads}."
    )
    assert model_mb.backbone.num_main_branches == num_subtype_classes
    assert isinstance(
        model_mb.backbone.attention_net, Attention
    ), "Non-gated attention should use the plain Attention class."

    # Branch layout puts the main task on the first num_main_branches and
    # appends one branch per pretext task.
    assert model_mb.backbone.branch_layout == {
        "subtyping": (0, num_subtype_classes),
        "sbs_regression": (num_subtype_classes, num_subtype_classes + 1),
    }

    # Subtyping head consumes the concatenated main-task branches.
    subtyping_head = model_mb.backbone.heads["subtyping"]
    assert isinstance(subtyping_head, nn.Linear)
    assert (
        subtyping_head.in_features
        == model_mb.backbone.projection_dim * num_subtype_classes
    )
    assert subtyping_head.out_features == num_subtype_classes

    # SBS pretext head consumes its single branch.
    sbs_head = model_mb.backbone.heads["sbs_regression"]
    assert isinstance(sbs_head, nn.Linear)
    assert sbs_head.in_features == model_mb.backbone.projection_dim
    assert sbs_head.out_features == 96

    # Optimizer plumbing should flow through ModelABC.
    assert isinstance(model_sb.configure_optimizers(), torch.optim.Adam)

    print("[OK] DualCLAM initialization test passed.")


def _test_real_data_no_pretext() -> None:
    """Without pretext tasks, DualCLAM reduces to plain CLAM on subtyping."""
    print("Testing DualCLAM without pretext tasks on real slide data...")

    datamodule, batch = _load_real_slide_batch(pretext_tasks=None)
    try:
        assert set(batch.keys()) == {
            "image",
            "mask",
            "target",
            "metadata",
        }, f"Unexpected batch keys without pretext: {set(batch.keys())}."

        num_classes = datamodule.num_main_labels
        model = _build_dual_clam(
            pretext_tasks=None,
            output_dims={"subtyping": num_classes},
        )
        model.eval()

        with torch.no_grad():
            outputs = model(batch)
            step_output = model.model_step(batch, batch_idx=0, stage="train")
        assert isinstance(step_output, tuple)
        loss, metrics = step_output
        assert isinstance(metrics, dict)

        prediction = outputs["subtyping"]
        assert prediction.shape == (batch["image"].shape[0], num_classes), (
            f"Subtyping prediction should have shape (B, {num_classes}). "
            f"Got: {prediction.shape}."
        )
        assert torch.isfinite(prediction).all()

        attention_weights = outputs["_attention_weights"]
        assert attention_weights is not None
        assert attention_weights.shape == (
            batch["image"].shape[0],
            1,
            batch["image"].shape[1],
        ), f"SB attention shape mismatch. Got: {attention_weights.shape}."
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(batch["image"].shape[0], 1),
            atol=1e-6,
        ), "Attention weights should normalize to 1 per bag."

        assert loss.ndim == 0, f"Expected scalar loss. Got shape: {loss.shape}."
        assert torch.isfinite(loss), "Loss should be finite."
        assert set(metrics.keys()) == {
            "subtyping_loss"
        }, f"Expected only subtyping_loss. Got: {set(metrics.keys())}."
    finally:
        datamodule.teardown()

    print("[OK] DualCLAM no-pretext real-data test passed.")


def _test_real_data_with_pretext_sb() -> None:
    """DualCLAM-SB with an SBS pretext should return both heads and both losses."""
    print("Testing DualCLAM (SB) with pretext sbs_regression on real slide data...")

    datamodule, batch = _load_real_slide_batch(pretext_tasks=["sbs_regression"])
    try:
        expected_keys = {"image", "mask", "target", "sbs_regression", "metadata"}
        assert (
            set(batch.keys()) == expected_keys
        ), f"Expected batch keys {expected_keys}. Got: {set(batch.keys())}."

        num_classes = datamodule.num_main_labels
        sbs_dim = datamodule.num_pretext_labels["sbs_regression"]
        model = _build_dual_clam(
            pretext_tasks=["sbs_regression"],
            output_dims={"subtyping": num_classes, "sbs_regression": sbs_dim},
        )
        model.eval()

        with torch.no_grad():
            outputs = model(batch)
            step_output = model.model_step(batch, batch_idx=0, stage="train")
        assert isinstance(step_output, tuple)
        loss, metrics = step_output
        assert isinstance(metrics, dict)

        # Main subtyping logits + SBS regression vector.
        assert outputs["subtyping"].shape == (batch["image"].shape[0], num_classes)
        sbs_pred = outputs["sbs_regression"]
        assert sbs_pred.shape == (
            batch["image"].shape[0],
            sbs_dim,
        ), f"sbs_regression shape mismatch. Got: {sbs_pred.shape}."

        # SB attention: single shared branch.
        attention_weights = outputs["_attention_weights"]
        assert attention_weights is not None
        assert attention_weights.shape == (
            batch["image"].shape[0],
            1,
            batch["image"].shape[1],
        )

        assert loss.ndim == 0 and torch.isfinite(loss)
        assert set(metrics.keys()) == {
            "subtyping_loss",
            "sbs_regression_loss",
        }, f"Expected main + pretext losses. Got: {set(metrics.keys())}."
    finally:
        datamodule.teardown()

    print("[OK] DualCLAM SB-with-pretext real-data test passed.")


def _test_real_data_with_pretext_mb() -> None:
    """DualCLAM-MB has one branch per subtype class plus one branch per pretext task."""
    print("Testing DualCLAM (MB) with pretext sbs_regression on real slide data...")

    datamodule, batch = _load_real_slide_batch(pretext_tasks=["sbs_regression"])
    try:
        num_classes = datamodule.num_main_labels
        sbs_dim = datamodule.num_pretext_labels["sbs_regression"]
        model = _build_dual_clam(
            pretext_tasks=["sbs_regression"],
            output_dims={"subtyping": num_classes, "sbs_regression": sbs_dim},
            multi_branch=True,
        )
        assert isinstance(
            model.backbone, DualCLAM_MB
        ), f"multi_branch=True should select DualCLAM_MB. Got: {type(model.backbone)}."
        expected_num_heads = num_classes + 1
        assert model.backbone.num_heads == expected_num_heads
        assert model.backbone.num_main_branches == num_classes

        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            step_output = model.model_step(batch, batch_idx=0, stage="train")
        assert isinstance(step_output, tuple)
        loss, metrics = step_output
        assert isinstance(metrics, dict)

        attention_weights = outputs["_attention_weights"]
        assert attention_weights is not None
        assert attention_weights.shape == (
            batch["image"].shape[0],
            expected_num_heads,
            batch["image"].shape[1],
        ), f"MB attention shape mismatch. Got: {attention_weights.shape}."
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(batch["image"].shape[0], expected_num_heads),
            atol=1e-6,
        ), "Each MB attention branch should normalize to 1."

        assert outputs["subtyping"].shape == (batch["image"].shape[0], num_classes)
        assert outputs["sbs_regression"].shape == (batch["image"].shape[0], sbs_dim)
        assert torch.isfinite(loss)
        assert set(metrics.keys()) == {"subtyping_loss", "sbs_regression_loss"}
    finally:
        datamodule.teardown()

    print("[OK] DualCLAM MB-with-pretext real-data test passed.")


def _test_instance_clustering_loss() -> None:
    """inst_weight>0 should add a subtyping_instance_loss metric."""
    print("Testing DualCLAM instance clustering loss...")

    datamodule, batch = _load_real_slide_batch(pretext_tasks=["sbs_regression"])
    try:
        num_classes = datamodule.num_main_labels
        sbs_dim = datamodule.num_pretext_labels["sbs_regression"]
        model = _build_dual_clam(
            pretext_tasks=["sbs_regression"],
            output_dims={"subtyping": num_classes, "sbs_regression": sbs_dim},
            multi_branch=True,
            inst_weight=0.5,
            out_of_class=True,
        )
        # Instance classifiers are anchored on the main subtyping task.
        inst_classifiers = model.backbone.instance_classifiers
        assert isinstance(inst_classifiers, nn.ModuleList)
        assert len(inst_classifiers) == num_classes, (
            "Should register one binary classifier per subtype class. "
            f"Got: {len(inst_classifiers)}."
        )
        unknown_class_index = model.backbone.unknown_class_index
        for c, cls in enumerate(inst_classifiers):
            if unknown_class_index is not None and c == unknown_class_index:
                assert isinstance(cls, nn.Identity)
            else:
                assert isinstance(cls, nn.Linear)
                assert cls.out_features == 2

        model.eval()
        with torch.no_grad():
            step_output = model.model_step(batch, batch_idx=0, stage="train")
        assert isinstance(step_output, tuple)
        loss, metrics = step_output
        assert isinstance(metrics, dict)

        assert torch.isfinite(loss)
        assert (
            "subtyping_instance_loss" in metrics
        ), f"Expected subtyping_instance_loss in metrics. Got: {set(metrics.keys())}."
        assert torch.isfinite(metrics["subtyping_instance_loss"])
    finally:
        datamodule.teardown()

    print("[OK] DualCLAM instance clustering loss test passed.")


def _test_error_handling() -> None:
    """Invalid configurations should raise explicit exceptions."""
    print("Testing DualCLAM error handling...")

    # Unsupported main task.
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="regression",
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"regression": 5},
    )

    # Unsupported pretext task.
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        pretext_tasks=["unsupported_pretext"],
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"subtyping": 5, "unsupported_pretext": 3},
    )

    # main_task missing from output_dims.
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"sbs_regression": 96},
    )

    # main_task with output_dim < 2 (classification needs >= 2 classes).
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"subtyping": 1},
    )

    # Invalid enc_dim / hidden_dims / dropout.
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        enc_dim=0,
        hidden_dims=[8],
        output_dims={"subtyping": 5},
    )
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        enc_dim=4,
        hidden_dims=[],
        output_dims={"subtyping": 5},
    )
    _assert_raises(
        ValueError,
        DualCLAM,
        main_task="subtyping",
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"subtyping": 5},
        dropout=1.0,
    )

    # DualCLAM_MB requires output_dims[main_task] >= 2.
    _assert_raises(
        ValueError,
        DualCLAM_MB,
        enc_dim=4,
        hidden_dims=[8],
        output_dims={"subtyping": 1},
    )

    # model_step without a main-task target.
    model = _build_dual_clam(
        pretext_tasks=None,
        output_dims={"subtyping": 5},
    )
    _assert_raises(
        KeyError,
        model.model_step,
        {"image": torch.rand(1, 2, 3, 8, 8)},
        0,
        "train",
    )

    print("[OK] DualCLAM error handling test passed.")


def _test_from_config() -> None:
    """DualCLAM.from_config should parse a full config into a working model."""
    print("Testing DualCLAM.from_config()...")

    # Case 1: Minimal config — encoder-less, no pretext, default main_task.
    minimal_model = DualCLAM.from_config(
        {
            "enc_dim": 6,
            "hidden_dims": [8],
            "output_dims": {"subtyping": 4},
        }
    )
    assert isinstance(minimal_model, DualCLAM)
    assert not minimal_model.backbone.has_encoder, "No tile_model implies encoder-less."
    assert isinstance(
        minimal_model.backbone, DualCLAM_SB
    ), "Default attn_kwargs.multi_branch=False should select the SB backbone."
    assert minimal_model.main_task == "subtyping"
    assert not minimal_model.pretext_tasks
    assert (
        not minimal_model.task_kwargs
    ), f"Omitting task_kwargs should leave it empty. Got: {minimal_model.task_kwargs}."

    minimal_model.eval()
    bag = torch.randn(2, 3, 6)
    with torch.no_grad():
        minimal_out = minimal_model(bag)
    assert minimal_out["subtyping"].shape == (2, 4)
    assert minimal_out["_attention_weights"].shape == (2, 1, 3)

    # Case 2: Full config — pretext, task weights/kwargs, MB, optimizer, scheduler.
    full_config = {
        "main_task": "subtyping",
        "pretext_tasks": ["sbs_regression"],
        "task_weights": {"subtyping": 2.0, "sbs_regression": 1.0},
        "task_kwargs": {"subtyping": {"unknown_class_index": 0}},
        "enc_dim": 6,
        "hidden_dims": [8],
        "output_dims": {"subtyping": 4, "sbs_regression": 96},
        "dropout": 0.1,
        "attn_kwargs": {
            "multi_branch": True,
            "gated": True,
            "hidden_dim": 4,
            "dropout": 0.0,
        },
        "cluster_kwargs": {
            "k_sample": 2,
            "inst_weight": 0.5,
            "out_of_class": True,
        },
        "optimizer": {
            "name": "AdamW",
            "params": {"lr": 5e-4, "weight_decay": 0.01},
        },
        "lr_scheduler": {
            "name": "StepLR",
            "params": {"step_size": 3, "gamma": 0.5},
            "config": {"interval": "epoch", "frequency": 1},
        },
    }
    full_model = DualCLAM.from_config(full_config)
    assert isinstance(
        full_model.backbone, DualCLAM_MB
    ), "multi_branch=True should select the MB backbone."
    expected_num_heads = 4 + 1  # output_dims[subtyping] + len(pretext_tasks)
    assert full_model.backbone.num_heads == expected_num_heads, (
        f"MB num_heads must equal num_main_branches + len(pretext_tasks) = "
        f"{expected_num_heads}. Got: {full_model.backbone.num_heads}."
    )
    assert full_model.backbone.num_main_branches == 4
    assert isinstance(full_model.backbone.attention_net, GatedAttention)
    assert full_model.pretext_tasks == ["sbs_regression"]

    # Task weights normalize to sum=1 while preserving the ratio.
    assert full_model.task_weights == {
        "subtyping": 2.0 / 3.0,
        "sbs_regression": 1.0 / 3.0,
    }, f"Expected normalized task_weights. Got: {full_model.task_weights}."

    # task_kwargs stored unchanged.
    assert full_model.task_kwargs == {
        "subtyping": {"unknown_class_index": 0}
    }, f"Expected task_kwargs to be stored. Got: {full_model.task_kwargs}."

    # CLAM clustering settings are parsed into flat attributes.
    assert full_model.inst_weight == 0.5
    assert full_model.out_of_class is True
    assert full_model.k_sample == 2

    # Optimizer and scheduler plumbing.
    assert (
        full_model.optimizer_factory is torch.optim.AdamW
    ), f"Expected AdamW optimizer factory. Got: {full_model.optimizer_factory}."
    assert full_model.optimizer_kwargs == {"lr": 5e-4, "weight_decay": 0.01}
    assert (
        full_model.lr_scheduler_factory is torch.optim.lr_scheduler.StepLR
    ), f"Expected StepLR scheduler factory. Got: {full_model.lr_scheduler_factory}."
    assert full_model.lr_scheduler_kwargs == {"step_size": 3, "gamma": 0.5}
    assert full_model.lr_scheduler_config == {"interval": "epoch", "frequency": 1}

    optimizers = full_model.configure_optimizers()
    assert isinstance(optimizers, dict)
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    assert isinstance(
        optimizers["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR
    )

    # Instance classifiers anchored on the main subtyping task.
    inst_classifiers = full_model.backbone.instance_classifiers
    assert isinstance(inst_classifiers, nn.ModuleList)
    assert len(inst_classifiers) == 4

    # End-to-end forward on a pre-computed bag.
    full_model.eval()
    with torch.no_grad():
        full_out = full_model(bag)
    assert full_out["subtyping"].shape == (2, 4)
    assert full_out["sbs_regression"].shape == (2, 96)
    assert full_out["_attention_weights"].shape == (2, expected_num_heads, 3)

    # Case 3: invalid configs should fail at the boundary.
    _assert_raises(TypeError, DualCLAM.from_config, "not a dict")
    _assert_raises(
        AssertionError,
        DualCLAM.from_config,
        {"hidden_dims": [8], "output_dims": {"subtyping": 4}},
    )
    _assert_raises(
        AssertionError,
        DualCLAM.from_config,
        {"enc_dim": 6, "output_dims": {"subtyping": 4}},
    )
    _assert_raises(
        AssertionError,
        DualCLAM.from_config,
        {"enc_dim": 6, "hidden_dims": [8]},
    )
    _assert_raises(
        TypeError,
        DualCLAM.from_config,
        {
            "enc_dim": 6,
            "hidden_dims": [8],
            "output_dims": {"subtyping": 4},
            "task_kwargs": "not a dict",
        },
    )

    print("[OK] DualCLAM.from_config() test passed.")


def test_DualCLAM() -> None:
    """Run all slide-level DualCLAM unit tests."""
    print("Running slide-level DualCLAM tests...")
    _test_init()
    _test_from_config()
    _test_real_data_no_pretext()
    _test_real_data_with_pretext_sb()
    _test_real_data_with_pretext_mb()
    _test_instance_clustering_loss()
    _test_error_handling()
    print("All slide-level DualCLAM tests passed!")
