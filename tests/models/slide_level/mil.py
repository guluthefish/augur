"""Unit tests for slide-level MIL models using real slide data."""

from __future__ import annotations

from typing import Any

import torch

from VexDR.datasets.tcga_slide_dataset import TCGASlideDataset
from VexDR.models.model_abc import ModelABC
from VexDR.models.slide_level.attention import Attention, GatedAttention
from VexDR.models.slide_level.mil import EmbeddingMIL, _MaxPool, _MeanPool


class _TinyTileEncoder(ModelABC):
    """Fast deterministic encoder for real tile batches in MIL tests."""

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


def _load_real_slide_batch() -> tuple[TCGASlideDataset, dict[str, Any]]:
    """Load one small real TCGA slide batch for MIL aggregation tests."""
    datamodule = TCGASlideDataset(
        root_dir="data/TCGA-BRCA-test",
        main_task="subtyping",
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


def _encode_expected_features(
    encoder: _TinyTileEncoder,
    image: torch.Tensor,
) -> torch.Tensor:
    """Encode a real slide tile batch to expected bag features."""
    batch_size, num_tiles = image.shape[:2]
    flat_features = encoder(image.flatten(0, 1))
    return flat_features.view(batch_size, num_tiles, -1)


def _assert_real_slide_batch(
    datamodule: TCGASlideDataset,
    batch: dict[str, Any],
) -> None:
    """Validate the real data batch contract consumed by EmbeddingMIL."""
    expected_keys = {"image", "mask", "target", "metadata"}
    assert (
        set(batch.keys()) == expected_keys
    ), f"Expected real slide batch keys {expected_keys}. Got: {set(batch.keys())}."

    image = batch["image"]
    target = batch["target"]
    assert isinstance(
        image, torch.Tensor
    ), f"Expected image tensor. Got: {type(image)}."
    K = image.shape[1]
    assert K >= 1, "Each slide bag should contain at least one tile."
    assert image.shape[2:] == (
        3,
        datamodule.image_size,
        datamodule.image_size,
    ), f"Unexpected real slide image shape: {image.shape}."
    assert torch.all(
        (image >= 0) & (image <= 1)
    ), "Real tile values should be in [0, 1]."

    assert isinstance(
        target, torch.Tensor
    ), f"Expected target tensor. Got: {type(target)}."
    assert target.shape == (image.shape[0],), (
        f"Subtyping target should be a (B,) long tensor. Got: {target.shape}."
    )
    assert (
        target.dtype == torch.long
    ), f"Subtyping target should be long. Got: {target.dtype}."
    assert torch.all(
        (target >= 0) & (target < datamodule.num_main_labels)
    ), "Subtyping class indices must lie in [0, num_main_labels)."


def _build_mil_model(
    aggregation_method: str,
    *,
    output_dim: int,
) -> EmbeddingMIL:
    """Build a tiny real-data MIL model for one aggregation method."""
    encoder = _TinyTileEncoder()
    if aggregation_method == "attention":
        return EmbeddingMIL(
            aggregation_method="attention",
            encoder=encoder,
            enc_dim=encoder.feature_dim * 2,
            hidden_dims=[],
            output_dim=output_dim,
            attn_kwargs={
                "input_dim": encoder.feature_dim,
                "hidden_dim": 8,
                "num_heads": 2,
                "dropout": 0.0,
            },
        )
    return EmbeddingMIL(
        aggregation_method=aggregation_method,  # type: ignore[arg-type]
        encoder=encoder,
        enc_dim=encoder.feature_dim,
        hidden_dims=[],
        output_dim=output_dim,
    )


def _assert_common_mil_outputs(
    model: EmbeddingMIL,
    batch: dict[str, Any],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Run forward and model_step checks shared by all aggregation methods."""
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        loss, metrics = model.model_step(batch, batch_idx=0, stage="train")

    prediction = outputs["subtyping"]
    target = batch["target"]
    assert prediction.shape == (target.shape[0], num_classes), (
        f"Subtyping prediction should have shape (B, num_classes). "
        f"Expected ({target.shape[0]}, {num_classes}). Got: {prediction.shape}."
    )
    assert torch.isfinite(prediction).all(), "MIL predictions should be finite."
    assert isinstance(loss, torch.Tensor), f"Expected loss tensor. Got: {type(loss)}."
    assert loss.ndim == 0, f"Expected scalar loss. Got shape: {loss.shape}."
    assert torch.isfinite(loss), "MIL loss should be finite."

    assert isinstance(metrics, dict), f"Expected metrics dict. Got: {type(metrics)}."
    assert set(metrics.keys()) == {
        "subtyping_loss"
    }, f"Expected subtyping_loss metric. Got: {set(metrics.keys())}."
    assert torch.isfinite(
        metrics["subtyping_loss"]
    ), "Subtyping loss metric should be finite."
    return outputs


def _test_real_data_max_and_mean_aggregation() -> None:
    """Max and mean MIL should aggregate encoded real slide tiles correctly."""
    print("Testing EmbeddingMIL max and mean aggregation on real slide data...")

    datamodule, batch = _load_real_slide_batch()
    try:
        _assert_real_slide_batch(datamodule, batch)

        for aggregation_method in ("max", "mean"):
            model = _build_mil_model(
                aggregation_method,
                output_dim=datamodule.num_main_labels,
            )
            outputs = _assert_common_mil_outputs(
                model, batch, num_classes=datamodule.num_main_labels
            )
            assert (
                outputs["_attention_weights"] is None
            ), f"{aggregation_method} aggregation should not return attention weights."

            features = _encode_expected_features(model.encoder, batch["image"])  # type: ignore[arg-type]
            expected_aggregated = (
                features.max(dim=1).values
                if aggregation_method == "max"
                else features.mean(dim=1)
            )
            assert torch.allclose(
                outputs["_aggregated"],
                expected_aggregated,
                atol=1e-6,
            ), f"{aggregation_method} aggregation should match encoded real tile features."
    finally:
        datamodule.teardown()

    print("[OK] EmbeddingMIL max and mean real-data aggregation test passed.")


def _test_real_data_attention_aggregation() -> None:
    """Attention MIL should aggregate encoded real slide tiles with valid weights."""
    print("Testing EmbeddingMIL attention aggregation on real slide data...")

    datamodule, batch = _load_real_slide_batch()
    try:
        _assert_real_slide_batch(datamodule, batch)
        model = _build_mil_model("attention", output_dim=datamodule.num_main_labels)

        outputs = _assert_common_mil_outputs(
            model, batch, num_classes=datamodule.num_main_labels
        )
        attention_weights = outputs["_attention_weights"]
        assert isinstance(
            attention_weights,
            torch.Tensor,
        ), f"Expected attention weights tensor. Got: {type(attention_weights)}."
        assert attention_weights.shape == (
            batch["image"].shape[0],
            2,
            batch["image"].shape[1],
        ), f"Unexpected attention weight shape: {attention_weights.shape}."
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(batch["image"].shape[0], 2),
            atol=1e-6,
        ), "Attention weights should sum to 1 for each real slide bag and head."

        features = _encode_expected_features(model.encoder, batch["image"])  # type: ignore[arg-type]
        expected_aggregated = torch.einsum(
            "bmk,bkd->bmd",
            attention_weights,
            features,
        ).flatten(start_dim=1)
        assert torch.allclose(
            outputs["_aggregated"],
            expected_aggregated,
            atol=1e-6,
        ), "Attention aggregation should match the weighted sum of encoded real tile features."
    finally:
        datamodule.teardown()

    print("[OK] EmbeddingMIL attention real-data aggregation test passed.")


def _test_from_config() -> None:
    """EmbeddingMIL.from_config should parse a full config into a working model."""
    print("Testing EmbeddingMIL.from_config()...")

    # Case 1: pre-computed features (no tile_model), mean aggregation, minimal config.
    mean_config = {
        "aggregation_method": "mean",
        "enc_dim": 8,
        "hidden_dims": [],
        "output_dim": 5,
    }
    mean_model = EmbeddingMIL.from_config(mean_config)
    assert isinstance(mean_model, EmbeddingMIL)
    assert mean_model.aggregation_method == "mean"
    assert not mean_model.has_encoder, "No tile_model implies encoder-less model."
    assert isinstance(mean_model.aggregator, _MeanPool)
    assert (
        mean_model.main_task == "subtyping"
    ), "main_task should default to 'subtyping' when omitted from config."
    assert (
        mean_model.unknown_class_index == 0
    ), "unknown_class_index should default to 0."

    mean_model.eval()
    bag = torch.randn(2, 4, 8)
    with torch.no_grad():
        outputs = mean_model(bag)
    assert outputs["subtyping"].shape == (2, 5)
    assert outputs["_attention_weights"] is None

    # Case 2: pre-computed features, max aggregation — different aggregator path.
    max_model = EmbeddingMIL.from_config(
        {
            "aggregation_method": "max",
            "enc_dim": 8,
            "hidden_dims": [16],
            "output_dim": 3,
            "dropout": 0.1,
            "task_kwargs": {"subtyping": {"unknown_class_index": None}},
        }
    )
    assert isinstance(max_model.aggregator, _MaxPool)
    assert (
        max_model.unknown_class_index is None
    ), "task_kwargs['subtyping']['unknown_class_index']=None should disable ignore-index."
    assert max_model.task_kwargs == {"subtyping": {"unknown_class_index": None}}
    max_model.eval()
    with torch.no_grad():
        out_max = max_model(bag)
    assert out_max["subtyping"].shape == (2, 3)

    # Case 3: attention aggregation with attn_kwargs + optimizer + lr_scheduler.
    attn_config = {
        "aggregation_method": "attention",
        "enc_dim": 12,  # = attn input_dim * num_heads (aggregated feature dim).
        "hidden_dims": [],
        "output_dim": 4,
        "attn_kwargs": {
            "input_dim": 6,
            "hidden_dim": 8,
            "num_heads": 2,
            "dropout": 0.0,
            "gated": True,
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
    attn_model = EmbeddingMIL.from_config(attn_config)
    assert attn_model.aggregation_method == "attention"
    assert isinstance(
        attn_model.aggregator, GatedAttention
    ), "gated=True should instantiate GatedAttention."
    assert attn_model.attn_gated is True
    assert attn_model.aggregator.num_heads == 2
    assert attn_model.aggregator.hidden_dim == 8

    assert (
        attn_model.optimizer_factory is torch.optim.AdamW
    ), f"Expected AdamW optimizer factory. Got: {attn_model.optimizer_factory}."
    assert attn_model.optimizer_kwargs == {
        "lr": 5e-4,
        "weight_decay": 0.01,
    }, f"Expected AdamW kwargs from config. Got: {attn_model.optimizer_kwargs}."
    assert (
        attn_model.lr_scheduler_factory is torch.optim.lr_scheduler.StepLR
    ), f"Expected StepLR scheduler factory. Got: {attn_model.lr_scheduler_factory}."
    assert attn_model.lr_scheduler_kwargs == {"step_size": 3, "gamma": 0.5}
    assert attn_model.lr_scheduler_config == {"interval": "epoch", "frequency": 1}

    optimizers = attn_model.configure_optimizers()
    assert isinstance(optimizers, dict)
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    assert isinstance(
        optimizers["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    )

    # Forward pass on a pre-computed bag shaped for the attention input_dim.
    attn_model.eval()
    attn_bag = torch.randn(2, 4, 6)
    with torch.no_grad():
        out_attn = attn_model(attn_bag)
    assert out_attn["subtyping"].shape == (2, 4)
    attention_weights = out_attn["_attention_weights"]
    assert isinstance(attention_weights, torch.Tensor)
    assert attention_weights.shape == (2, 2, 4)
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones(2, 2),
        atol=1e-6,
    )

    # Case 4: non-gated attention should fall back to the plain Attention class.
    plain_attn_model = EmbeddingMIL.from_config(
        {
            "aggregation_method": "attention",
            "enc_dim": 6,
            "hidden_dims": [],
            "output_dim": 2,
            "attn_kwargs": {
                "input_dim": 6,
                "hidden_dim": 4,
                "num_heads": 1,
                "gated": False,
            },
        }
    )
    assert isinstance(plain_attn_model.aggregator, Attention) and not isinstance(
        plain_attn_model.aggregator, GatedAttention
    ), "gated=False should instantiate plain Attention."
    assert plain_attn_model.attn_gated is False

    # Case 5: unsupported main_task should fail loudly.
    try:
        EmbeddingMIL.from_config(
            {
                "main_task": "regression",
                "enc_dim": 8,
                "hidden_dims": [],
                "output_dim": 5,
            }
        )
        assert False, "Expected unsupported main_task to raise."
    except (AssertionError, ValueError):
        pass

    print("[OK] EmbeddingMIL.from_config() test passed.")


def test_EmbeddingMIL() -> None:
    """Run all slide-level EmbeddingMIL unit tests."""
    print("Running slide-level EmbeddingMIL tests...")
    _test_from_config()
    _test_real_data_max_and_mean_aggregation()
    _test_real_data_attention_aggregation()
    print("All slide-level EmbeddingMIL tests passed!")
