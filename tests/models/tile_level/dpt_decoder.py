"""Unit tests for the tile-level DPT decoder."""

from __future__ import annotations

import torch

from augur.models.model_abc import ModelABC
from augur.models.tile_level.dpt_decoder import DPTDecoder
from augur.models.tile_level.vit_encoder import ViTEncoder


def _test_init():
    """The decoder should inherit the shared Lightning base class."""
    print("Testing DPTDecoder initialization...")
    decoder = DPTDecoder(
        output_channels=3,
        embed_dim=192,
        optimizer_factory=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )

    assert isinstance(decoder, ModelABC)
    assert isinstance(decoder.configure_optimizers(), torch.optim.AdamW)
    print("[OK] DPTDecoder initialization test passed.")


def _test_forward_with_vit_tokens():
    """The decoder should reconstruct full-resolution logits from ViT tokens."""
    print("Testing DPTDecoder.forward() with ViT tokens...")
    encoder = ViTEncoder(model_name="vit_tiny_patch16_224")
    decoder = DPTDecoder(
        output_channels=2,
        embed_dim=encoder.embed_dim,  # type: ignore
        patch_size=16,
        num_prefix_tokens=encoder.num_prefix_tokens,
    )
    image = torch.randn(2, 3, 224, 224)

    logits = decoder(encoder(image))

    assert logits.shape == (
        2,
        2,
        224,
        224,
    ), f"Expected DPTDecoder logits shape (2, 2, 224, 224). Got: {logits.shape}."
    print("[OK] DPTDecoder.forward() test passed.")


def _test_predict_step_with_feature_dict():
    """Lightning predict batches should accept token metadata dicts."""
    print("Testing DPTDecoder.predict_step()...")
    decoder = DPTDecoder(
        output_channels=1,
        embed_dim=192,
        patch_size=16,
        num_prefix_tokens=1,
    )
    tokens = torch.randn(1, 65, 192)

    logits = decoder.predict_step({"tokens": tokens}, batch_idx=0)

    assert logits.shape == (
        1,
        1,
        128,
        128,
    ), (
        f"Expected DPTDecoder predict logits shape (1, 1, 128, 128). Got: {logits.shape}."
    )
    print("[OK] DPTDecoder.predict_step() test passed.")


def _test_model_step():
    """Training hooks should stay explicit until a task-specific loss is defined."""
    print("Testing that DPTDecoder.model_step() requires an override...")
    decoder = DPTDecoder(output_channels=1, embed_dim=192)
    tokens = torch.randn(1, 197, 192)

    try:
        decoder.model_step({"tokens": tokens}, 0, "train")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("DPTDecoder.model_step() should require an override.")

    print("[OK] DPTDecoder.model_step() test passed.")


def _test_from_config():
    """The DPTDecoder should be instantiable from a config dict."""
    print("Testing DPTDecoder.from_config()...")
    config = {
        "output_channels": 1,
        "embed_dim": 192,
        "feature_channels": 128,
        "head_channels": 64,
        "patch_size": 16,
        "num_prefix_tokens": 1,
        "dropout": 0.1,
        "upsample_mode": "bilinear",
        "align_corners": True,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "params": {"lr": 1e-4, "weight_decay": 0.01},
        },
        "lr_scheduler": {
            "name": "torch.optim.lr_scheduler.StepLR",
            "params": {"step_size": 5, "gamma": 0.5},
            "config": {"interval": "epoch"},
        },
    }

    decoder = DPTDecoder.from_config(config)

    assert isinstance(decoder, DPTDecoder), (
        f"Expected a DPTDecoder instance. Got: {type(decoder)}."
    )
    assert isinstance(decoder.configure_optimizers(), dict), (
        f"Expected a dict of optimizers and schedulers from config. Got: {type(decoder.configure_optimizers())}."
    )
    assert isinstance(decoder.configure_optimizers()["optimizer"], torch.optim.AdamW), (
        f"Expected AdamW optimizer from config. Got: {type(decoder.configure_optimizers()['optimizer'])}."
    )
    assert isinstance(decoder.configure_optimizers()["lr_scheduler"], dict), (
        f"Expected a dict of schedulers from config. Got: {type(decoder.configure_optimizers()['lr_scheduler'])}."
    )
    assert isinstance(
        decoder.configure_optimizers()["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    ), (
        f"Expected StepLR scheduler from config. Got: {type(decoder.configure_optimizers()['lr_scheduler']['scheduler'])}."
    )

    assert decoder.output_channels == 1, (
        f"Expected output_channels to be 1. Got: {decoder.output_channels}."
    )
    assert decoder.patch_size == (
        16,
        16,
    ), f"Expected patch_size to be (16, 16). Got: {decoder.patch_size}."
    assert decoder.feature_channels == 128, (
        f"Expected feature_channels to be 128. Got: {decoder.feature_channels}."
    )
    assert decoder.head_channels == 64, (
        f"Expected head_channels to be 64. Got: {decoder.head_channels}."
    )
    assert decoder.align_corners is True, (
        f"Expected align_corners to be True. Got: {decoder.align_corners}."
    )

    print("[OK] DPTDecoder.from_config() test passed.")


def test_DPTDecoder():
    """Run all DPTDecoder unit tests."""
    print("Running all DPTDecoder tests...")
    _test_init()
    _test_forward_with_vit_tokens()
    _test_predict_step_with_feature_dict()
    _test_model_step()
    _test_from_config()
    print("All DPTDecoder tests passed!")
