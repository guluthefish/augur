"""Unit tests for the tile-level ViT encoder."""

from __future__ import annotations

import torch

from VexDR.models.model_abc import ModelABC
from VexDR.models.tile_level.vit_encoder import ViTEncoder


def _test_init():
    """The encoder should inherit the shared Lightning base class."""
    print("Testing ViTEncoder initialization...")
    encoder = ViTEncoder(
        model_name="vit_tiny_patch16_224",
        optimizer_factory=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )

    assert isinstance(encoder, ModelABC)
    assert isinstance(encoder.configure_optimizers(), torch.optim.AdamW)
    assert encoder.model_name == "vit_tiny_patch16_224"
    print("[OK] ViTEncoder initialization test passed.")


def _test_forward():
    """The encoder should return the token sequence from the timm backbone."""
    print("Testing ViTEncoder.forward()...")
    encoder = ViTEncoder(model_name="vit_tiny_patch16_224")
    image = torch.randn(2, 3, 224, 224)

    tokens = encoder(image)

    assert tokens.shape == (
        2,
        197,
        192,
    ), f"Expected ViT token shape (2, 197, 192) for vit_tiny_patch16_224. Got: {tokens.shape}."
    print("[OK] ViTEncoder.forward() test passed.")


def _test_predict_step():
    """Lightning predict batches should accept the project's batch dict shape."""
    print("Testing ViTEncoder.predict_step()...")
    encoder = ViTEncoder(model_name="vit_tiny_patch16_224")
    batch = {"image": torch.randn(1, 3, 224, 224)}

    tokens = encoder.predict_step(batch, batch_idx=0)

    assert tokens.shape == (
        1,
        197,
        192,
    ), f"Expected ViT predict tokens shape (1, 197, 192). Got: {tokens.shape}."
    print("[OK] ViTEncoder.predict_step() test passed.")


def _test_from_config():
    """Config construction should use the reduced timm-facing argument surface."""
    print("Testing ViTEncoder.from_config()...")
    encoder = ViTEncoder.from_config(
        {
            "model_name": "vit_tiny_patch16_224",
            "pretrained": False,
            "img_size": 224,
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 5e-4},
            },
            "lr_scheduler": {
                "name": "StepLR",
                "params": {"step_size": 5, "gamma": 0.5},
                "config": {"interval": "epoch", "frequency": 1},
            },
        }
    )

    assert (
        encoder.optimizer_factory is torch.optim.AdamW
    ), "Expected AdamW optimizer factory from config."
    assert encoder.optimizer_kwargs == {
        "lr": 5e-4
    }, f"Expected AdamW optimizer kwargs from config. Got: {encoder.optimizer_kwargs}"
    assert (
        encoder.lr_scheduler_factory is torch.optim.lr_scheduler.StepLR
    ), "Expected StepLR scheduler factory from config."
    assert encoder.lr_scheduler_kwargs == {
        "step_size": 5,
        "gamma": 0.5,
    }, f"Expected StepLR kwargs from config. Got: {encoder.lr_scheduler_kwargs}"
    print("[OK] ViTEncoder.from_config() test passed.")


def _test_model_step():
    """Training hooks should stay explicit until a task-specific loss is defined."""
    print("Testing that ViTEncoder.model_step() requires an override...")
    encoder = ViTEncoder(model_name="vit_tiny_patch16_224")

    try:
        encoder.model_step({"image": torch.randn(1, 3, 224, 224)}, 0, "train")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("ViTEncoder.model_step() should require an override.")

    print("[OK] ViTEncoder.model_step() test passed.")


def test_ViTEncoder():
    """Run all ViTEncoder unit tests."""
    print("Running all ViTEncoder tests...")
    _test_init()
    _test_forward()
    _test_predict_step()
    _test_from_config()
    _test_model_step()
    print("All ViTEncoder tests passed!")
