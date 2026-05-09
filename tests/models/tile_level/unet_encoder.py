"""Unit tests for the tile-level U-Net encoder."""

from __future__ import annotations

import torch

from augur.models.model_abc import ModelABC
from augur.models.tile_level.unet_decoder import UNetDecoder
from augur.models.tile_level.unet_encoder import UNetEncoder


def _test_init():
    """The encoder should inherit the shared Lightning base class."""
    print("Testing UNetEncoder initialization...")
    encoder = UNetEncoder(
        optimizer_factory=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )

    assert isinstance(encoder, ModelABC)
    assert isinstance(encoder.configure_optimizers(), torch.optim.AdamW)
    assert encoder.feature_channels == (
        64,
        64,
        128,
        256,
        512,
    ), (
        f"Expected default feature channels (64, 64, 128, 256, 512). Got: {encoder.feature_channels}."
    )
    print("[OK] UNetEncoder initialization test passed.")


def _test_forward():
    """The encoder should return the expected five-scale feature pyramid."""
    print("Testing UNetEncoder.forward()...")
    encoder = UNetEncoder()
    image = torch.randn(2, 3, 224, 224)

    c0, c1, c2, c3, c4 = encoder(image)

    assert c0.shape == (
        2,
        64,
        56,
        56,
    ), f"Expected c0 shape (2, 64, 56, 56). Got: {c0.shape}."
    assert c1.shape == (
        2,
        64,
        56,
        56,
    ), f"Expected c1 shape (2, 64, 56, 56). Got: {c1.shape}."
    assert c2.shape == (
        2,
        128,
        28,
        28,
    ), f"Expected c2 shape (2, 128, 28, 28). Got: {c2.shape}."
    assert c3.shape == (
        2,
        256,
        14,
        14,
    ), f"Expected c3 shape (2, 256, 14, 14). Got: {c3.shape}."
    assert c4.shape == (
        2,
        512,
        7,
        7,
    ), f"Expected c4 shape (2, 512, 7, 7). Got: {c4.shape}."
    print("[OK] UNetEncoder.forward() test passed.")


def _test_predict_step():
    """Lightning predict batches should accept the project's batch dict shape."""
    print("Testing UNetEncoder.predict_step()...")
    encoder = UNetEncoder()
    batch = {"image": torch.randn(1, 3, 128, 128)}

    c0, c1, c2, c3, c4 = encoder.predict_step(batch, batch_idx=0)

    assert c0.shape == (
        1,
        64,
        32,
        32,
    ), f"Expected c0 shape (1, 64, 32, 32). Got: {c0.shape}."
    assert c1.shape == (
        1,
        64,
        32,
        32,
    ), f"Expected c1 shape (1, 64, 32, 32). Got: {c1.shape}."
    assert c2.shape == (
        1,
        128,
        16,
        16,
    ), f"Expected c2 shape (1, 128, 16, 16). Got: {c2.shape}."
    assert c3.shape == (
        1,
        256,
        8,
        8,
    ), f"Expected c3 shape (1, 256, 8, 8). Got: {c3.shape}."
    assert c4.shape == (
        1,
        512,
        4,
        4,
    ), f"Expected c4 shape (1, 512, 4, 4). Got: {c4.shape}."
    print("[OK] UNetEncoder.predict_step() test passed.")


def _test_decoder_compatibility():
    """The default encoder pyramid should work with the default U-Net decoder."""
    print("Testing UNetEncoder compatibility with UNetDecoder...")
    encoder = UNetEncoder()
    decoder = UNetDecoder(output_channels=3)
    image = torch.randn(2, 3, 224, 224)

    logits = decoder(encoder(image))

    assert logits.shape == (
        2,
        3,
        224,
        224,
    ), f"Expected encoder-decoder logits shape (2, 3, 224, 224). Got: {logits.shape}."
    print("[OK] UNetEncoder compatibility test passed.")


def _test_model_step():
    """Training hooks should stay explicit until a task-specific loss is defined."""
    print("Testing that UNetEncoder.model_step() requires an override...")
    encoder = UNetEncoder()

    try:
        encoder.model_step({"image": torch.randn(1, 3, 64, 64)}, 0, "train")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("UNetEncoder.model_step() should require an override.")

    print("[OK] UNetEncoder.model_step() test passed.")


def _test_from_config():
    """UNetEncoder.from_config() should pass parsed optimizer and scheduler settings."""
    print("Testing UNetEncoder.from_config()...")
    encoder = UNetEncoder.from_config(
        {
            "input_channels": 3,
            "feature_channels": (64, 64, 128, 256, 512),
            "dropout": 0.1,
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 5e-4, "weight_decay": 0.01},
            },
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "params": {"patience": 3},
                "config": {"monitor": "val/loss", "interval": "epoch"},
            },
        }
    )

    assert encoder.optimizer_factory is torch.optim.AdamW, (
        "Expected UNetEncoder.from_config() to attach the AdamW factory."
    )
    assert encoder.optimizer_kwargs == {
        "lr": 5e-4,
        "weight_decay": 0.01,
    }, (
        f"Expected UNetEncoder optimizer kwargs to match the config. Got: {encoder.optimizer_kwargs}"
    )
    assert encoder.lr_scheduler_factory is torch.optim.lr_scheduler.ReduceLROnPlateau, (
        "Expected UNetEncoder.from_config() to attach the ReduceLROnPlateau factory."
    )
    assert encoder.lr_scheduler_kwargs == {"patience": 3}, (
        f"Expected UNetEncoder scheduler kwargs to match the config. Got: {encoder.lr_scheduler_kwargs}"
    )
    assert encoder.lr_scheduler_config == {
        "monitor": "val/loss",
        "interval": "epoch",
    }, (
        f"Expected UNetEncoder scheduler config to match the config. Got: {encoder.lr_scheduler_config}"
    )

    assert encoder.input_channels == 3, (
        f"Expected input_channels to be 3. Got: {encoder.input_channels}."
    )
    assert encoder.feature_channels == (
        64,
        64,
        128,
        256,
        512,
    ), (
        f"Expected feature_channels to be (64, 64, 128, 256, 512). Got: {encoder.feature_channels}."
    )
    assert encoder.dropout == 0.1, (
        f"Expected dropout to be 0.1. Got: {encoder.dropout}."
    )
    assert isinstance(encoder, UNetEncoder), (
        f"Expected UNetEncoder.from_config() to return a UNetEncoder instance. Got: {type(encoder)}."
    )

    print("[OK] UNetEncoder.from_config() test passed.")


def test_UNetEncoder():
    """Run all UNetEncoder unit tests."""
    print("Running all UNetEncoder tests...")
    _test_init()
    _test_forward()
    _test_predict_step()
    _test_decoder_compatibility()
    _test_model_step()
    _test_from_config()
    print("All UNetEncoder tests passed!")
