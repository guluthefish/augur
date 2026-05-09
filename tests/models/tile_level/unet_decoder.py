"""Unit tests for the tile-level U-Net decoder."""

from __future__ import annotations

import torch
from torchvision.models.resnet import BasicBlock

from VexDR.models.model_abc import ModelABC
from VexDR.models.tile_level.resnet_encoder import ResNetEncoder
from VexDR.models.tile_level.unet_decoder import UNetDecoder


def _test_init():
    """The decoder should inherit the shared Lightning base class."""
    print("Testing UNetDecoder initialization...")
    decoder = UNetDecoder(
        output_channels=3,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )

    assert isinstance(decoder, ModelABC)
    assert isinstance(decoder.configure_optimizers(), torch.optim.Adam)
    print("[OK] UNetDecoder initialization test passed.")


def _test_forward():
    """The decoder should reconstruct full-resolution logits from encoder features."""
    print("Testing UNetDecoder.forward()...")
    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )
    decoder = UNetDecoder(output_channels=3)
    image = torch.randn(2, 3, 224, 224)

    logits = decoder(encoder(image))

    assert logits.shape == (
        2,
        3,
        224,
        224,
    ), f"The decoder should return full-resolution logits. Expected shape: (2, 3, 224, 224). Got shape: {logits.shape}."
    print("[OK] UNetDecoder.forward() test passed.")


def _test_predict_step():
    """Lightning predict batches should accept feature-map dicts."""
    print("Testing UNetDecoder.predict_step()...")
    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )
    decoder = UNetDecoder(output_channels=1)
    features = encoder(torch.randn(1, 3, 128, 128))

    logits = decoder.predict_step({"features": features}, batch_idx=0)

    assert logits.shape == (
        1,
        1,
        128,
        128,
    ), f"The decoder predict step should preserve full tile resolution. Expected shape: (1, 1, 128, 128). Got shape: {logits.shape}."
    print("[OK] UNetDecoder.predict_step() test passed.")


def _test_model_step():
    """Training hooks should stay explicit until a task-specific loss is defined."""
    print("Testing that UNetDecoder.model_step() requires an override...")
    decoder = UNetDecoder(output_channels=2)
    features = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in (
            (64, 16),
            (64, 16),
            (128, 8),
            (256, 4),
            (512, 2),
        )
    )

    try:
        decoder.model_step({"features": features}, 0, "train")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("UNetDecoder.model_step() should require an override.")

    print("[OK] UNetDecoder.model_step() test passed.")


def _test_from_config():
    """The UNetDecoder should be instantiable from a config dict."""
    print("Testing UNetDecoder.from_config()...")
    config = {
        "output_channels": 2,
        "decoder_channels": [64, 64, 128, 256],
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

    decoder = UNetDecoder.from_config(config)

    assert isinstance(
        decoder, UNetDecoder
    ), f"Expected a UNetDecoder instance. Got: {type(decoder)}."
    assert isinstance(
        decoder.configure_optimizers(), dict
    ), f"Expected a dict of optimizers and schedulers from config. Got: {type(decoder.configure_optimizers())}."
    assert isinstance(
        decoder.configure_optimizers()["optimizer"], torch.optim.AdamW
    ), f"Expected AdamW optimizer from config. Got: {type(decoder.configure_optimizers()['optimizer'])}."
    assert isinstance(
        decoder.configure_optimizers()["lr_scheduler"], dict
    ), f"Expected a dict of schedulers from config. Got: {type(decoder.configure_optimizers()['lr_scheduler'])}."
    assert isinstance(
        decoder.configure_optimizers()["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    ), f"Expected StepLR scheduler from config. Got: {type(decoder.configure_optimizers()['lr_scheduler']['scheduler'])}."

    assert (
        decoder.output_channels == 2
    ), f"Expected output_channels to be 2. Got: {decoder.output_channels}."
    assert decoder.decoder_channels == (
        64,
        64,
        128,
        256,
    ), f"Expected decoder_channels to be (64, 64, 128, 256). Got: {decoder.decoder_channels}."
    assert (
        decoder.dropout == 0.1
    ), f"Expected dropout to be 0.1. Got: {decoder.dropout}."
    assert (
        decoder.upsample_mode == "bilinear"
    ), f"Expected upsample_mode to be 'bilinear'. Got: {decoder.upsample_mode}."
    assert (
        decoder.align_corners is True
    ), f"Expected align_corners to be True. Got: {decoder.align_corners}."

    print("[OK] UNetDecoder.from_config() test passed.")


def test_UNetDecoder():
    """Run all UNetDecoder unit tests."""
    print("Running all UNetDecoder tests...")
    _test_init()
    _test_forward()
    _test_predict_step()
    _test_model_step()
    _test_from_config()
    print("All UNetDecoder tests passed!")
