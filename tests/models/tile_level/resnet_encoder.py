"""Unit tests for the tile-level ResNet encoder."""

from __future__ import annotations

import torch
from torchvision.models.resnet import BasicBlock

from augur.models.model_abc import ModelABC
from augur.models.tile_level.resnet_encoder import ResNetEncoder


def _test_init():
    """The encoder should inherit the shared Lightning base class."""
    print("Testing ResNetEncoder initialization...")
    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        optimizer_factory=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1},
    )

    assert isinstance(encoder, ModelABC)
    assert isinstance(encoder.fc, torch.nn.Identity), (
        f"Expected classifier head to be removed. Got: {type(encoder.fc)}."
    )
    assert isinstance(encoder.configure_optimizers(), torch.optim.SGD)
    print("[OK] ResNetEncoder initialization test passed.")


def _test_forward():
    """The encoder should preserve the original multi-scale forward API."""
    print("Testing ResNetEncoder.forward()...")
    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )
    image = torch.randn(2, 3, 224, 224)

    c0, c1, c2, c3, c4 = encoder(image)

    assert c0.shape == (
        2,
        64,
        56,
        56,
    ), (
        f"The initial conv layer should produce 64 channels at 1/4 resolution. Expected shape: (2, 64, 56, 56). Got shape: {c0.shape}."
    )
    assert c1.shape == (
        2,
        64,
        56,
        56,
    ), (
        f"The first ResNet block should preserve the number of channels and resolution. Expected shape: (2, 64, 56, 56). Got shape: {c1.shape}."
    )
    assert c2.shape == (
        2,
        128,
        28,
        28,
    ), (
        f"The second ResNet block should double the number of channels and halve the resolution. Expected shape: (2, 128, 28, 28). Got shape: {c2.shape}."
    )
    assert c3.shape == (
        2,
        256,
        14,
        14,
    ), (
        f"The third ResNet block should double the number of channels and halve the resolution. Expected shape: (2, 256, 14, 14). Got shape: {c3.shape}."
    )
    assert c4.shape == (
        2,
        512,
        7,
        7,
    ), (
        f"The fourth ResNet block should double the number of channels and halve the resolution. Expected shape: (2, 512, 7, 7). Got shape: {c4.shape}."
    )

    print("[OK] ResNetEncoder.forward() test passed.")


def _test_predict_step():
    """Lightning predict batches should accept the project's batch dict shape."""
    print("Testing ResNetEncoder.predict_step()...")
    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )
    batch = {"image": torch.randn(1, 3, 128, 128)}

    c0, c1, c2, c3, c4 = encoder.predict_step(batch, batch_idx=0)

    assert c0.shape == (
        1,
        64,
        32,
        32,
    ), f"Expected shape: (1, 64, 32, 32). Got shape: {c0.shape}."
    assert c1.shape == (
        1,
        64,
        32,
        32,
    ), f"Expected shape: (1, 64, 32, 32). Got shape: {c1.shape}."
    assert c2.shape == (
        1,
        128,
        16,
        16,
    ), f"Expected shape: (1, 128, 16, 16). Got shape: {c2.shape}."
    assert c3.shape == (
        1,
        256,
        8,
        8,
    ), f"Expected shape: (1, 256, 8, 8). Got shape: {c3.shape}."
    assert c4.shape == (
        1,
        512,
        4,
        4,
    ), f"Expected shape: (1, 512, 4, 4). Got shape: {c4.shape}."
    print("[OK] ResNetEncoder.predict_step() test passed.")


def _test_model_step():
    """Training hooks should stay explicit until a task-specific loss is defined."""
    print("Testing that ResNetEncoder.model_step() ...")

    encoder = ResNetEncoder(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )

    try:
        encoder.model_step({"image": torch.randn(1, 3, 64, 64)}, 0, "train")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("ResNetEncoder.model_step() should require an override.")

    print("[OK] ResNetEncoder.model_step() test passed.")


def _test_from_config():
    """The from_config method should parse config dicts and resolve optimization helpers."""
    print("Testing ResNetEncoder.from_config()...")
    config = {
        "block_name": "BasicBlock",
        "layers": [2, 2, 2, 2],
        "optimizer": {
            "name": "torch.optim.AdamW",
            "params": {"weight_decay": 0.01},
        },
        "lr_scheduler": {
            "name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "params": {"patience": 2, "factor": 0.5},
            "config": {"monitor": "val/loss", "interval": "epoch"},
        },
    }
    encoder = ResNetEncoder.from_config(config)

    assert isinstance(encoder, ResNetEncoder), (
        f"Expected a ResNetEncoder instance. Got: {type(encoder)}."
    )
    assert isinstance(encoder.configure_optimizers(), dict), (
        f"Expected a dict of optimizers and schedulers from config. Got: {type(encoder.configure_optimizers())}."
    )
    assert isinstance(encoder.configure_optimizers()["optimizer"], torch.optim.AdamW), (
        f"Expected AdamW optimizer from config. Got: {type(encoder.configure_optimizers()['optimizer'])}."
    )
    assert isinstance(
        encoder.configure_optimizers()["lr_scheduler"],
        dict,
    ), (
        f"Expected a dict of scheduler and metadata from config. Got: {type(encoder.configure_optimizers()['lr_scheduler'])}."
    )
    assert isinstance(
        encoder.configure_optimizers()["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ), (
        f"Expected ReduceLROnPlateau scheduler from config. Got: {type(encoder.configure_optimizers()['lr_scheduler'])}."
    )

    # We can also test that the learning rate scheduler is properly configured by checking the internal attributes of the encoder.
    assert encoder.lr_scheduler_factory is torch.optim.lr_scheduler.ReduceLROnPlateau, (
        f"Expected ReduceLROnPlateau scheduler factory. Got: {encoder.lr_scheduler_factory}."
    )
    assert encoder.lr_scheduler_kwargs == {
        "patience": 2,
        "factor": 0.5,
    }, f"Expected scheduler kwargs to match config. Got: {encoder.lr_scheduler_kwargs}."
    assert encoder.lr_scheduler_config == {
        "monitor": "val/loss",
        "interval": "epoch",
    }, f"Expected scheduler config to match config. Got: {encoder.lr_scheduler_config}."

    print("[OK] ResNetEncoder.from_config() test passed.")


def test_ResNetEncoder():
    """Run all ResNetEncoder unit tests."""
    print("Running all ResNetEncoder tests...")
    _test_init()
    _test_forward()
    _test_predict_step()
    _test_model_step()
    _test_from_config()
    print("All ResNetEncoder tests passed!")
