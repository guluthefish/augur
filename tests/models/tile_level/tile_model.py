"""Unit tests for the TileModel class."""

import torch

from augur.models.tile_level.tile_model import TileModel
from augur.models.tile_level.unet_decoder import UNetDecoder
from augur.models.tile_level.unet_encoder import UNetEncoder


def _test_init():
    """Test that the TileModel initializes correctly with valid inputs."""
    print("Testing TileModel initialization...")

    encoder = UNetEncoder(
        optimizer_factory=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )
    decoder = UNetDecoder(
        output_channels=22,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )

    model = TileModel(
        encoder=encoder,
        decoders={"tissue_segmentation": decoder},
        task_weights={"tissue_segmentation": 2.0},
    )

    assert isinstance(model, TileModel), (
        f"Expected model to be an instance of TileModel. Got: {type(model)!r}"
    )

    assert model.encoder is encoder, (
        "The model's encoder should be the one provided during initialization."
    )
    assert "tissue_segmentation" in model.decoders, (
        "The model's decoders should include the 'tissue_segmentation' task."
    )
    assert model.decoders["tissue_segmentation"] is decoder, (
        "The model's 'tissue_segmentation' decoder should be the one provided during initialization."
    )
    assert model.task_weights["tissue_segmentation"] == 1.0, (
        "The model's task weight for 'tissue_segmentation' should be 1.0 as provided during initialization."
    )

    print("[OK] TileModel initialization test passed.")


def _test_from_config():
    """Test that the TileModel can be initialized from a config dict."""
    print("Testing TileModel.from_config()...")

    config = {
        "encoder_config": {
            "name": "UNetEncoder",
            "params": {
                "optimizer_factory": torch.optim.AdamW,
                "optimizer_kwargs": {"lr": 1e-3},
            },
        },
        "decoders_config": {
            "tissue_segmentation": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 22,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
            "hematoxylin": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 1,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
        },
        "task_weights": {"tissue_segmentation": 1.0, "hematoxylin": 1.0},
    }

    model = TileModel.from_config(config)

    assert isinstance(model, TileModel), (
        f"Expected model to be an instance of TileModel. Got: {type(model)!r}"
    )
    assert isinstance(model.encoder, UNetEncoder), (
        f"Expected encoder to be an instance of UNetEncoder. Got: {type(model.encoder)!r}"
    )

    assert isinstance(model.decoders["tissue_segmentation"], UNetDecoder), (
        f"Expected 'tissue_segmentation' decoder to be an instance of UNetDecoder. Got: {type(model.decoders['tissue_segmentation'])!r}"
    )
    assert model.task_weights["tissue_segmentation"] == 0.5, (
        "The model's task weight for 'tissue_segmentation' should be 0.5 as provided in the config."
    )

    assert isinstance(model.decoders["hematoxylin"], UNetDecoder), (
        f"Expected 'hematoxylin' decoder to be an instance of UNetDecoder. Got: {type(model.decoders['hematoxylin'])!r}"
    )
    assert model.task_weights["hematoxylin"] == 0.5, (
        "The model's task weight for 'hematoxylin' should be 0.5 as provided in the config."
    )

    print("[OK] TileModel.from_config() test passed.")


def _test_forward():
    """Test that the TileModel's forward method produces outputs of the expected shape."""
    print("Testing TileModel forward pass...")

    config = {
        "encoder_config": {
            "name": "UNetEncoder",
            "params": {
                "optimizer_factory": torch.optim.AdamW,
                "optimizer_kwargs": {"lr": 1e-3},
            },
        },
        "decoders_config": {
            "tissue_segmentation": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 22,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
            "hematoxylin": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 1,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
        },
        "task_weights": {"tissue_segmentation": 1.0, "hematoxylin": 1.0},
    }

    model = TileModel.from_config(config)

    batch_size, input_channels, img_size = 4, 3, 224
    n_classes = 22

    dummy_input = {
        "tissue_segmentation": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
            "target": torch.randint(0, 1, (batch_size, n_classes, img_size, img_size)),
        },
        "hematoxylin": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
            "target": torch.rand(batch_size, 1, img_size, img_size),
        },
    }

    outputs = model(dummy_input)
    assert isinstance(outputs, dict), (
        f"Expected model output to be a dict. Got: {type(outputs)!r}"
    )

    # Check tissue_segmentation output
    assert "tissue_segmentation" in outputs, (
        "Expected output to include 'tissue_segmentation' key."
    )
    assert isinstance(outputs["tissue_segmentation"], torch.Tensor), (
        f"Expected 'tissue_segmentation' output to be a Tensor. Got: {type(outputs['tissue_segmentation'])!r}"
    )
    assert outputs["tissue_segmentation"].shape == (
        batch_size,
        n_classes,
        img_size,
        img_size,
    ), (
        f"Expected 'tissue_segmentation' output shape to be {(batch_size, n_classes, img_size, img_size)}. Got: {outputs['tissue_segmentation'].shape}"
    )

    # Check hematoxylin output
    assert "hematoxylin" in outputs, "Expected output to include 'hematoxylin' key."
    assert isinstance(outputs["hematoxylin"], torch.Tensor), (
        f"Expected 'hematoxylin' output to be a Tensor. Got: {type(outputs['hematoxylin'])!r}"
    )
    assert outputs["hematoxylin"].shape == (
        batch_size,
        1,
        img_size,
        img_size,
    ), (
        f"Expected 'hematoxylin' output shape to be {(batch_size, 1, img_size, img_size)}. Got: {outputs['hematoxylin'].shape}"
    )

    print("[OK] TileModel.forward() test passed.")


def _test_predict_step():
    """Test that the TileModel's predict_step method produces outputs of the expected shape."""
    print("Testing TileModel predict_step...")

    config = {
        "encoder_config": {
            "name": "UNetEncoder",
            "params": {
                "optimizer_factory": torch.optim.AdamW,
                "optimizer_kwargs": {"lr": 1e-3},
            },
        },
        "decoders_config": {
            "tissue_segmentation": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 22,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
            "hematoxylin": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 1,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
        },
        "task_weights": {"tissue_segmentation": 1.0, "hematoxylin": 1.0},
    }

    model = TileModel.from_config(config)

    batch_size, input_channels, img_size = 4, 3, 224

    dummy_input = {
        "tissue_segmentation": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
        },
        "hematoxylin": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
        },
    }

    outputs = model.predict_step(dummy_input, batch_idx=0)
    assert isinstance(outputs, dict), (
        f"Expected model output to be a dict. Got: {type(outputs)!r}"
    )

    # Check tissue_segmentation output
    assert "tissue_segmentation" in outputs, (
        "Expected output to include 'tissue_segmentation' key."
    )
    assert isinstance(outputs["tissue_segmentation"], torch.Tensor), (
        f"Expected 'tissue_segmentation' output to be a Tensor. Got: {type(outputs['tissue_segmentation'])!r}"
    )
    assert outputs["tissue_segmentation"].shape == (
        batch_size,
        22,
        img_size,
        img_size,
    ), (
        f"Expected 'tissue_segmentation' output shape to be {(batch_size, 22, img_size, img_size)}. Got: {outputs['tissue_segmentation'].shape}"
    )

    print("[OK] TileModel.predict_step() test passed.")


def _test_model_step():
    """Test that TileModel's model_step return the expected loss dict structure."""
    print("Testing TileModel model_step...")
    config = {
        "encoder_config": {
            "name": "UNetEncoder",
            "params": {
                "optimizer_factory": torch.optim.AdamW,
                "optimizer_kwargs": {"lr": 1e-3},
            },
        },
        "decoders_config": {
            "tissue_segmentation": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 22,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
            "hematoxylin": {
                "name": "UNetDecoder",
                "params": {
                    "output_channels": 1,
                    "optimizer_factory": torch.optim.Adam,
                    "optimizer_kwargs": {"lr": 1e-3},
                },
            },
        },
        "task_weights": {"tissue_segmentation": 1.0, "hematoxylin": 1.0},
        "task_kwargs": {"tissue_segmentation": {"unknown_class_index": 0}},
    }

    model = TileModel.from_config(config)

    batch_size, input_channels, img_size = 4, 3, 224
    n_classes = 22

    dummy_input = {
        "tissue_segmentation": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
            "target": torch.randint(0, 1, (batch_size, n_classes, img_size, img_size)),
        },
        "hematoxylin": {
            "image": torch.randn(batch_size, input_channels, img_size, img_size),
            "target": torch.rand(batch_size, 1, img_size, img_size),
        },
    }

    total_loss, metrics = model.model_step(dummy_input, batch_idx=0, stage="train")

    assert isinstance(total_loss, torch.Tensor), (
        f"Expected total_loss to be a Tensor. Got: {type(total_loss)!r}"
    )
    assert total_loss.ndim == 0, (
        f"Expected total_loss to be a scalar Tensor. Got a Tensor with shape: {total_loss.shape}"
    )

    assert isinstance(metrics, dict), (
        f"Expected metrics to be a dict. Got: {type(metrics)!r}"
    )
    assert "tissue_segmentation_loss" in metrics, (
        "Expected metrics to include 'tissue_segmentation_loss' key."
    )
    assert "hematoxylin_loss" in metrics, (
        "Expected metrics to include 'hematoxylin_loss' key."
    )
    assert isinstance(metrics["tissue_segmentation_loss"], torch.Tensor), (
        f"Expected 'tissue_segmentation_loss' to be a Tensor. Got: {type(metrics['tissue_segmentation_loss'])!r}"
    )
    assert isinstance(metrics["hematoxylin_loss"], torch.Tensor), (
        f"Expected 'hematoxylin_loss' to be a Tensor. Got: {type(metrics['hematoxylin_loss'])!r}"
    )
    assert metrics["tissue_segmentation_loss"].ndim == 0, (
        f"Expected 'tissue_segmentation_loss' to be a scalar Tensor. Got a Tensor with shape: {metrics['tissue_segmentation_loss'].shape}"
    )
    assert metrics["hematoxylin_loss"].ndim == 0, (
        f"Expected 'hematoxylin_loss' to be a scalar Tensor. Got a Tensor with shape: {metrics['hematoxylin_loss'].shape}"
    )

    print("[OK] TileModel.model_step() test passed.")


def test_TileModel():
    """Run all TileModel tests."""
    print("Running all TileModel tests...")
    _test_init()
    _test_from_config()
    _test_forward()
    _test_predict_step()
    _test_model_step()
    print("All TileModel tests passed!")


# if __name__ == "__main__":
#     test_TileModel()
