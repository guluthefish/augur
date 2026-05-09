"""Unit tests for model configuration helpers."""

from __future__ import annotations

import torch

from VexDR.models.tile_level.unet_encoder import UNetEncoder
from VexDR.models.utils import (
    get_lr_scheduler_from_config,
    get_optimizer_from_config,
)
from VexDR.utils import load_yaml_config


def _test_get_optimizer_from_config():
    """Optimizer config parsing should resolve factories and default kwargs."""
    print("Testing get_optimizer_from_config()...")
    optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
        {
            "name": "torch.optim.AdamW",
            "params": {"weight_decay": 0.01},
        }
    )

    assert (
        optimizer_factory is torch.optim.AdamW
    ), "Expected torch.optim.AdamW from the optimizer config."
    assert optimizer_kwargs == {
        "lr": 1e-3,
        "weight_decay": 0.01,
    }, f"Expected default AdamW lr merged with weight_decay. Got: {optimizer_kwargs}"

    try:
        get_optimizer_from_config({"params": {"lr": 1e-4}})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "get_optimizer_from_config() should reject optimizer kwargs without a name."
        )

    print("[OK] get_optimizer_from_config() test passed.")


def _test_get_lr_scheduler_from_config():
    """Scheduler config parsing should resolve factories, kwargs, and metadata."""
    print("Testing get_lr_scheduler_from_config()...")
    scheduler_factory, scheduler_kwargs, scheduler_config = (
        get_lr_scheduler_from_config(
            {
                "name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "params": {"patience": 2, "factor": 0.5},
                "config": {"monitor": "val/loss", "interval": "epoch"},
            }
        )
    )

    assert (
        scheduler_factory is torch.optim.lr_scheduler.ReduceLROnPlateau
    ), "Expected torch.optim.lr_scheduler.ReduceLROnPlateau from the scheduler config."
    assert scheduler_kwargs == {
        "patience": 2,
        "factor": 0.5,
    }, f"Expected scheduler kwargs to be preserved. Got: {scheduler_kwargs}"
    assert scheduler_config == {
        "monitor": "val/loss",
        "interval": "epoch",
    }, f"Expected Lightning scheduler metadata to be preserved. Got: {scheduler_config}"
    print("[OK] get_lr_scheduler_from_config() test passed.")


def _test_load_yaml_config_with_extends():
    """Config inheritance should deep-merge shared params into the model config."""
    print("Testing load_yaml_config() with extends...")
    config = load_yaml_config("configs/encoder-prov-gigapath.yaml")

    assert config["name"] == "ViTEncoder", f"Unexpected model name: {config['name']}"
    assert (
        config["params"]["model_name"] == "hf_hub:prov-gigapath/prov-gigapath"
    ), "The local ViT config should keep its model_name after merging."
    assert (
        config["params"]["pretrained"] is True
    ), "The local ViT config should keep its pretrained flag after merging."
    assert config["params"]["optimizer"] == {
        "name": "adamw",
        "params": {"lr": 0.0001, "weight_decay": 0.01},
    }, f"Unexpected merged optimizer config: {config['params']['optimizer']}"
    assert config["params"]["lr_scheduler"] == {
        "name": "cosineannealinglr",
        "params": {"T_max": 10, "eta_min": 1e-6},
    }, f"Unexpected merged lr_scheduler config: {config['params']['lr_scheduler']}"
    print("[OK] load_yaml_config() extends test passed.")


def test_all_models_utils():
    """Run all model utility unit tests."""
    print("Running model utility tests...")
    _test_get_optimizer_from_config()
    _test_get_lr_scheduler_from_config()
    _test_load_yaml_config_with_extends()
    print("All model utility tests passed!")
