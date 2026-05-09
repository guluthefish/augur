"""Tests for augur/utils/config.py."""

from torch import device

from augur.utils.config import load_yaml_config


def _test_load_yaml_config():
    """Test loading a YAML config file."""
    print("Testing load_yaml_config()...")
    config_path = "configs/encoder-prov-gigapath.yaml"
    config = load_yaml_config(config_path)

    assert isinstance(
        config, dict
    ), f"Expected config to be a dict. Got: {type(config)}"

    assert (
        config.get("name", None) == "ViTEncoder"
    ), f"Expected name 'ViTEncoder'. Got: {config.get('name')}"

    params = config.get("params", None)
    assert isinstance(
        params, dict
    ), f"Expected params to be a dict. Got: {type(params)}"
    assert (
        params.get("model_name", None) == "hf_hub:prov-gigapath/prov-gigapath"
    ), f"Expected model_name 'hf_hub:prov-gigapath/prov-gigapath'. Got: {params.get('model_name')}"
    assert (
        params.get("pretrained", None) is True
    ), f"Expected pretrained True. Got: {params.get('pretrained')}"

    optimizer_config = params.get("optimizer", None)
    assert isinstance(
        optimizer_config, dict
    ), f"Expected optimizer config to be a dict. Got: {type(optimizer_config)}"
    assert (
        optimizer_config.get("name", None) == "adamw"
    ), f"Expected optimizer name 'adamw'. Got: {optimizer_config.get('name')}"
    optimizer_params = optimizer_config.get("params", None)
    assert isinstance(
        optimizer_params, dict
    ), f"Expected optimizer params to be a dict or None. Got: {type(optimizer_params)}"

    lr_scheduler_config = params.get("lr_scheduler", None)
    assert isinstance(
        lr_scheduler_config, dict
    ), f"Expected lr_scheduler config to be a dict. Got: {type(lr_scheduler_config)}"
    assert (
        lr_scheduler_config.get("name", None) == "cosineannealinglr"
    ), f"Expected lr_scheduler name 'cosineannealinglr'. Got: {lr_scheduler_config.get('name')}"

    print("[OK] load_yaml_config() test passed.")


def test_all_utils_config():
    """Run all tests in augur/utils/config.py."""
    print("Running all config tests...")
    _test_load_yaml_config()
    print("All config tests passed!")
