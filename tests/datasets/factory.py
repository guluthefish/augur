"""Unit tests for the dataset factory module."""

import yaml

from augur.datasets.factory import get_dataset_from_config


def _test_get_dataset_from_config():
    """Test the dataset factory function with a sample configuration."""
    print("Testing get_dataset_from_config ...")

    config_path = "configs/tile_dataset-TCGA-BRCA-test.yaml"

    # Load the configuration
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Create the dataset instance using the factory function
    dataset = get_dataset_from_config(config)

    # Assert that the dataset instance is created successfully
    assert dataset is not None, "Failed to create dataset instance from config."
    assert dataset.__class__.__name__ == "TCGATileDataset", (
        f"Expected dataset type 'TCGATileDataset', got '{dataset.__class__.__name__}'."
    )

    # Check with invalid dataset name
    invalid_config = {"name": "InvalidDataset"}
    try:
        get_dataset_from_config(invalid_config)
        assert False, "Expected ValueError for unsupported dataset name."
    except ValueError as e:
        assert str(e) == "Unsupported dataset name: InvalidDataset", (
            f"Unexpected error message: {str(e)}"
        )
    print("[OK] get_dataset_from_config tests passed.")


def test_all_datasets_factory():
    """Run all tests for the dataset factory."""
    print("Running dataset factory tests...")
    _test_get_dataset_from_config()
    print("All dataset factory tests passed!")
