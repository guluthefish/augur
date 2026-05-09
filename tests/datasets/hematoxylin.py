"""Unit tests for the hematoxylin task functions."""

import numpy as np

from augur.datasets.hematoxylin import (
    extract_hematoxylin_channel,
    process_hematoxylin_task,
)


def _test_extract_hematoxylin_channel():
    """Test that the hematoxylin channel is extracted correctly."""
    print("Testing extract_hematoxylin_channel...")
    # Create a simple RGB image with known values
    rgb_image = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        ],
        dtype=np.uint8,
    )
    # Extract the hematoxylin channel
    hematoxylin_channel = extract_hematoxylin_channel(rgb_image)
    # Check the shape and dtype of the output
    assert hematoxylin_channel.shape == (
        2,
        3,
    ), f"Expected shape (2, 3). Got: {hematoxylin_channel.shape}"
    assert hematoxylin_channel.dtype == np.float32, (
        f"Expected dtype float32. Got: {hematoxylin_channel.dtype}"
    )
    # Check that the values are in the range [0, 1]
    assert np.all((hematoxylin_channel >= 0) & (hematoxylin_channel <= 1)), (
        "Values should be in the range [0, 1]"
    )
    print("[OK] extract_hematoxylin_channel test passed.")


def _test_process_hematoxylin_task():
    """Test that the process_hematoxylin_task function runs without errors."""
    print("Testing process_hematoxylin_task...")

    # Create a dummy base image (RGB)
    base_image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    item_dict = process_hematoxylin_task(base_image=base_image)
    # Check that the output contains the expected keys
    assert "image" in item_dict, "Output should contain 'image' key."
    assert "target" in item_dict, "Output should contain 'target' key."
    # Check that the image is a 2D array (hematoxylin channel)
    assert item_dict["target"].shape == (
        64,
        64,
    ), f"Expected target shape (64, 64). Got: {item_dict['target'].shape}"
    # Check that the image values are in the range [0, 1]
    assert np.all((item_dict["image"] >= 0) & (item_dict["image"] <= 1)), (
        "Image values should be in the range [0, 1]"
    )
    assert np.all((item_dict["target"] >= 0) & (item_dict["target"] <= 1)), (
        "Target values should be in the range [0, 1]"
    )

    print("[OK] process_hematoxylin_task test passed.")


def test_all_datasets_hematoxylin():
    """Run all hematoxylin task tests."""
    print("Running hematoxylin channel extraction tests...")
    _test_extract_hematoxylin_channel()
    _test_process_hematoxylin_task()
    print("All hematoxylin task tests passed!")


if __name__ == "__main__":
    test_all_datasets_hematoxylin()
