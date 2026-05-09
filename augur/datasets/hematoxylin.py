"""Hematoxylin task processors for TCGA histology tiles."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


def extract_hematoxylin_channel(
    image: np.ndarray, logger: logging.Logger | None = None
) -> np.ndarray:
    """
    Approximate the hematoxylin channel with color deconvolution.

    Parameters
    ----------
    image:
        RGB image array of shape (H, W, 3) with dtype uint8.
    logger:
        Optional logger for error reporting.

    Returns
    -------
    np.ndarray:
        Hematoxylin channel of the input image, normalized to [0, 1] with dtype float32.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        if logger is not None:
            logger.error(
                "Expected an RGB image array with shape HxWx3. Got: %s", image.shape
            )
        raise ValueError("Expected an RGB image array with shape HxWx3.")

    rgb = np.clip(image.astype(np.float32) / 255.0, 1e-6, 1.0)
    optical_density = -np.log(rgb)

    # Standard HED stain matrix for hematoxylin, eosin, and DAB (normalized to unit length)
    rgb_from_hed = np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78],
        ],
        dtype=np.float32,
    )
    hed_from_rgb = np.linalg.inv(rgb_from_hed)
    hed = optical_density @ hed_from_rgb.T
    hematoxylin = np.maximum(hed[..., 0], 0.0)

    low = float(np.percentile(hematoxylin, 1.0))
    high = float(np.percentile(hematoxylin, 99.0))
    if high <= low:
        return np.zeros_like(hematoxylin, dtype=np.float32)

    hematoxylin = np.clip((hematoxylin - low) / (high - low), 0.0, 1.0)
    return hematoxylin.astype(np.float32)


def process_hematoxylin_task(
    *,
    base_image: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Build the hematoxylin auxiliary target from the base tile.

    Parameters
    ----------
    base_image:
        RGB image array of shape (H, W, 3) with dtype uint8.
    logger:
        Optional logger for error reporting.

    Returns
    -------
    dict[str, Any]:
        A dictionary containing:
        - "image": The input RGB image normalized to [0, 1] with dtype float32.
        - "target": The extracted hematoxylin channel as a 2D array of shape (H, W) with dtype float32.
    """
    return {
        "image": base_image.astype(np.float32) / 255.0,
        "target": extract_hematoxylin_channel(base_image, logger),
    }
