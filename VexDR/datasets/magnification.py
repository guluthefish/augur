"""Magnification task processors for TCGA histology tiles."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

import numpy as np
from openslide import OpenSlide

from VexDR.datasets.utils import read_tile_at_mpp


def process_magnification_task(
    *,
    slide: OpenSlide,
    center_x: int,
    center_y: int,
    output_size: int,
    target_mpps: Sequence[float],
    rng: random.Random,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Read one tile at a randomly selected mpp and predict that mpp.

    Parameters
    ----------
    slide:
        The whole slide image to read tiles from.
    center_x:
        The x-coordinate of the center of the tile in the slide's coordinate space.
    center_y:
        The y-coordinate of the center of the tile in the slide's coordinate space.
    output_size:
        The width and height of the output tile image in pixels.
    target_mpps:
        The mpp values to choose from for the tile. The target mpp will be randomly selected from this list.
    rng:
        A random number generator for selecting the target mpp.
    logger:
        A logger for debug messages. If None, no logging will be performed.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - "image": The tile image as a numpy array of shape (output_size, output_size, 3) with pixel values in [0, 1].
        - "target": The index of the target mpp in the list of target_mpps.
        - "target_mpp": The actual target mpp value used for reading the tile.
        - "mpp_candidates": A tuple of all mpp values that were candidates for selection.
    """
    if not target_mpps:
        raise ValueError("target_mpps must contain at least one value.")

    target_mpp = float(rng.choice(list(target_mpps)))
    target_index = list(target_mpps).index(target_mpp)
    image = read_tile_at_mpp(
        slide,
        center_x=center_x,
        center_y=center_y,
        output_size=output_size,
        target_mpp=target_mpp,
        logger=logger,
    )
    return {
        "image": image.astype(np.float32) / 255.0,
        "target": target_index,
        "target_mpp": target_mpp,
        "mpp_candidates": tuple(float(mpp) for mpp in target_mpps),
    }
