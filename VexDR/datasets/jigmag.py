"""JigMag task processors for TCGA histology tiles."""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Sequence

import numpy as np
from openslide import OpenSlide

from VexDR.datasets.utils import read_tile_at_mpp


def build_jigmag_permutations(num_tiles: int) -> list[tuple[int, ...]]:
    """Return all tile permutations for a JigMag puzzle."""
    return list(itertools.permutations(range(num_tiles)))


def process_jigmag_task(
    *,
    slide: OpenSlide,
    center_x: int,
    center_y: int,
    output_size: int,
    target_mpps: Sequence[float],
    rng: random.Random,
    logger: logging.Logger | None = None,
    permutations: Sequence[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """
    Build a 2x2 multi-mpp puzzle and predict its permutation index.

    Parameters
    ----------
    slide:
        The whole slide image to read tiles from.
    center_x:
        The x-coordinate of the center of the puzzle in the slide's coordinate space.
    center_y:
        The y-coordinate of the center of the puzzle in the slide's coordinate space.
    output_size:
        The width and height of the output puzzle image in pixels.
    target_mpps:
        The mpp values for each tile in the puzzle, in the order they will be read.
    rng:
        A random number generator for selecting the permutation.
    logger:
        A logger for debug messages. If None, no logging will be performed.
    permutations:
        A list of tile permutations to choose from. If None, all possible permutations will be used.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - "image": The puzzle image as a numpy array of shape (output_size, output_size, 3) with pixel values in [0, 1].
        - "target": The index of the correct permutation in the list of permutations.
        - "permutation": The actual permutation of tile indices used in the puzzle.
        - "mpps": The mpp values for each tile in the order they were read.
    """
    if len(target_mpps) != 4:
        raise ValueError(
            "JigMag currently expects exactly 4 mpp values for a 2x2 puzzle."
        )

    tile_size = output_size // 2
    tiles = [
        read_tile_at_mpp(
            slide,
            center_x=center_x,
            center_y=center_y,
            output_size=tile_size,
            target_mpp=float(target_mpp),
            logger=logger,
        )
        for target_mpp in target_mpps
    ]

    all_permutations = (
        list(permutations)
        if permutations is not None
        else build_jigmag_permutations(len(tiles))
    )
    permutation_index = rng.randrange(len(all_permutations))
    permutation = all_permutations[permutation_index]

    puzzle = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    ordered_tiles = [tiles[idx] for idx in permutation]
    puzzle[:tile_size, :tile_size] = ordered_tiles[0]
    puzzle[:tile_size, tile_size:] = ordered_tiles[1]
    puzzle[tile_size:, :tile_size] = ordered_tiles[2]
    puzzle[tile_size:, tile_size:] = ordered_tiles[3]

    return {
        "image": puzzle.astype(np.float32) / 255.0,
        "target": permutation_index,
        "permutation": permutation,
        "mpps": tuple(float(target_mpp) for target_mpp in target_mpps),
    }
