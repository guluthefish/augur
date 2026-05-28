"""Tissue segmentation task processors and label metadata utilities."""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide

from augur.datasets.utils import (
    SlideRecord,
    TileRecord,
    load_tissue_mask_label_for_free_tile,
    resolve_tissue_label_metadata_path,
)


def process_tissue_segmentation_task(
    *,
    base_image: np.ndarray,
    tile_record: TileRecord,
    slide: OpenSlide,
    image_size: int,
    base_mpp: float,
    n_classes: int,
    root_dir: str,
    tissue_roi_groups: dict[str, pd.DataFrame] | None,
    slide_name_by_filename: dict[str, str],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Build the tissue segmentation task output for one base tile.

    Parameters
    ----------
    base_image:
        Raw RGB image array of the tile read from the slide (uint8).
    tile_record:
        The TileRecord describing the sampled tile.
    slide:
        The opened slide handle that ``tile_record`` was sampled from.
    image_size:
        The desired width/height of the output image and mask in pixels.
    base_mpp:
        The microns-per-pixel value at which ``base_image`` was sampled.
    n_classes:
        Total number of tissue classes for the segmentation target.
    root_dir:
        Root directory containing tissue labels under ``labels/tissues/masks``.
    tissue_roi_groups:
        Optional mapping from tissue-source slide name to ROI bounds DataFrame
        used for compositing the tissue mask for the tile.
    slide_name_by_filename:
        Mapping from slide filename (basename of ``slide_path``) to the
        precomputed ``slide_name`` stored in the tissue slide metadata. Built
        once from the slide metadata so this hot path does no parsing.
    logger:
        Optional logger for warnings/errors.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - "image": the resized tile image as float32 in [0, 1].
        - "target": one-hot tissue mask of shape (n_classes, image_size, image_size).
    """
    resized_image = cv2.resize(  # pylint: disable=no-member
        base_image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
    )

    slide_name = slide_name_by_filename.get(
        os.path.basename(tile_record.slide_path), ""
    )
    slide_rois = (
        tissue_roi_groups.get(slide_name)
        if tissue_roi_groups is not None and slide_name
        else None
    )
    target_mask = load_tissue_mask_label_for_free_tile(
        root_dir=root_dir,
        tile_record=tile_record,
        slide=slide,
        base_mpp=base_mpp,
        output_size=image_size,
        slide_name=slide_name,
        slide_rois=slide_rois,
        logger=logger,
    )

    target_one_hot = np.zeros(
        (n_classes, image_size, image_size),
        dtype=np.uint8,
    )
    for class_idx in range(n_classes):
        target_one_hot[class_idx] = (target_mask == class_idx).astype(np.uint8)

    return {
        "image": resized_image.astype(np.float32) / 255.0,
        "target": target_one_hot,
    }


def load_tissue_metadata(
    root_dir: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    """
    Load tissue semantic segmentation ground-truth codes, slide metadata, and ROI bounds.

    Parameters
    ----------
    root_dir:
        Root directory containing the tissue metadata files (resolved via
        :func:`augur.datasets.utils.resolve_tissue_label_metadata_path`).
    logger:
        Optional logger for diagnostic output.

    Returns
    -------
    tuple[dict[str, str], pd.DataFrame, pd.DataFrame]
        Mapping from label name to ground-truth code, the slide metadata
        DataFrame, and the ROI bounds DataFrame.
    """
    gt_codes_path, roi_bounds_path, slide_metadata_path = (
        resolve_tissue_label_metadata_path(root_dir=root_dir, logger=logger)
    )

    if not os.path.exists(slide_metadata_path):
        raise FileNotFoundError(f"Slide metadata file not found: {slide_metadata_path}")
    if not os.path.exists(roi_bounds_path):
        raise FileNotFoundError(f"ROI bounds file not found: {roi_bounds_path}")
    if not os.path.exists(gt_codes_path):
        raise FileNotFoundError(f"Ground truth codes file not found: {gt_codes_path}")

    gt_codes_df = pd.read_csv(gt_codes_path, sep="\t", dtype=str)
    gt_codes_df = gt_codes_df.rename(columns={"label": "label", "GT_code": "gt_code"})
    gt_codes = dict(zip(gt_codes_df["label"], gt_codes_df["gt_code"]))

    slide_df = pd.read_csv(slide_metadata_path, dtype=str)
    slide_df = slide_df.rename(
        columns={
            slide_df.columns[0]: "submitter_id",
            "name": "filename",
            "_id": "tissue_item_id",
        }
    )
    required_slide_columns = {"submitter_id", "filename", "slide_name", "magnification"}
    missing_slide_columns = required_slide_columns.difference(slide_df.columns)
    if missing_slide_columns:
        raise ValueError(
            "Tissue slide metadata is missing required columns: "
            f"{sorted(missing_slide_columns)}"
        )

    slide_df["submitter_id"] = slide_df["submitter_id"].astype(str).str.strip()
    slide_df["filename"] = slide_df["filename"].astype(str).str.strip()
    slide_df["slide_name"] = slide_df["slide_name"].astype(str).str.strip()

    roi_df = pd.read_csv(roi_bounds_path, dtype=str)
    roi_df = roi_df.rename(columns={roi_df.columns[0]: "slide_name"})
    required_roi_columns = {"slide_name", "xmin", "ymin", "xmax", "ymax"}
    missing_roi_columns = required_roi_columns.difference(roi_df.columns)
    if missing_roi_columns:
        raise ValueError(
            f"Tissue ROI bounds file is missing required columns: {sorted(missing_roi_columns)}"
        )

    roi_df["slide_name"] = roi_df["slide_name"].astype(str).str.strip()
    for column in ("xmin", "ymin", "xmax", "ymax"):
        roi_df[column] = pd.to_numeric(roi_df[column], errors="coerce")
    if roi_df[["xmin", "ymin", "xmax", "ymax"]].isna().any().any():
        raise ValueError("Tissue ROI bounds file contains invalid coordinates.")

    if logger is not None:
        logger.info(
            "Loaded tissue metadata: %d slide rows, %d ROI rows, %d GT codes.",
            len(slide_df),
            len(roi_df),
            len(gt_codes),
        )
    return gt_codes, slide_df, roi_df


def build_tissue_roi_groups(roi_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group tissue ROI rows by slide_name for per-tile mask compositing."""
    return {
        str(slide_name): group.reset_index(drop=True)
        for slide_name, group in roi_df.groupby("slide_name", sort=False)
    }


def compute_labeled_submitter_ids(
    slide_records: Sequence[SlideRecord],
    slide_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> tuple[set[str], set[str]]:
    """
    Return labelled submitter IDs and slide paths matched against tissue segmentation metadata.

    Parameters
    ----------
    slide_records:
        Slide records (e.g., from a TCGA manifest) to filter against tissue segmentation labels.
    slide_df:
        Tissue slide metadata DataFrame as returned by :func:`load_tissue_metadata`.
    logger:
        Optional logger for diagnostic output.

    Returns
    -------
    tuple[set[str], set[str]]
        Set of labelled submitter IDs and set of labelled slide paths.
    """
    expected_submitter_by_filename = dict(
        zip(slide_df["filename"], slide_df["submitter_id"])
    )

    labeled_submitter_ids: set[str] = set()
    labeled_slide_paths: set[str] = set()
    for slide_record in slide_records:
        filename = os.path.basename(slide_record.slide_path)
        expected_submitter_id = expected_submitter_by_filename.get(filename)
        if expected_submitter_id == slide_record.submitter_id:
            labeled_submitter_ids.add(slide_record.submitter_id)
            labeled_slide_paths.add(slide_record.slide_path)

    if logger is not None:
        logger.info(
            "Matched %d tissue-labelled slide(s) from %d manifest slide(s) "
            "(%d unique labelled submitter(s)).",
            len(labeled_slide_paths),
            len(slide_records),
            len(labeled_submitter_ids),
        )
    return labeled_submitter_ids, labeled_slide_paths
