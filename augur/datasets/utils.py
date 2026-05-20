"""Utility functions for handling manifest paths in Augur datasets."""

from dataclasses import dataclass
import logging
import math
import os
import random
from typing import Sequence

import cv2
import numpy as np
from openslide import OpenSlide
import pandas as pd
import torch


@dataclass(frozen=True)
class SlideRecord:
    """Metadata required to locate one WSI."""

    slide_id: str
    submitter_id: str
    slide_path: str


@dataclass(frozen=True)
class TileRecord:
    """One sampled tile location from a WSI."""

    slide_id: str
    submitter_id: str
    slide_path: str
    x: int
    y: int
    level: int
    size: int
    roi_name: str | None = None
    roi_xmin: int | None = None
    roi_ymin: int | None = None
    roi_xmax: int | None = None
    roi_ymax: int | None = None


def _resolve_path_from_atlas(
    atlas_path: str,
    entry_key: str,
    logger: logging.Logger | None = None,
) -> str:
    """
    Resolve a file path from an atlas file given an entry key,
    with optional validation against a needed path.
    """
    if not os.path.exists(atlas_path):
        if logger:
            logger.error("Atlas file not found at %s", atlas_path)
        raise FileNotFoundError(f"Atlas file not found at {atlas_path}")

    atlas_df = pd.read_table(atlas_path, index_col=0, dtype=str)
    if entry_key not in atlas_df.index:
        if logger:
            logger.error("Atlas file is missing the '%s' entry.", entry_key)
        raise ValueError(f"Atlas file is missing the '{entry_key}' entry.")
    resolved_path = str(atlas_df.loc[entry_key, "path"]).strip()
    if resolved_path is None or not os.path.exists(resolved_path):
        if logger:
            logger.error(
                "Resolved path for '%s' not found: %s", entry_key, resolved_path
            )
        raise FileNotFoundError(
            f"Resolved path for '{entry_key}' not found: {resolved_path}"
        )
    return resolved_path


def resolve_manifest_path(
    root_dir: str, manifest_path: str | None, logger: logging.Logger | None = None
) -> str:
    """
    Resolve the downloaded manifest path from an explicit path or atlas.

    Parameters
    ----------
    root_dir:
        Root directory containing the "atlases/manifest_atlas.txt" file if manifest_path is not provided.
    manifest_path:
        Optional explicit path to the manifest file.
        If provided, this path will be used directly if it exists.
        If None, the function will attempt to resolve the manifest path from the atlas.
    logger:
        Optional logger for logging errors. If None, no logging will be performed.

    Returns
    -------
    str
        The resolved manifest path.

    Raises
    ------
    FileNotFoundError
        If the manifest file cannot be found at the provided path or resolved from the atlas.
    ValueError
        If the manifest atlas is missing required entries or if the manifest_path is invalid.
    """
    if manifest_path is not None:
        if os.path.exists(manifest_path):
            return manifest_path

        if logger:
            logger.error("Manifest file not found: %s", manifest_path)
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    atlas_path = os.path.join(root_dir, "atlases", "manifest_atlas.txt")
    manifest_path = _resolve_path_from_atlas(
        atlas_path, entry_key="manifest_downloaded", logger=logger
    )

    return manifest_path


def resolve_slide_main_label_path(
    root_dir: str,
    task: str,
    labels_path: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Resolve the slide-level main-task label table path.

    The atlas at ``root_dir/atlases/slide_main_atlas.txt`` maps a ``task`` key
    (currently ``"subtyping"``) to a label table path.
    """
    if labels_path is not None:
        if os.path.exists(labels_path):
            return labels_path
        if logger:
            logger.error("Slide main labels file not found: %s", labels_path)
        raise FileNotFoundError(f"Slide main labels file not found: {labels_path}")

    atlas_path = os.path.join(root_dir, "atlases", "slide_main_atlas.txt")
    return _resolve_path_from_atlas(atlas_path, entry_key=task, logger=logger)


def resolve_slide_pretext_label_path(
    root_dir: str,
    task: str,
    labels_path: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Resolve the slide-level pretext-task label table path.

    The atlas at ``root_dir/atlases/slide_pretext_atlas.txt`` maps a ``task``
    key (e.g. ``"sbs_regression"``, ``"sbs_thresholded_multilabel"``,
    ``"sbs_ranked_multilabel"``) to a label table path.
    """

    if labels_path is not None:
        if os.path.exists(labels_path):
            return labels_path
        if logger:
            logger.error("Slide pretext labels file not found: %s", labels_path)
        raise FileNotFoundError(f"Slide pretext labels file not found: {labels_path}")

    atlas_path = os.path.join(root_dir, "atlases", "slide_pretext_atlas.txt")
    return _resolve_path_from_atlas(atlas_path, entry_key=task, logger=logger)


def resolve_tissue_label_metadata_path(
    root_dir: str,
    gt_path: str | None = None,
    roi_path: str | None = None,
    metadata_path: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str, str, str]:
    """
    Resolve the downloaded tissue label metadata paths from explicit paths or atlas.

    Parameters
    ----------
    root_dir:
        Root directory containing the "atlases/tissues_label_atlas.txt" file if any of the paths are not provided.
    gt_path:
        Optional explicit path to the ground truth labels. If None, it will be resolved from the atlas.
    roi_path:
        Optional explicit path to the ROI bounds. If None, it will be resolved from the atlas.
    metadata_path:
        Optional explicit path to the slide metadata. If None, it will be resolved from the atlas.
    logger:
        Optional logger for logging errors. If None, no logging will be performed.

    Returns
    -------
    tuple[str, str, str]
        The resolved paths for the ground truth labels, ROI bounds, and slide metadata.
    """
    atlas_path = os.path.join(root_dir, "atlases", "tissues_label_atlas.txt")

    if gt_path is not None:
        if not os.path.exists(gt_path):
            if logger:
                logger.error("Ground truth labels file not found: %s", gt_path)
            raise FileNotFoundError(f"Ground truth labels file not found: {gt_path}")
    if roi_path is not None:
        if not os.path.exists(roi_path):
            if logger:
                logger.error("ROI bounds file not found: %s", roi_path)
            raise FileNotFoundError(f"ROI bounds file not found: {roi_path}")
    if metadata_path is not None:
        if not os.path.exists(metadata_path):
            if logger:
                logger.error("Slide metadata file not found: %s", metadata_path)
            raise FileNotFoundError(f"Slide metadata file not found: {metadata_path}")

    gt_path = gt_path or _resolve_path_from_atlas(
        atlas_path, entry_key="groundtruth_codes", logger=logger
    )
    roi_path = roi_path or _resolve_path_from_atlas(
        atlas_path, entry_key="roi_bounds", logger=logger
    )
    metadata_path = metadata_path or _resolve_path_from_atlas(
        atlas_path, entry_key="slide_metadata", logger=logger
    )

    return gt_path, roi_path, metadata_path


def load_slide_records(
    *, manifest_path: str, ordered_data_dir: str, max_slides: int | None = None
) -> list[SlideRecord]:
    """
    Load slide image metadata from a manifest and resolve file paths.

    Parameters
    ----------
    manifest_path:
        Path to the manifest file.
    ordered_data_dir:
        Directory containing the ordered data.
    max_slides:
        Maximum number of slides to load. If None, all slides are loaded.

    Returns
    -------
    list[SlideRecord]
        List of slide records.

    Raises
    ------
    RuntimeError
        If no slide image rows are found in the manifest.
    """
    manifest_df = pd.read_table(manifest_path, index_col=0, dtype=str)
    image_mask = (
        manifest_df["subfolder"].fillna("").eq("images")
        | manifest_df["data_format"].fillna("").str.upper().isin({"SVS", "TIFF", "TIF"})
        | manifest_df["filename"]
        .fillna("")
        .str.lower()
        .str.endswith((".svs", ".tif", ".tiff"))
    )
    image_df = manifest_df.loc[image_mask].copy()
    if image_df.empty:
        raise RuntimeError(
            f"No slide image rows were found in manifest: {manifest_path}"
        )

    records: list[SlideRecord] = []
    for file_id, row in image_df.iterrows():
        submitter_id = str(row["submitter_id"]).strip()
        subfolder = str(row.get("subfolder", "images")).strip() or "images"
        filename = str(row["filename"]).strip()
        slide_path = os.path.join(
            ordered_data_dir, submitter_id, subfolder, str(file_id), filename
        )
        if os.path.exists(slide_path):
            records.append(
                SlideRecord(
                    slide_id=str(file_id),
                    submitter_id=submitter_id,
                    slide_path=slide_path,
                )
            )

    if max_slides is not None:
        records = records[:max_slides]
    return records


def _three_way_submitter_counts(
    n: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    """Split ``n`` submitters into 3 nonneg ints approximating the given fractions."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        if val_fraction >= test_fraction:
            return 1, 1, 0
        return 1, 0, 1

    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))

    n_train = min(max(n_train, 1), n - 2)
    n_val_min = 1 if val_fraction > 0 else 0
    n_val = min(max(n_val, n_val_min), n - n_train - 1)
    n_test = max(n - n_train - n_val, 1)
    return n_train, n_val, n_test


def split_slide_records_with_budget(
    slides: Sequence[SlideRecord],
    *,
    labeled_submitter_ids: set[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    train_budget: float,
    val_budget: float,
    test_budget: float,
    seed: int,
    logger: logging.Logger | None = None,
) -> dict[str, list[SlideRecord]]:
    """
    Patient-level split that places labeled submitters by budget and fills
    each split with unlabeled submitters up to the fraction-derived total.

    Labeled submitters are distributed among train/val/test by
    ``train_budget``/``val_budget``/``test_budget``. Unlabeled submitters then
    fill each split so the combined submitter count approximates
    ``train_fraction``/``val_fraction``/``test_fraction`` of the total submitter
    count. If a split's labeled budget already exceeds its fraction-derived
    target, the unlabeled count for that split is clamped to 0 and a warning is
    logged.
    """
    submitters_to_slides: dict[str, list[SlideRecord]] = {}
    for slide in slides:
        submitters_to_slides.setdefault(slide.submitter_id, []).append(slide)

    submitter_ids = list(submitters_to_slides)
    labeled_ids = [sid for sid in submitter_ids if sid in labeled_submitter_ids]
    unlabeled_ids = [sid for sid in submitter_ids if sid not in labeled_submitter_ids]

    rng = random.Random(seed)
    rng.shuffle(labeled_ids)
    rng_u = random.Random(seed + 1)
    rng_u.shuffle(unlabeled_ids)

    n_labeled = len(labeled_ids)
    n_unlabeled = len(unlabeled_ids)
    n_total = n_labeled + n_unlabeled

    labeled_train_n, labeled_val_n, labeled_test_n = _three_way_submitter_counts(
        n_labeled, train_budget, val_budget, test_budget
    )

    target_train = int(round(n_total * train_fraction))
    target_val = int(round(n_total * val_fraction))
    target_test = n_total - target_train - target_val

    unlabeled_targets: list[int] = []
    for split_name, target, labeled_n in (
        ("train", target_train, labeled_train_n),
        ("val", target_val, labeled_val_n),
        ("test", target_test, labeled_test_n),
    ):
        u = target - labeled_n
        if u < 0:
            if logger is not None:
                logger.warning(
                    "%s labeled budget allocates %d submitter(s) but fraction "
                    "target is only %d. Clamping unlabeled count to 0.",
                    split_name,
                    labeled_n,
                    target,
                )
            u = 0
        unlabeled_targets.append(u)

    total_unlabeled_target = sum(unlabeled_targets)
    if total_unlabeled_target > n_unlabeled and total_unlabeled_target > 0:
        scale = n_unlabeled / total_unlabeled_target
        scaled = [int(math.floor(u * scale)) for u in unlabeled_targets]
        remainder = n_unlabeled - sum(scaled)
        fractional = [
            (unlabeled_targets[i] * scale) - scaled[i] for i in range(3)
        ]
        order = sorted(range(3), key=lambda i: -fractional[i])
        for i in order[:remainder]:
            scaled[i] += 1
        unlabeled_targets = scaled

    u_train, u_val, u_test = unlabeled_targets

    train_labeled = labeled_ids[:labeled_train_n]
    val_labeled = labeled_ids[labeled_train_n : labeled_train_n + labeled_val_n]
    test_labeled = labeled_ids[
        labeled_train_n + labeled_val_n : labeled_train_n + labeled_val_n + labeled_test_n
    ]

    train_unlabeled = unlabeled_ids[:u_train]
    val_unlabeled = unlabeled_ids[u_train : u_train + u_val]
    test_unlabeled = unlabeled_ids[u_train + u_val : u_train + u_val + u_test]

    train_ids = set(train_labeled) | set(train_unlabeled)
    val_ids = set(val_labeled) | set(val_unlabeled)
    test_ids = set(test_labeled) | set(test_unlabeled)

    return {
        "train": [slide for slide in slides if slide.submitter_id in train_ids],
        "val": [slide for slide in slides if slide.submitter_id in val_ids],
        "test": [slide for slide in slides if slide.submitter_id in test_ids],
    }


def split_slide_records(
    slides: Sequence[SlideRecord],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[SlideRecord]]:
    """
    Perform a patient-level split using submitter_id to avoid leakage.

    Parameters
    ----------
    slides:
        List of slide records to split.
    train_fraction:
        Fraction of submitters to include in the training set.
    val_fraction:
        Fraction of submitters to include in the validation set.
    test_fraction:
        Fraction of submitters to include in the test set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, list[SlideRecord]]
        Dictionary with keys "train", "val", and "test" dict to lists of slide records.
    """
    slides_by_submitter: dict[str, list[SlideRecord]] = {}
    for slide in slides:
        slides_by_submitter.setdefault(slide.submitter_id, []).append(slide)

    submitter_ids = list(slides_by_submitter)
    rng = random.Random(seed)
    rng.shuffle(submitter_ids)

    n_submitters = len(submitter_ids)
    if n_submitters == 1:
        n_train, n_val, n_test = 1, 0, 0
    elif n_submitters == 2:
        n_train = 1
        if val_fraction >= test_fraction:
            n_val, n_test = 1, 0
        else:
            n_val, n_test = 0, 1
    else:
        n_train = int(round(n_submitters * train_fraction))
        n_val = int(round(n_submitters * val_fraction))

        n_train = min(max(n_train, 1), n_submitters - 2)
        n_val_min = 1 if val_fraction > 0 else 0
        n_val = min(max(n_val, n_val_min), n_submitters - n_train - 1)
        n_test = max(n_submitters - n_train - n_val, 1)

    train_ids = set(submitter_ids[:n_train])
    val_ids = set(submitter_ids[n_train : n_train + n_val])
    test_ids = set(submitter_ids[n_train + n_val : n_train + n_val + n_test])

    return {
        "train": [slide for slide in slides if slide.submitter_id in train_ids],
        "val": [slide for slide in slides if slide.submitter_id in val_ids],
        "test": [slide for slide in slides if slide.submitter_id in test_ids],
    }


def scaled_thumbnail_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """
    Scale thumbnail dimensions while preserving aspect ratio.

    Parameters
    ----------
    width:
        Original width of the thumbnail.
    height:
        Original height of the thumbnail.
    max_size:
        Maximum size for the longest dimension.

    Returns
    -------
    tuple[int, int]
        Scaled width and height.
    """
    longest = max(width, height)
    scale = min(max_size / longest, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def compute_tissue_mask(
    thumbnail: np.ndarray,
    *,
    white_threshold: float,
) -> np.ndarray:
    """
    Heuristic thumbnail tissue mask that filters bright background.

    Parameters
    ----------
    thumbnail:
        RGB thumbnail image as a uint8 HWC array.
    white_threshold:
        Intensity threshold in [0, 1] to separate white background from tissue.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates tissue regions.
    """
    intensity = thumbnail.astype(np.float32).mean(axis=2) / 255.0
    chroma = (
        thumbnail.astype(np.float32).max(axis=2)
        - thumbnail.astype(np.float32).min(axis=2)
    ) / 255.0
    return (intensity < white_threshold) & (chroma > 0.001)


def as_image_tensor(
    image: np.ndarray | torch.Tensor, logger: logging.Logger | None = None
) -> torch.Tensor:
    """
    Convert an HWC RGB image to ``float32`` CHW tensor.

    Parameters
    ----------
    image:
        Input image as a NumPy array or PyTorch tensor.
    logger:
        Optional logger for error messages.

    Returns
    -------
    torch.Tensor
        Converted image tensor.
    """
    if isinstance(image, torch.Tensor):
        return image.float()

    image_np = np.asarray(image, dtype=np.float32)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        if logger:
            logger.error(
                "Image must have shape HxWx3 before tensor conversion. Got: %s",
                image_np.shape,
            )
        raise ValueError("Image must have shape HxWx3 before tensor conversion.")
    return torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).float()


def as_mask_tensor(
    target: np.ndarray | torch.Tensor, logger: logging.Logger | None = None
) -> torch.Tensor:
    """
    Convert a 2D mask to ``float32`` tensor with a channel dimension in shape (C, H, W).

    Parameters
    ----------
    target:
        Input mask as a NumPy array or PyTorch tensor.
    logger:
        Optional logger for error messages.

    Returns
    -------
    torch.Tensor
        Converted mask tensor.
    """
    if isinstance(target, torch.Tensor):
        target_tensor = target.float()
    else:
        target_np = np.asarray(target, dtype=np.float32)
        if target_np.ndim == 2:
            target_np = target_np[None, ...]
        elif target_np.ndim == 3 and target_np.shape[-1] == 1:
            target_np = np.moveaxis(target_np, -1, 0)
        elif target_np.ndim != 3:
            if logger:
                logger.error(
                    "Mask target must be 2D or single-channel 3D. Got: %s",
                    target_np.shape,
                )
            raise ValueError("Mask target must be 2D or single-channel 3D.")
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_np)).float()

    if target_tensor.ndim == 2:
        target_tensor = target_tensor.unsqueeze(0)

    return target_tensor


def read_tile_from_record(slide: OpenSlide, record: TileRecord) -> np.ndarray:
    """
    Read a tile described by ``TileRecord`` as RGB uint8 HWC.

    Parameters
    ----------
    slide:
        OpenSlide object representing the whole slide image.
    record:
        TileRecord describing the location and size of the tile to read.

    Returns
    -------
    np.ndarray
        RGB tile as a uint8 HWC array.
    """
    tile = slide.read_region(
        (record.x, record.y), record.level, (record.size, record.size)
    )
    tile = tile.convert("RGB")
    return np.asarray(tile, dtype=np.uint8)


def derive_bcss_slide_name(
    slide_path_or_filename: str, submitter_id: str | None = None
) -> str:
    """
    Derive the BCSS slide name such as ``TCGA-A1-A0SK-DX1``.

    Parameters
    ----------
    slide_path_or_filename:
        Path or filename of the slide.
    submitter_id:
        Optional submitter ID to include in the slide name.

    Returns
    -------
    str
        Derived BCSS slide name.
    """
    basename = os.path.basename(str(slide_path_or_filename).strip())
    stem = os.path.splitext(basename)[0]
    stem = stem.split(".")[0]
    parts = [part for part in stem.split("-") if part]

    if len(parts) >= 6:
        return "-".join(parts[:3] + [parts[-1]])
    if submitter_id is not None and parts:
        return f"{submitter_id}-{parts[-1]}"
    return stem


def load_tissue_mask_label(
    *,
    root_dir: str,
    tile_record: TileRecord,
    slide: OpenSlide,
    base_mpp: float,
    output_size: int,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Load a BCSS tissue mask crop aligned to a sampled tile.

    Areas that fall outside the BCSS ROI bounds are padded with label ``0``
    (``outside_roi``) so the returned mask always matches the tile dimensions.

    Parameters
    ----------
    root_dir:
        Root directory containing the "labels/tissues/masks" folder with pre-cropped BCSS tissue masks.
    tile_record:
        TileRecord describing the location and size of the tile to load.
    slide:
        OpenSlide object representing the whole slide image.
    base_mpp:
        Base microns per pixel (MPP) for the tile.
    output_size:
        Desired output size of the mask.
    logger:
        Optional logger for logging warnings and errors.

    Returns
    -------
    np.ndarray
        array of shape (output_size, output_size) with dtype uint8 containing integer class labels for each pixel.
        0 indicates areas outside the BCSS ROI, and other values indicate tissue classes within the ROI.
    """
    if (
        tile_record.roi_xmin is None
        or tile_record.roi_ymin is None
        or tile_record.roi_xmax is None
        or tile_record.roi_ymax is None
    ):
        if logger is not None:
            logger.warning(
                "TileRecord is missing roi bounds required for BCSS mask loading. Returning empty mask."
            )
        return np.zeros((output_size, output_size), dtype=np.uint8)

    if base_mpp <= 0:
        raise ValueError("base_mpp must be positive.")

    slide_name = derive_bcss_slide_name(
        tile_record.slide_path, tile_record.submitter_id
    )
    mask_filename = (
        f"{slide_name}_xmin{int(tile_record.roi_xmin)}_ymin{int(tile_record.roi_ymin)}"
        f"_MPP-{float(base_mpp):.4f}.png"
    )
    mask_path = os.path.join(root_dir, "labels", "tissues", "masks", mask_filename)
    if not os.path.exists(mask_path):
        if logger is not None:
            logger.error("BCSS tissue mask not found: %s", mask_path)
        raise FileNotFoundError(f"BCSS tissue mask not found: {mask_path}")

    mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
    if mask_np is None:
        if logger is not None:
            logger.error("Failed to read BCSS tissue mask: %s", mask_path)
        raise FileNotFoundError(f"Failed to read BCSS tissue mask: {mask_path}")

    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]

    roi_width_l0 = max(int(tile_record.roi_xmax) - int(tile_record.roi_xmin), 1)
    roi_height_l0 = max(int(tile_record.roi_ymax) - int(tile_record.roi_ymin), 1)
    scale_x = mask_np.shape[1] / roi_width_l0
    scale_y = mask_np.shape[0] / roi_height_l0

    downsample = float(slide.level_downsamples[tile_record.level])
    tile_extent_l0 = tile_record.size * downsample
    target_width = max(int(round(tile_extent_l0 * scale_x)), 1)
    target_height = max(int(round(tile_extent_l0 * scale_y)), 1)

    mask_left = int(round((tile_record.x - int(tile_record.roi_xmin)) * scale_x))
    mask_top = int(round((tile_record.y - int(tile_record.roi_ymin)) * scale_y))
    mask_right = mask_left + target_width
    mask_bottom = mask_top + target_height

    padded_crop = np.zeros((target_height, target_width), dtype=np.uint8)

    src_left = max(mask_left, 0)
    src_top = max(mask_top, 0)
    src_right = min(mask_right, mask_np.shape[1])
    src_bottom = min(mask_bottom, mask_np.shape[0])

    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - mask_left
        dst_top = src_top - mask_top
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        padded_crop[dst_top:dst_bottom, dst_left:dst_right] = mask_np[
            src_top:src_bottom, src_left:src_right
        ]

    if padded_crop.shape != (tile_record.size, tile_record.size):
        padded_crop = cv2.resize(  # pylint: disable=no-member
            padded_crop,
            (tile_record.size, tile_record.size),
            interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
        )

    padded_crop = cv2.resize(  # pylint: disable=no-member
        padded_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
    )

    return np.asarray(padded_crop, dtype=np.uint8)


def load_tissue_mask_label_for_free_tile(
    *,
    root_dir: str,
    tile_record: TileRecord,
    slide: OpenSlide,
    base_mpp: float,
    output_size: int,
    slide_name: str,
    slide_rois: "pd.DataFrame | None",
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Build a BCSS tissue mask aligned to a free-sampled tile by compositing every
    ROI that overlaps the tile. Pixels outside any ROI stay at class ``0``
    (``outside_roi``).

    Parameters
    ----------
    slide_name:
        BCSS slide name (as produced by :func:`derive_bcss_slide_name`) used to
        look up per-ROI mask files.
    slide_rois:
        DataFrame of ROI rows for this slide (columns ``xmin``, ``ymin``,
        ``xmax``, ``ymax``). When ``None`` or empty, an all-zero mask is
        returned.
    """
    out_mask = np.zeros((output_size, output_size), dtype=np.uint8)
    if slide_rois is None or slide_rois.empty:
        return out_mask
    if base_mpp <= 0:
        raise ValueError("base_mpp must be positive.")

    downsample = float(slide.level_downsamples[tile_record.level])
    tile_extent_l0 = tile_record.size * downsample
    if tile_extent_l0 <= 0:
        return out_mask
    tile_x_l0 = float(tile_record.x)
    tile_y_l0 = float(tile_record.y)
    tile_right_l0 = tile_x_l0 + tile_extent_l0
    tile_bottom_l0 = tile_y_l0 + tile_extent_l0
    output_per_l0 = output_size / tile_extent_l0

    for roi_row in slide_rois.itertuples(index=False):
        roi_xmin = int(roi_row.xmin)
        roi_ymin = int(roi_row.ymin)
        roi_xmax = int(roi_row.xmax)
        roi_ymax = int(roi_row.ymax)

        ix_lo = max(tile_x_l0, float(roi_xmin))
        iy_lo = max(tile_y_l0, float(roi_ymin))
        ix_hi = min(tile_right_l0, float(roi_xmax))
        iy_hi = min(tile_bottom_l0, float(roi_ymax))
        if ix_hi <= ix_lo or iy_hi <= iy_lo:
            continue

        mask_filename = (
            f"{slide_name}_xmin{roi_xmin}_ymin{roi_ymin}"
            f"_MPP-{float(base_mpp):.4f}.png"
        )
        mask_path = os.path.join(root_dir, "labels", "tissues", "masks", mask_filename)
        if not os.path.exists(mask_path):
            if logger is not None:
                logger.warning(
                    "BCSS tissue mask missing for free-sampled tile composite: %s",
                    mask_path,
                )
            continue
        mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
        if mask_np is None:
            if logger is not None:
                logger.warning("Failed to read BCSS tissue mask: %s", mask_path)
            continue
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        roi_width_l0 = max(roi_xmax - roi_xmin, 1)
        roi_height_l0 = max(roi_ymax - roi_ymin, 1)
        scale_x = mask_np.shape[1] / roi_width_l0
        scale_y = mask_np.shape[0] / roi_height_l0

        src_left = int(round((ix_lo - roi_xmin) * scale_x))
        src_top = int(round((iy_lo - roi_ymin) * scale_y))
        src_right = int(round((ix_hi - roi_xmin) * scale_x))
        src_bottom = int(round((iy_hi - roi_ymin) * scale_y))
        src_left = max(0, min(src_left, mask_np.shape[1]))
        src_right = max(src_left, min(src_right, mask_np.shape[1]))
        src_top = max(0, min(src_top, mask_np.shape[0]))
        src_bottom = max(src_top, min(src_bottom, mask_np.shape[0]))
        if src_right <= src_left or src_bottom <= src_top:
            continue

        dst_left = int(round((ix_lo - tile_x_l0) * output_per_l0))
        dst_top = int(round((iy_lo - tile_y_l0) * output_per_l0))
        dst_right = int(round((ix_hi - tile_x_l0) * output_per_l0))
        dst_bottom = int(round((iy_hi - tile_y_l0) * output_per_l0))
        dst_left = max(0, min(dst_left, output_size))
        dst_right = max(dst_left, min(dst_right, output_size))
        dst_top = max(0, min(dst_top, output_size))
        dst_bottom = max(dst_top, min(dst_bottom, output_size))
        if dst_right <= dst_left or dst_bottom <= dst_top:
            continue

        crop = mask_np[src_top:src_bottom, src_left:src_right]
        resized = cv2.resize(  # pylint: disable=no-member
            crop,
            (dst_right - dst_left, dst_bottom - dst_top),
            interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
        )
        out_mask[dst_top:dst_bottom, dst_left:dst_right] = resized

    return out_mask


def get_slide_mpp(slide: OpenSlide, logger: logging.Logger | None = None) -> float:
    """
    Return the average microns-per-pixel for a slide.

    Parameters
    ----------
    slide:
        OpenSlide object representing the whole slide image.
    logger:
        Optional logger for error messages.

    Returns
    -------
    float
        Average microns-per-pixel for the slide.
    """
    mpp_x = slide.properties.get("openslide.mpp-x")
    mpp_y = slide.properties.get("openslide.mpp-y")
    if mpp_x is None and mpp_y is None:
        if logger:
            logger.error("Slide is missing openslide.mpp-x / openslide.mpp-y metadata.")
        raise ValueError("Slide is missing microns-per-pixel metadata.")

    values = []
    if mpp_x is not None:
        values.append(float(mpp_x))
    if mpp_y is not None:
        values.append(float(mpp_y))
    return float(sum(values) / len(values))


def tile_record_center_l0(slide: OpenSlide, record: TileRecord) -> tuple[int, int]:
    """
    Return the center of a ``TileRecord`` in level-0 coordinates.

    Parameters
    ----------
    slide:
        OpenSlide object representing the whole slide image.
    record:
        TileRecord describing the location and size of the tile.

    Returns
    -------
    tuple[int, int]
        Center coordinates (x, y) of the tile in level-0 coordinates.
    """
    downsample = float(slide.level_downsamples[record.level])
    extent_l0 = record.size * downsample
    center_x = int(round(record.x + extent_l0 / 2.0))
    center_y = int(round(record.y + extent_l0 / 2.0))
    return center_x, center_y


def read_tile_at_mpp(
    slide: OpenSlide,
    *,
    center_x: int,
    center_y: int,
    output_size: int,
    target_mpp: float,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Read an RGB tile centered at a level-0 coordinate for a target ``mpp``.

    Parameters
    ----------
    slide:
        OpenSlide object representing the whole slide image.
    center_x:
        X-coordinate of the tile center in level-0 coordinates.
    center_y:
        Y-coordinate of the tile center in level-0 coordinates.
    output_size:
        Desired output size of the tile.
    target_mpp:
        Target microns-per-pixel for the tile.
    logger:
        Optional logger for error messages.

    Returns
    -------
    np.ndarray
        RGB tile as a uint8 HWC array.
    """
    slide_mpp = get_slide_mpp(slide, logger)
    target_downsample = target_mpp / slide_mpp
    level = slide.get_best_level_for_downsample(target_downsample)
    actual_downsample = float(slide.level_downsamples[level])
    read_size = max(int(round(output_size * target_downsample / actual_downsample)), 1)

    tile_extent_l0 = read_size * actual_downsample
    width_l0, height_l0 = slide.dimensions
    max_x = max(int(round(width_l0 - tile_extent_l0)), 0)
    max_y = max(int(round(height_l0 - tile_extent_l0)), 0)
    x = min(max(int(round(center_x - tile_extent_l0 / 2.0)), 0), max_x)
    y = min(max(int(round(center_y - tile_extent_l0 / 2.0)), 0), max_y)

    tile = slide.read_region((x, y), level, (read_size, read_size)).convert("RGB")
    if read_size != output_size:
        tile = tile.resize((output_size, output_size))
    return np.asarray(tile, dtype=np.uint8)


def sample_tile_records(
    slides: Sequence[SlideRecord],
    *,
    output_size: int,
    base_mpp: float,
    context_mpp: float | None,
    tiles_per_slide: int,
    min_tissue_fraction: float,
    thumbnail_max_size: int,
    white_threshold: float,
    seed: int,
    logger: logging.Logger | None = None,
) -> list[TileRecord]:
    """
    Sample base ``TileRecord`` objects from slides using thumbnail tissue masks.

    Parameters
    ----------
    slides:
        Sequence of SlideRecord objects representing the slides to sample from.
    output_size:
        Desired output size of the tiles.
    base_mpp:
        Base microns-per-pixel for the tiles.
    context_mpp:
        Optional context microns-per-pixel for the tiles.
    tiles_per_slide:
        Number of tiles to sample per slide.
    min_tissue_fraction:
        Minimum fraction of tissue required in a tile.
    thumbnail_max_size:
        Maximum size of the thumbnail for tissue mask generation.
    white_threshold:
        Threshold for white pixels in the tissue mask.
    seed:
        Random seed for reproducibility.
    logger:
        Optional logger for error messages.

    Returns
    -------
    list[TileRecord]
        List of sampled TileRecord objects.
    """
    records: list[TileRecord] = []
    total_slides = len(slides)
    for slide_index, slide_record in enumerate(slides, start=1):
        if logger and (
            slide_index == 1 or slide_index == total_slides or slide_index % 25 == 0
        ):
            logger.info(
                "Sampling thumbnail-based tiles for slide %d/%d (%s).",
                slide_index,
                total_slides,
                slide_record.slide_id,
            )
        records.extend(
            _sample_slide_tile_records(
                slide_record,
                output_size=output_size,
                base_mpp=base_mpp,
                context_mpp=context_mpp or base_mpp,
                tiles_per_slide=tiles_per_slide,
                min_tissue_fraction=min_tissue_fraction,
                thumbnail_max_size=thumbnail_max_size,
                white_threshold=white_threshold,
                seed=seed,
                logger=logger,
            )
        )
    return records


def sample_tile_record_in_roi_bounds(
    slide_record: SlideRecord,
    *,
    roi_name: str,
    roi_xmin: int,
    roi_ymin: int,
    roi_xmax: int,
    roi_ymax: int,
    output_size: int,
    base_mpp: float,
    seed: int | str,
    slide: OpenSlide | None = None,
    logger: logging.Logger | None = None,
) -> TileRecord:
    """
    Sample one base-resolution tile from within a BCSS ROI.

    When the requested tile cannot fit fully inside the ROI, the tile center is still
    sampled inside the ROI and downstream mask loading pads the out-of-ROI area with 0.

    Parameters
    ----------
    slide_record:
        SlideRecord representing the slide to sample from.
    roi_name:
        Name of the ROI for logging and record-keeping purposes.
    roi_xmin:
        Minimum x-coordinate of the ROI in level-0 coordinates.
    roi_ymin:
        Minimum y-coordinate of the ROI in level-0 coordinates.
    roi_xmax:
        Maximum x-coordinate of the ROI in level-0 coordinates.
    roi_ymax:
        Maximum y-coordinate of the ROI in level-0 coordinates.
    output_size:
        Desired output size of the tile.
    base_mpp:
        Base microns-per-pixel for the tile.
    seed:
        Random seed for reproducibility. Can be an integer or a string.
    slide:
        Optional OpenSlide object representing the whole slide image.
        If None, the function will open the slide itself and close it after sampling.
    logger:
        Optional logger for error messages.

    Returns
    -------
    TileRecord
        Sampled TileRecord object with ROI information included.
    """
    owns_slide = slide is None
    if slide is None:
        slide = OpenSlide(slide_record.slide_path)

    try:
        slide_mpp = get_slide_mpp(slide, logger)
        target_downsample = base_mpp / slide_mpp
        level = slide.get_best_level_for_downsample(target_downsample)
        actual_downsample = float(slide.level_downsamples[level])
        read_size = max(
            int(round(output_size * target_downsample / actual_downsample)), 1
        )
        tile_extent_l0 = read_size * actual_downsample

        rng = random.Random(f"{seed}:{slide_record.slide_id}:{roi_name}")
        center_x = _sample_center_in_roi(
            int(roi_xmin),
            int(roi_xmax),
            tile_extent_l0=tile_extent_l0,
            rng=rng,
        )
        center_y = _sample_center_in_roi(
            int(roi_ymin),
            int(roi_ymax),
            tile_extent_l0=tile_extent_l0,
            rng=rng,
        )

        base_record = _make_tile_record_for_mpp(
            slide_record,
            slide,
            center_x=center_x,
            center_y=center_y,
            output_size=output_size,
            target_mpp=base_mpp,
            logger=logger,
        )
        return TileRecord(
            slide_id=base_record.slide_id,
            submitter_id=base_record.submitter_id,
            slide_path=base_record.slide_path,
            x=base_record.x,
            y=base_record.y,
            level=base_record.level,
            size=base_record.size,
            roi_name=str(roi_name),
            roi_xmin=int(roi_xmin),
            roi_ymin=int(roi_ymin),
            roi_xmax=int(roi_xmax),
            roi_ymax=int(roi_ymax),
        )
    finally:
        if owns_slide:
            slide.close()


def enumerate_slide_tile_centers(
    slide_record: SlideRecord,
    *,
    output_size: int,
    context_mpp: float,
    min_tissue_fraction: float,
    thumbnail_max_size: int,
    white_threshold: float,
    stride: int | None = None,
    slide: OpenSlide | None = None,
    logger: logging.Logger | None = None,
) -> list[tuple[int, int]]:
    """Return all valid tissue tile centers for a slide in level-0 coordinates.

    Uses the slide thumbnail and a tissue heuristic to enumerate every tile-sized
    window that meets ``min_tissue_fraction``. ``stride`` controls the spacing
    between consecutive candidate tile centers in the same units as
    ``output_size`` (i.e., pixels at ``context_mpp``); when omitted it defaults
    to ``output_size`` (non-overlapping tiles). Caller can later resample from
    the returned centers to build ``TileRecord`` objects on demand.
    """
    owns_slide = slide is None
    if slide is None:
        slide = OpenSlide(slide_record.slide_path)
    try:
        slide_mpp = get_slide_mpp(slide, logger)
        context_extent_l0 = max(int(round(output_size * (context_mpp / slide_mpp))), 1)
        stride_value = output_size if stride is None else int(stride)
        if stride_value <= 0:
            raise ValueError(f"stride must be a positive integer. Got: {stride_value}")
        stride_extent_l0 = max(int(round(stride_value * (context_mpp / slide_mpp))), 1)
        thumb_size = scaled_thumbnail_size(*slide.dimensions, thumbnail_max_size)
        thumbnail = np.asarray(
            slide.get_thumbnail(thumb_size).convert("RGB"), dtype=np.uint8
        )
        tissue_mask = compute_tissue_mask(thumbnail, white_threshold=white_threshold)

        width_l0, height_l0 = slide.dimensions
        scale_x = width_l0 / thumbnail.shape[1]
        scale_y = height_l0 / thumbnail.shape[0]
        tile_w_thumb = max(int(round(context_extent_l0 / scale_x)), 1)
        tile_h_thumb = max(int(round(context_extent_l0 / scale_y)), 1)
        stride_w_thumb = max(int(round(stride_extent_l0 / scale_x)), 1)
        stride_h_thumb = max(int(round(stride_extent_l0 / scale_y)), 1)

        return _enumerate_center_candidates(
            tissue_mask,
            scale_x=scale_x,
            scale_y=scale_y,
            tile_extent_l0=context_extent_l0,
            tile_w_thumb=tile_w_thumb,
            tile_h_thumb=tile_h_thumb,
            stride_w_thumb=stride_w_thumb,
            stride_h_thumb=stride_h_thumb,
            width_l0=width_l0,
            height_l0=height_l0,
            min_tissue_fraction=min_tissue_fraction,
        )
    finally:
        if owns_slide:
            slide.close()


def _sample_slide_tile_records(
    slide_record: SlideRecord,
    *,
    output_size: int,
    base_mpp: float,
    context_mpp: float,
    tiles_per_slide: int,
    min_tissue_fraction: float,
    thumbnail_max_size: int,
    white_threshold: float,
    seed: int,
    logger: logging.Logger | None = None,
) -> list[TileRecord]:
    """Sample tile records for one slide."""
    slide = OpenSlide(slide_record.slide_path)
    try:
        centers = enumerate_slide_tile_centers(
            slide_record,
            output_size=output_size,
            context_mpp=context_mpp,
            min_tissue_fraction=min_tissue_fraction,
            thumbnail_max_size=thumbnail_max_size,
            white_threshold=white_threshold,
            slide=slide,
            logger=logger,
        )
        if not centers:
            if logger:
                logger.warning(
                    "No valid tissue regions found in thumbnail for slide %s.",
                    slide_record.slide_id,
                )
            return []

        rng = random.Random(f"{seed}:{slide_record.slide_id}")
        if len(centers) > tiles_per_slide:
            centers = rng.sample(centers, tiles_per_slide)

        return [
            _make_tile_record_for_mpp(
                slide_record,
                slide,
                center_x=center_x,
                center_y=center_y,
                output_size=output_size,
                target_mpp=base_mpp,
                logger=logger,
            )
            for center_x, center_y in centers
        ]
    finally:
        slide.close()


def _sample_center_in_roi(
    roi_min: int,
    roi_max: int,
    *,
    tile_extent_l0: float,
    rng: random.Random,
) -> int:
    """Sample a tile center inside an ROI while preferring fully-contained tiles."""
    roi_min = int(roi_min)
    roi_max = int(roi_max)
    if roi_max <= roi_min:
        return roi_min

    half_extent = tile_extent_l0 / 2.0
    center_min_if_fit = int(math.ceil(roi_min + half_extent))
    center_max_if_fit = int(math.floor(roi_max - half_extent))
    if center_min_if_fit <= center_max_if_fit:
        return rng.randint(center_min_if_fit, center_max_if_fit)

    center_min = roi_min
    center_max = max(roi_min, roi_max - 1)
    if center_min >= center_max:
        return center_min
    return rng.randint(center_min, center_max)


def _make_tile_record_for_mpp(
    slide_record: SlideRecord,
    slide: OpenSlide,
    *,
    center_x: int,
    center_y: int,
    output_size: int,
    target_mpp: float,
    logger: logging.Logger | None = None,
) -> TileRecord:
    """Create a ``TileRecord`` that reads a tile at the requested ``mpp``."""
    slide_mpp = get_slide_mpp(slide, logger)
    target_downsample = target_mpp / slide_mpp
    level = slide.get_best_level_for_downsample(target_downsample)
    actual_downsample = float(slide.level_downsamples[level])
    read_size = max(int(round(output_size * target_downsample / actual_downsample)), 1)

    tile_extent_l0 = read_size * actual_downsample
    width_l0, height_l0 = slide.dimensions
    max_x = max(int(round(width_l0 - tile_extent_l0)), 0)
    max_y = max(int(round(height_l0 - tile_extent_l0)), 0)
    x = min(max(int(round(center_x - tile_extent_l0 / 2.0)), 0), max_x)
    y = min(max(int(round(center_y - tile_extent_l0 / 2.0)), 0), max_y)

    return TileRecord(
        slide_id=slide_record.slide_id,
        submitter_id=slide_record.submitter_id,
        slide_path=slide_record.slide_path,
        x=x,
        y=y,
        level=level,
        size=read_size,
    )


def _enumerate_center_candidates(
    tissue_mask: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    tile_extent_l0: int,
    tile_w_thumb: int,
    tile_h_thumb: int,
    stride_w_thumb: int | None = None,
    stride_h_thumb: int | None = None,
    width_l0: int,
    height_l0: int,
    min_tissue_fraction: float,
) -> list[tuple[int, int]]:
    """Convert thumbnail mask regions into valid level-0 tile centers."""
    height_thumb, width_thumb = tissue_mask.shape
    stride_x = max(stride_w_thumb if stride_w_thumb is not None else tile_w_thumb, 1)
    stride_y = max(stride_h_thumb if stride_h_thumb is not None else tile_h_thumb, 1)
    half_extent = tile_extent_l0 / 2.0
    min_center_x = int(np.ceil(half_extent))
    min_center_y = int(np.ceil(half_extent))
    max_center_x = max(int(np.floor(width_l0 - half_extent)), min_center_x)
    max_center_y = max(int(np.floor(height_l0 - half_extent)), min_center_y)

    candidates: list[tuple[int, int]] = []
    x_starts = list(range(0, max(width_thumb - tile_w_thumb + 1, 1), stride_x))
    y_starts = list(range(0, max(height_thumb - tile_h_thumb + 1, 1), stride_y))
    for y_start in y_starts:
        for x_start in x_starts:
            window = tissue_mask[
                y_start : y_start + tile_h_thumb,
                x_start : x_start + tile_w_thumb,
            ]
            if window.size == 0 or float(window.mean()) < min_tissue_fraction:
                continue

            center_x_thumb = x_start + tile_w_thumb / 2.0
            center_y_thumb = y_start + tile_h_thumb / 2.0
            center_x_l0 = int(round(center_x_thumb * scale_x))
            center_y_l0 = int(round(center_y_thumb * scale_y))
            center_x_l0 = min(max(center_x_l0, min_center_x), max_center_x)
            center_y_l0 = min(max(center_y_l0, min_center_y), max_center_y)
            candidates.append((center_x_l0, center_y_l0))

    return candidates
