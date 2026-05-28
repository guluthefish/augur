"""Download BCSS ROI images, masks, and annotations from the public HistomicsTK demo."""

from __future__ import annotations

import io
import logging
import os
import subprocess
from argparse import ArgumentParser
from typing import Any, Optional

import pandas as pd
import requests
import yaml
from PIL import Image

from augur.datasets.utils import derive_tissue_slide_name
from augur.utils.logger import setup_logger

DEFAULT_API_URL = "https://demo.kitware.com/histomicstk/api/v1"
SUPPORTED_PIPELINES = ("images", "masks", "annotations")
STREAM_CHUNK_SIZE = 1024 * 1024
Image.MAX_IMAGE_PIXELS = None


def normalize_pipelines(pipelines: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalize the requested BCSS pipeline names while preserving their order.

    Parameters
    ----------
    pipelines:
        Sequence of requested BCSS download stages.

    Returns
    -------
    tuple[str, ...]
        Deduplicated pipeline names in their original order.
    """
    if not isinstance(pipelines, (list, tuple)):
        raise ValueError("pipelines must be provided as a list or tuple of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for pipeline in pipelines:
        pipeline_name = str(pipeline).strip().lower()
        if not pipeline_name:
            continue
        if pipeline_name not in SUPPORTED_PIPELINES:
            raise ValueError(
                "Unsupported BCSS pipeline requested: "
                f"{pipeline_name}. Supported values: {SUPPORTED_PIPELINES}"
            )
        if pipeline_name not in seen:
            normalized.append(pipeline_name)
            seen.add(pipeline_name)

    if not normalized:
        raise ValueError("pipelines must include at least one BCSS download stage.")

    return tuple(normalized)


def prepare_output_dirs(root_dir: str, pipelines: tuple[str, ...]) -> dict[str, str]:
    """
    Create the BCSS output folder structure under ``<root_dir>/labels/tissues``.

    Parameters
    ----------
    root_dir:
        Dataset root directory.
    pipelines:
        Requested BCSS pipeline names.

    Returns
    -------
    dict[str, str]
        Paths for the base BCSS directory and each requested pipeline directory.
    """
    bcss_dir = os.path.join(root_dir, "labels", "tissues")
    os.makedirs(bcss_dir, exist_ok=True)

    output_dirs = {"base": bcss_dir}
    for pipeline in pipelines:
        pipeline_dir = os.path.join(bcss_dir, pipeline)
        os.makedirs(pipeline_dir, exist_ok=True)
        output_dirs[pipeline] = pipeline_dir

    return output_dirs


def load_bcss_metadata(root_dir: str) -> pd.DataFrame:
    """
    Load ROI bounds and slide metadata required for BCSS downloads.

    Parameters
    ----------
    root_dir:
        Dataset root directory containing ``metadata/bcss_roi_bounds.csv`` and
        ``metadata/bcss_slide_metadata.csv``.

    Returns
    -------
    pd.DataFrame
        ROI-level metadata joined with HistomicsTK item ids.
    """
    metadata_dir = os.path.join(root_dir, "metadata")
    roi_bounds_path = os.path.join(metadata_dir, "bcss_roi_bounds.csv")
    slide_metadata_path = os.path.join(metadata_dir, "bcss_slide_metadata.csv")

    if not os.path.exists(roi_bounds_path):
        raise FileNotFoundError(f"Missing BCSS ROI bounds file: {roi_bounds_path}")
    if not os.path.exists(slide_metadata_path):
        raise FileNotFoundError(
            f"Missing BCSS slide metadata file: {slide_metadata_path}"
        )

    roi_df = pd.read_csv(roi_bounds_path, dtype={"mask_link": str})
    roi_name_col = roi_df.columns[0]
    roi_df = roi_df.rename(columns={roi_name_col: "roi_name"})

    required_roi_columns = {"roi_name", "xmin", "ymin", "xmax", "ymax", "mask_link"}
    missing_roi_columns = required_roi_columns.difference(roi_df.columns)
    if missing_roi_columns:
        raise ValueError(
            f"BCSS ROI bounds file missing columns: {sorted(missing_roi_columns)}"
        )

    roi_df["roi_name"] = roi_df["roi_name"].astype(str).str.strip()
    roi_df["slide_id"] = roi_df["roi_name"].str.slice(0, 12)
    for column in ("xmin", "ymin", "xmax", "ymax"):
        roi_df[column] = pd.to_numeric(roi_df[column], errors="coerce")

    if roi_df[["xmin", "ymin", "xmax", "ymax"]].isna().any().any():
        raise ValueError(
            f"BCSS ROI bounds file contains non-numeric ROI coordinates: {roi_bounds_path}"
        )

    slide_df = pd.read_csv(slide_metadata_path, index_col=0, dtype=str)
    slide_df.index = slide_df.index.astype(str).str.strip()
    slide_df = slide_df.rename(columns={"name": "filename", "_id": "item_id"})

    required_slide_columns = {"filename", "slide_name", "item_id", "magnification"}
    missing_slide_columns = required_slide_columns.difference(slide_df.columns)
    if missing_slide_columns:
        raise ValueError(
            f"BCSS slide metadata file missing columns: {sorted(missing_slide_columns)}"
        )

    slide_df["magnification"] = pd.to_numeric(
        slide_df["magnification"], errors="coerce"
    )

    merged_df = roi_df.merge(
        slide_df[["filename", "slide_name", "item_id", "magnification"]],
        left_on="slide_id",
        right_index=True,
        how="left",
    )

    missing_slide_ids = merged_df.loc[merged_df["item_id"].isna(), "slide_id"].unique()
    if len(missing_slide_ids) > 0:
        examples = ", ".join(missing_slide_ids[:5])
        raise ValueError(
            "Some BCSS ROIs could not be matched to HistomicsTK slide ids. "
            f"Examples: {examples}"
        )

    return merged_df


def infer_base_mpp(magnification: float) -> float:
    """
    Infer a slide's base microns-per-pixel from its reported magnification.

    Parameters
    ----------
    magnification:
        Scanner magnification (for example, 40.0).

    Returns
    -------
    float
        Approximate base mpp using the standard 40x -> 0.25 um/px convention.
    """
    if pd.isna(magnification) or float(magnification) <= 0:
        raise ValueError("magnification must be a positive number.")
    return 10.0 / float(magnification)


def build_roi_stem(roi_name: str, xmin: int, ymin: int, mpp: Optional[float]) -> str:
    """
    Build the BCSS output stem used for both image and mask PNG files.

    Parameters
    ----------
    roi_name:
        BCSS ROI name from the metadata table.
    xmin:
        Left ROI coordinate at level 0.
    ymin:
        Top ROI coordinate at level 0.
    mpp:
        Requested output microns-per-pixel.

    Returns
    -------
    str
        Filename stem without the extension.
    """
    scale_tag = "MAG-0" if mpp is None else f"MPP-{float(mpp):.4f}"
    return f"{roi_name}_xmin{xmin}_ymin{ymin}_{scale_tag}"


def open_image_from_response(
    response: requests.Response, force_rgb: bool = False
) -> Image.Image:
    """
    Open an image from a ``requests`` response object.

    Parameters
    ----------
    response:
        Successful HTTP response containing image bytes.
    force_rgb:
        Convert the image to RGB if True.

    Returns
    -------
    PIL.Image.Image
        Loaded image detached from the underlying byte stream.
    """
    with Image.open(io.BytesIO(response.content)) as image:
        if force_rgb:
            image = image.convert("RGB")
        return image.copy()


def write_response_to_path(response: requests.Response, output_path: str) -> None:
    """
    Stream an HTTP response body to disk without materializing it in memory.

    Parameters
    ----------
    response:
        Successful HTTP response to persist.
    output_path:
        Destination file path.
    """
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def require_cv2():
    """
    Import cv2 only when mask processing actually needs it.

    Returns
    -------
    module
        The imported cv2 module.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "cv2 is only required for BCSS mask resizing. "
            "Use the annotations pipeline alone, or install a working OpenCV build."
        ) from exc
    return cv2


def estimate_target_size(row: pd.Series | Any, mpp: Optional[float]) -> tuple[int, int]:
    """
    Estimate the target ROI size for a given mpp.

    Parameters
    ----------
    row:
        ROI metadata row containing ROI coordinates and magnification.
    mpp:
        Requested output microns-per-pixel.

    Returns
    -------
    tuple[int, int]
        Target ``(width, height)`` in pixels.
    """
    width = max(int(round(float(row.xmax) - float(row.xmin))), 1)
    height = max(int(round(float(row.ymax) - float(row.ymin))), 1)
    if mpp is None:
        return width, height

    base_mpp = infer_base_mpp(float(row.magnification))
    scale = base_mpp / float(mpp)
    target_width = max(int(round(width * scale)), 1)
    target_height = max(int(round(height * scale)), 1)
    return target_width, target_height


def download_roi_images(
    roi_df: pd.DataFrame,
    output_dir: str,
    api_url: str,
    mpp: Optional[float],
    session: requests.Session,
    request_timeout: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """
    Download BCSS RGB ROI crops from the HistomicsTK tile server.

    Parameters
    ----------
    roi_df:
        ROI-level BCSS metadata.
    output_dir:
        Target directory for PNG image crops.
    api_url:
        HistomicsTK API base URL.
    mpp:
        Requested output microns-per-pixel.
    session:
        Active ``requests.Session`` for network calls.
    request_timeout:
        Timeout in seconds for HTTP requests.
    logger:
        Configured logger.

    Returns
    -------
    dict[str, int]
        Download summary with ``downloaded`` and ``skipped`` counts.
    """
    downloaded = 0
    skipped = 0
    total = len(roi_df)

    for idx, row in enumerate(roi_df.itertuples(index=False), start=1):
        xmin = int(round(float(row.xmin)))  # type: ignore[attr-defined]
        ymin = int(round(float(row.ymin)))  # type: ignore[attr-defined]
        xmax = int(round(float(row.xmax)))  # type: ignore[attr-defined]
        ymax = int(round(float(row.ymax)))  # type: ignore[attr-defined]
        stem = build_roi_stem(row.roi_name, xmin, ymin, mpp)  # type: ignore[attr-defined]
        output_path = os.path.join(output_dir, f"{stem}.png")

        if os.path.exists(output_path):
            skipped += 1
            continue

        logger.info(
            "Downloading BCSS RGB ROI %d/%d: %s -> %s",
            idx,
            total,
            row.roi_name,
            output_path,
        )

        params: dict[str, int | float] = {
            "left": xmin,
            "right": xmax,
            "top": ymin,
            "bottom": ymax,
        }
        if mpp is not None:
            params["mm_x"] = 0.001 * float(mpp)
            params["mm_y"] = 0.001 * float(mpp)

        response = session.get(
            f"{api_url}/item/{row.item_id}/tiles/region",
            params=params,
            timeout=request_timeout,
        )
        response.raise_for_status()

        image = open_image_from_response(response, force_rgb=True)
        image.save(output_path)
        downloaded += 1

    return {"downloaded": downloaded, "skipped": skipped}


def download_roi_masks(
    roi_df: pd.DataFrame,
    output_dir: str,
    image_dir: Optional[str],
    mpp: Optional[float],
    session: requests.Session,
    request_timeout: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """
    Download and resize BCSS mask PNGs so they align with the ROI crops.

    Parameters
    ----------
    roi_df:
        ROI-level BCSS metadata.
    output_dir:
        Target directory for mask PNGs.
    image_dir:
        Optional image directory used to match mask sizes to downloaded RGB crops.
    mpp:
        Requested output microns-per-pixel.
    session:
        Active ``requests.Session`` for network calls.
    request_timeout:
        Timeout in seconds for HTTP requests.
    logger:
        Configured logger.

    Returns
    -------
    dict[str, int]
        Download summary with ``downloaded`` and ``skipped`` counts.
    """
    downloaded = 0
    skipped = 0
    total = len(roi_df)

    for idx, row in enumerate(roi_df.itertuples(index=False), start=1):
        xmin = int(round(float(row.xmin)))  # type: ignore[attr-defined]
        ymin = int(round(float(row.ymin)))  # type: ignore[attr-defined]
        stem = build_roi_stem(row.roi_name, xmin, ymin, mpp)  # type: ignore[attr-defined]
        output_path = os.path.join(output_dir, f"{stem}.png")

        if os.path.exists(output_path):
            skipped += 1
            continue

        logger.info(
            "Downloading BCSS mask %d/%d: %s -> %s",
            idx,
            total,
            row.roi_name,
            output_path,
        )

        response = session.get(row.mask_link, timeout=request_timeout)  # type: ignore[attr-defined]
        response.raise_for_status()
        mask = open_image_from_response(response)

        target_size: Optional[tuple[int, int]] = None
        if image_dir is not None:
            image_path = os.path.join(image_dir, f"{stem}.png")
            if os.path.exists(image_path):
                with Image.open(image_path) as rgb_image:
                    target_size = rgb_image.size

        if target_size is None:
            target_size = estimate_target_size(row, mpp)

        if mask.size != target_size:
            cv2 = require_cv2()
            mask = mask.resize(
                target_size,
                resample=cv2.INTER_NEAREST,  # pylint: disable=no-member
            )

        mask.save(output_path)
        downloaded += 1

    return {"downloaded": downloaded, "skipped": skipped}


def download_annotations(
    roi_df: pd.DataFrame,
    output_dir: str,
    api_url: str,
    session: requests.Session,
    request_timeout: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """
    Download HistomicsTK JSON annotations for the BCSS slides used by the ROIs.

    Parameters
    ----------
    roi_df:
        ROI-level BCSS metadata.
    output_dir:
        Target directory for JSON annotation files.
    api_url:
        HistomicsTK API base URL.
    session:
        Active ``requests.Session`` for network calls.
    request_timeout:
        Timeout in seconds for HTTP requests.
    logger:
        Configured logger.

    Returns
    -------
    dict[str, int]
        Download summary with ``downloaded`` and ``skipped`` counts.
    """
    slide_df = roi_df[["slide_id", "filename", "item_id"]].drop_duplicates()
    downloaded = 0
    skipped = 0
    total = len(slide_df)

    for idx, row in enumerate(slide_df.itertuples(index=False), start=1):
        annotation_name = f"{os.path.splitext(str(row.filename))[0]}.json"
        output_path = os.path.join(output_dir, annotation_name)

        if os.path.exists(output_path):
            skipped += 1
            continue

        logger.info(
            "Downloading BCSS annotations %d/%d: %s -> %s",
            idx,
            total,
            row.slide_id,
            output_path,
        )

        with session.get(
            f"{api_url}/annotation/item/{row.item_id}",
            timeout=request_timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            write_response_to_path(response, output_path)
        downloaded += 1

    return {"downloaded": downloaded, "skipped": skipped}


def download_metadata(root_dir: str, logger: logging.Logger) -> None:
    """Download BCSS metadata files and save paths to an atlas file."""
    # Create atlas file at <root_dir>/atlases/tissue_label_atlas.txt
    atlas_dir = os.path.join(root_dir, "atlases")
    os.makedirs(atlas_dir, exist_ok=True)
    bcss_dir = os.path.join(root_dir, "labels", "tissues")
    os.makedirs(bcss_dir, exist_ok=True)
    metadata_dir = os.path.join(root_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    atlas_path = os.path.join(atlas_dir, "tissue_label_atlas.txt")
    atlas_df = pd.DataFrame(
        {
            "type": ["groundtruth_codes", "roi_bounds", "slide_metadata"],
            "path": [
                os.path.join(bcss_dir, "bcss_groundtruth_codes.tsv"),
                os.path.join(bcss_dir, "bcss_roi_bounds.csv"),
                os.path.join(bcss_dir, "bcss_slide_metadata.csv"),
            ],
        }
    )
    atlas_df.to_csv(atlas_path, sep="\t", index=False)

    filenames = [
        "bcss_groundtruth_codes.tsv",
        "bcss_roi_bounds.csv",
        "bcss_slide_metadata.csv",
    ]
    url_names = [
        "gtruth_codes.tsv",
        "roiBounds.csv",
        "slide_magnifications.csv",
    ]
    for filename, url_name in zip(filenames, url_names):
        proc = subprocess.Popen(
            [
                "curl",
                "-o",
                os.path.join(metadata_dir, filename),
                "-l",
                f"https://raw.githubusercontent.com/PathologyDataScience/BCSS/refs/heads/master/meta/{url_name}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        proc.wait()
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info("[curl] %s", line.rstrip("\n"))

    # Persist a precomputed slide_name column in slide_metadata so downstream
    # consumers don't need to re-derive the TCGA barcode from filenames at
    # runtime. The derivation is BCSS/TCGA-specific and lives only here.
    slide_metadata_path = os.path.join(metadata_dir, "bcss_slide_metadata.csv")
    slide_df = pd.read_csv(slide_metadata_path, dtype=str)
    slide_df = slide_df.rename(columns={slide_df.columns[0]: "submitter_id"})
    slide_df["slide_name"] = slide_df.apply(
        lambda row: derive_tissue_slide_name(row["name"], row["submitter_id"]),
        axis=1,
    )
    slide_df.to_csv(slide_metadata_path, index=False)
    logger.info(
        "Added slide_name column to %s (%d rows).",
        slide_metadata_path,
        len(slide_df),
    )


def download_bcss(
    root_dir: str,
    mpp: Optional[float],
    pipelines: list[str] | tuple[str, ...],
    slides: Optional[list[str]] = None,
    api_url: str = DEFAULT_API_URL,
    request_timeout: int = 4 * 60 * 60,
    logger: Optional[logging.Logger] = None,
) -> dict[str, dict[str, int]]:
    """
    Download BCSS ROI data into ``<root_dir>/labels/bcss``.

    Parameters
    ----------
    root_dir:
        Dataset root directory containing BCSS metadata files.
    mpp:
        Requested output microns-per-pixel. Use ``None`` for base magnification.
    pipelines:
        Requested BCSS download stages. Supported values are ``images``,
        ``masks``, and ``annotations``.
    slides:
        Optional list of slide names (TCGA barcodes like ``TCGA-A1-A0SK-DX1``)
        to restrict the download to. When ``None``, every slide referenced by
        the BCSS metadata is downloaded.
    api_url:
        HistomicsTK API base URL.
    request_timeout:
        Timeout in seconds for each HTTP request.
    logger:
        Optional pre-configured logger.

    Returns
    -------
    dict[str, dict[str, int]]
        Per-pipeline summary with ``downloaded`` and ``skipped`` counts.
    """
    os.makedirs(root_dir, exist_ok=True)

    normalized_pipelines = normalize_pipelines(pipelines)

    if mpp is not None:
        mpp = float(mpp)
        if mpp <= 0:
            raise ValueError("mpp must be a positive number or None.")

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="download_bcss")

    download_metadata(root_dir, logger)
    roi_df = load_bcss_metadata(root_dir)
    output_dirs = prepare_output_dirs(root_dir, normalized_pipelines)

    if slides is not None:
        requested = {str(s).strip() for s in slides if str(s).strip()}
        if not requested:
            raise ValueError(
                "slides must contain at least one slide name when provided."
            )
        roi_df = roi_df[roi_df["slide_name"].isin(requested)].reset_index(drop=True)
        unmatched = sorted(requested.difference(roi_df["slide_name"].unique()))
        if unmatched:
            logger.warning(
                "Some requested BCSS slides have no ROIs in the metadata and will be skipped: %s",
                unmatched,
            )
        if roi_df.empty:
            raise ValueError(
                "No BCSS ROIs match the requested slides; nothing to download."
            )

    logger.info(
        "Loaded BCSS metadata for %d ROIs across %d slides.",
        len(roi_df),
        roi_df["slide_id"].nunique(),
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "augur/download_bcss"})

    summary: dict[str, dict[str, int]] = {}
    try:
        if "images" in normalized_pipelines:
            summary["images"] = download_roi_images(
                roi_df=roi_df,
                output_dir=output_dirs["images"],
                api_url=api_url,
                mpp=mpp,
                session=session,
                request_timeout=request_timeout,
                logger=logger,
            )

        if "masks" in normalized_pipelines:
            summary["masks"] = download_roi_masks(
                roi_df=roi_df,
                output_dir=output_dirs["masks"],
                image_dir=output_dirs.get("images"),
                mpp=mpp,
                session=session,
                request_timeout=request_timeout,
                logger=logger,
            )

        if "annotations" in normalized_pipelines:
            summary["annotations"] = download_annotations(
                roi_df=roi_df,
                output_dir=output_dirs["annotations"],
                api_url=api_url,
                session=session,
                request_timeout=request_timeout,
                logger=logger,
            )
    finally:
        session.close()

    logger.info("BCSS download summary: %s", summary)
    return summary


def read_bcss_config(config_path: str) -> dict[str, Any]:
    """
    Load and validate the config keys needed for BCSS downloads.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Parsed and validated BCSS config values.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file did not parse into a dictionary: {config_path}")

    if "root_dir" not in config:
        raise KeyError("Config file is missing required key: root_dir")
    if "bcss_pipelines" not in config:
        raise KeyError("Config file is missing required key: bcss_pipelines")

    root_dir = os.path.expanduser(str(config["root_dir"]))
    mpp = config.get("bcss_mpp")
    pipelines = config["bcss_pipelines"]

    slides = config.get("bcss_slides")
    if slides is not None and not (
        isinstance(slides, (list, tuple)) and all(isinstance(s, str) for s in slides)
    ):
        raise ValueError("bcss_slides must be a list of strings or omitted.")

    return {
        "root_dir": root_dir,
        "mpp": None if mpp is False else mpp,
        "pipelines": pipelines,
        "slides": list(slides) if slides is not None else None,
    }


def main():
    """
    Run the BCSS downloader from a YAML config file.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to data config file.")
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="Optional HistomicsTK API URL override.",
    )
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide --config for the BCSS downloader.")

    config = read_bcss_config(args.config)
    download_bcss(
        root_dir=config["root_dir"],
        mpp=config["mpp"],
        pipelines=config["pipelines"],
        slides=config["slides"],
        api_url=args.api_url,
    )


if __name__ == "__main__":
    main()
