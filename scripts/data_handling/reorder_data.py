"""Reorder downloaded files based on manifest metadata and move them to an ordered directory structure."""

import logging
import os
import shutil
import time
from typing import Optional

import pandas as pd

from VexDR.utils.logger import setup_logger


def reorder_data(
    root_dir: str, manifests_ready: list[str], logger: Optional[logging.Logger] = None
) -> str:
    """
    Reorder downloaded files from raw_data/ to ordered_data/ based on submitter_id and subfolder.

    For each manifest in `manifests_ready`, this function reads the manifest TSV and moves
    files from <root_dir>/raw_data/<id>/ to ordered_data/<submitter_id>/<subfolder>/<id>/.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/ready_for_download/
          - raw_data/
          - ordered_data/
          - logs/
    manifests_ready:
        List of manifest filenames under <root_dir>/manifests/ready_for_download/
        to process for reordering.
    logger:
        Optional pre-configured logger. If None, a default one is created.

    Returns
    ---
    str
        Manifest filename saved under <root_dir>/manifests/downloaded

    Raises
    ------
    ValueError:
        If the manifest is missing required columns (submitter_id, subfolder)
        or if expected metadata fields are missing for validation.
    OSError:
        If file moving fails due to OS errors (e.g., permission issues, missing files).
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="reorder_data")
    logger.info(
        "--- STARTING reorder_data WITH %d MANIFEST(S) READY FOR REORDERING ---",
        len(manifests_ready),
    )

    t0 = time.perf_counter()
    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    raw_data_dir = os.path.join(root_dir, "raw_data")
    ordered_data_dir = os.path.join(root_dir, "ordered_data")
    os.makedirs(ordered_data_dir, exist_ok=True)
    downloaded_dir = os.path.join(root_dir, "manifests", "downloaded")
    os.makedirs(downloaded_dir, exist_ok=True)

    manifest_df = pd.concat(
        [
            pd.read_table(os.path.join(ready_dir, file), index_col=0, dtype=str)
            for file in manifests_ready
        ],
        axis=0,
    )

    if manifest_df.empty:
        moved_file = time.strftime(
            "gdc_manifest.%Y-%m-%d_%H-%M-%S.moved.txt", time.localtime()
        )
        out_path = os.path.join(downloaded_dir, moved_file)
        manifest_df.to_csv(out_path, sep="\t")
        logger.info("No rows to reorder. Empty moved manifest saved: %s", out_path)
        return moved_file

    if "submitter_id" not in manifest_df.columns:
        raise ValueError("Manifest missing required column: submitter_id")
    if "subfolder" not in manifest_df.columns:
        manifest_df["subfolder"] = manifest_df.apply(
            lambda row: (
                "images"
                if str(row.get("data_format", "")).upper() == "SVS"
                else str(row.get("data_category", "unknown")).lower().replace(" ", "_")
            ),
            axis=1,
        )

    ids = manifest_df.index.astype(str).to_list()
    submitter_ids = manifest_df["submitter_id"].fillna("unknown").astype(str).to_list()
    subfolders = manifest_df["subfolder"].fillna("unknown").astype(str).to_list()

    moved_positions: list[int] = []
    moved_count = 0
    already_ordered_count = 0
    missing_count = 0

    for i, (file_id, submitter_id, subfolder) in enumerate(
        zip(ids, submitter_ids, subfolders)
    ):
        file_id = file_id.strip()
        submitter_id = submitter_id.strip() or "unknown"
        subfolder = subfolder.strip() or "unknown"

        src_dir = os.path.join(raw_data_dir, file_id)
        dst_dir = os.path.join(ordered_data_dir, submitter_id, subfolder, file_id)

        if os.path.isdir(dst_dir):
            moved_positions.append(i)
            already_ordered_count += 1
            continue

        if not os.path.isdir(src_dir):
            missing_count += 1
            continue

        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        try:
            os.rename(src_dir, dst_dir)
        except OSError:
            shutil.move(src_dir, dst_dir)

        moved_positions.append(i)
        moved_count += 1

    manifest_moved_df = manifest_df.iloc[moved_positions].copy()
    moved_file = time.strftime(
        "gdc_manifest.%Y-%m-%d_%H-%M-%S.moved.txt", time.localtime()
    )
    out_path = os.path.join(downloaded_dir, moved_file)
    manifest_moved_df.to_csv(out_path, sep="\t")

    logger.info(
        "Reordered files: moved=%d, already_ordered=%d, missing=%d, manifest=%s",
        moved_count,
        already_ordered_count,
        missing_count,
        out_path,
    )
    logger.info(
        "--- END OF reorder_data. TIME ELAPSED: %.2fs ---", time.perf_counter() - t0
    )
    return moved_file
