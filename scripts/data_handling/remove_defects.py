"""Remove defective samples from the manifest."""

import os
import time

from openslide import OpenSlide
import pandas as pd

from VexDR.utils.logger import setup_logger


def remove_missing_mpp_metadata(root_dir: str, manifest_downloaded: str) -> str:
    """
    Remove samples from the manifest that do not have MPP metadata.

    Parameters
    ----------
    root_dir:
        Root folder containing:
            - manifests/downloaded/<manifest_downloaded>
            - ordered_data/<submitter_id>/<subfolder>/<id>
    manifest_downloaded:
        The downloaded manifest file name.

    Returns
    -------
    str
        Path to the updated manifest file with samples missing MPP metadata removed.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir, name="remove_defects")
    logger.info(
        "--- STARTING remove_missing_mpp_metadata WITH MANIFEST %s ---",
        manifest_downloaded,
    )

    t0 = time.perf_counter()

    ordered_data_dir = os.path.join(root_dir, "ordered_data")
    os.makedirs(ordered_data_dir, exist_ok=True)
    downloaded_dir = os.path.join(root_dir, "manifests", "downloaded")
    os.makedirs(downloaded_dir, exist_ok=True)

    manifest_path = os.path.join(downloaded_dir, manifest_downloaded)
    manifest_df = pd.read_table(manifest_path, index_col=0, dtype=str)
    original_sample_count = len(manifest_df)

    for idx, row in manifest_df.iterrows():
        submitter_id = row["submitter_id"]
        subfolder = row["subfolder"]
        filename = row["filename"]

        if subfolder != "images":
            continue  # Only check slides in the "images" subfolder

        slide_path = os.path.join(
            ordered_data_dir, submitter_id, subfolder, str(idx), filename
        )

        slide = OpenSlide(slide_path)
        metadata = slide.properties
        if "openslide.mpp-x" not in metadata or "openslide.mpp-y" not in metadata:
            logger.warning(
                "Sample %s is missing MPP metadata and will be removed.", idx
            )
            manifest_df.drop(index=idx, inplace=True)

    saved_filename = time.strftime(
        "gdc_manifest.%Y-%m-%d_%H-%M-%S.final.txt", time.localtime()
    )
    out_path = os.path.join(downloaded_dir, saved_filename)
    manifest_df.to_csv(out_path, sep="\t")

    logger.info(
        "Removed %d samples missing MPP metadata. Updated manifest saved: %s",
        original_sample_count - len(manifest_df),
        out_path,
    )

    logger.info(
        "--- END OF remove_missing_mpp_metadata. TIME ELAPSED: %.2fs ---",
        time.perf_counter() - t0,
    )

    return saved_filename
