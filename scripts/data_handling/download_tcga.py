"""Download GDC files using gdc-client based on prepared manifest TSVs."""

from argparse import ArgumentParser
import io
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from openslide import OpenSlide
import pandas as pd
import requests
import yaml

from augur.utils.logger import setup_logger


def _setup_module_logger(root_dir: str) -> logging.Logger:
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="download_tcga")


def create_ready_for_download_manifest(
    root_dir: str, raw_manifest: str, logger: Optional[logging.Logger] = None
):
    """
    Merge raw GDC manifest TSV file(s) and augment them with selected metadata from the GDC API.

    The merged manifest is saved under <root_dir>/manifests/ready_for_download/gdc_manifest.merged.txt.

    Parameters
    ----------
    root_dir:
        Root folder containing:
            - manifests/raw/  (input manifests)
            - manifests/ready_for_download/ (outputs will be saved here)
            - logs/ (logs)
    raw_manifest:
        A raw manifest filename under <root_dir>/manifests/raw/.
        Each manifest is expected to be a TSV with the first column as file UUID (id).
    logger:
        Optional logger. If None, a default logger will be created.

    Raises
    ------
    requests.HTTPError
        If the GDC API request fails.
    ValueError:
        If the GDC API response is missing expected columns or if the merged manifest
        is missing required metadata fields.
    """

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or _setup_module_logger(root_dir)

    raw_dir = os.path.join(root_dir, "manifests", "raw")

    df = pd.read_table(os.path.join(raw_dir, raw_manifest), index_col=0, dtype=str)
    logger.info("Finished reading manifests. Total rows: %d", len(df))

    file_ids = list(df.index)

    fields = [
        "cases.submitter_id",
        "data_category",
        "data_format",
        "data_type",
    ]

    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "files.file_id", "value": file_ids},
        },
        "format": "tsv",
        "fields": ",".join(fields),
        "size": str(max(1, len(file_ids))),
    }

    logger.info("Requesting metadata from GDC for %d file_id(s)...", len(file_ids))
    try:
        r = requests.post("https://api.gdc.cancer.gov/files", json=payload, timeout=300)
        r.raise_for_status()
    except requests.RequestException:
        logger.exception("GDC metadata request failed.")
        raise
    logger.info("Finished requesting metadata from GDC.")

    metadata_df = pd.read_table(io.StringIO(r.content.decode()), dtype=str)

    # GDC TSV typically includes an "id" column; keep robust fallback.
    if "id" in metadata_df.columns:
        metadata_df = metadata_df.set_index("id")
    elif "file_id" in metadata_df.columns:
        metadata_df = metadata_df.set_index("file_id")
    else:
        raise ValueError(
            f"Metadata response missing expected id column. Columns: {list(metadata_df.columns)}"
        )

    # Create a convenient submitter_id column (first case submitter id)
    if "cases.0.submitter_id" in metadata_df.columns:
        metadata_df["submitter_id"] = metadata_df["cases.0.submitter_id"]

    # Drop all expanded cases.*.submitter_id columns to avoid clutter
    pat = re.compile(r"cases\.\d+\.submitter_id")
    drop_columns = [col for col in metadata_df.columns if pat.match(col)]
    if drop_columns:
        metadata_df = metadata_df.drop(columns=drop_columns)

    merged_df = df.join(metadata_df, how="left")
    merged_df["subfolder"] = merged_df.apply(
        lambda row: (
            "images"
            if row.get("data_format", "").upper() == "SVS"
            else str(row.get("data_category", "unknown")).lower().replace(" ", "_")
        ),
        axis=1,
    )

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    os.makedirs(ready_dir, exist_ok=True)

    merged_file = "gdc_manifest.merged.txt"
    merged_path = os.path.join(ready_dir, merged_file)
    merged_df.to_csv(merged_path, sep="\t")

    logger.info("Merged manifest saved: %s", merged_path)


def filter_out_already_downloaded(
    root_dir: str, merged_manifest: str, logger: Optional[logging.Logger] = None
):
    """
    Filter merged manifest(s) by removing file UUIDs that are already downloaded.

    The filtered manifest is saved under <root_dir>/manifests/ready_for_download/gdc_manifest.filtered.txt.

    A UUID folder under <root_dir>/raw_data/<id>/ is considered "incomplete" and will be DELETED if:
        - it contains any '*.partial' file (interrupted download)
        - it is empty
        - it contains only a 'logs/' folder
        - expected file is missing
        - expected file size does not match manifest

    A UUID folder is considered "downloaded" if:
        - it exists under <root_dir>/raw_data/ and does not meet any of the
          "incomplete" criteria above
        - it exists under <root_dir>/ordered_data/<submitter_id>/<subfolder>/<id>

    Incomplete UUIDs are NOT treated as downloaded, so they remain in the filtered
    manifest for re-download.

    Parameters
    ----------
    root_dir:
        Root folder containing raw_data/ and manifests/ready_for_download/.
    merged_manifest:
        Filename of the merged manifest (under manifests/ready_for_download/)
        to filter.
    logger:
        Optional logger. If None, a default logger will be created.

    Raises
    ------
    ValueError:
        If the merged manifest is missing required columns or if expected metadata fields
        are missing for validation.
    """

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or _setup_module_logger(root_dir)

    raw_data_dir = os.path.join(root_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    merged_df = pd.read_table(
        os.path.join(ready_dir, merged_manifest), index_col=0, dtype=str
    )
    merged_df.index = merged_df.index.astype(str).str.strip().str.lower()

    required_columns = {"filename", "size"}
    missing_columns = required_columns.difference(merged_df.columns)
    if missing_columns:
        raise ValueError(
            f"Merged manifest missing required columns: {sorted(missing_columns)}"
        )

    duplicate_mask = merged_df.index.duplicated(keep="first")
    if duplicate_mask.any():
        n_dup = int(duplicate_mask.sum())
        logger.warning(
            "Found %d duplicate file_id rows in merged manifest; keeping first occurrence per UUID.",
            n_dup,
        )
        merged_df = merged_df.loc[~duplicate_mask].copy()

    expected_df = merged_df[["filename", "size"]].copy()
    expected_df["filename"] = expected_df["filename"].astype(str).str.strip()
    expected_df["size"] = pd.to_numeric(expected_df["size"], errors="coerce")
    expected_by_id = expected_df.to_dict(orient="index")

    downloaded_ids: list[str] = []
    deleted_ids: list[str] = []
    deleted_reasons: dict[str, int] = {}

    logger.info(
        "Scanning raw_data for downloaded UUID folders: %s",
        raw_data_dir,
    )

    # --- Check each UUID folder under raw_data/ ---
    scan_entries = os.listdir(raw_data_dir)
    total_entries = len(scan_entries)
    progress_every = max(1, total_entries // 100) if total_entries else 1

    def _log_scan_progress(current: int, total: int, folder_name: str) -> None:
        width = 30
        if total <= 0:
            prg_bar = "#" * width
            pct = 100.0
        else:
            filled = int(width * current / total)
            prg_bar = "#" * filled + "-" * (width - filled)
            pct = (100.0 * current) / total
        logger.info(
            "%s scan progress [%s] %d/%d (%.1f%%)",
            folder_name,
            prg_bar,
            current,
            total,
            pct,
        )

    if total_entries == 0:
        logger.info("No entries found under raw_data.")
    else:
        _log_scan_progress(0, total_entries, "raw_data")

    for idx, file_id in enumerate(scan_entries, start=1):
        file_id_path = os.path.join(raw_data_dir, file_id)

        if not os.path.isdir(file_id_path) or os.path.islink(file_id_path):
            if idx % progress_every == 0 or idx == total_entries:
                _log_scan_progress(idx, total_entries, "raw_data")
            continue

        file_id_norm = file_id.strip().lower()
        entries = [e for e in os.listdir(file_id_path) if not e.startswith(".")]

        only_logs = (
            len(entries) == 1
            and entries[0] == "logs"
            and os.path.isdir(os.path.join(file_id_path, "logs"))
        )
        is_empty = len(entries) == 0

        has_partial = False
        for _, _, files in os.walk(file_id_path):
            if any(f.endswith(".partial") for f in files):
                has_partial = True
                break

        reason: Optional[str] = None
        if has_partial or only_logs or is_empty:
            reason = (
                "partial" if has_partial else ("logs-only" if only_logs else "empty")
            )
        else:
            expected = expected_by_id.get(file_id_norm)
            if expected is None:
                logger.warning(
                    "UUID folder not present in merged manifest; treating as downloaded: %s",
                    file_id_path,
                )
            else:
                expected_filename = str(expected["filename"]).strip()
                expected_size_raw = expected["size"]

                if not expected_filename:
                    reason = "missing-manifest-filename"
                else:
                    target_path = os.path.join(file_id_path, expected_filename)
                    if not os.path.isfile(target_path):
                        reason = "missing-expected-file"
                    else:
                        try:
                            local_size = os.path.getsize(target_path)
                        except OSError:
                            reason = "unreadable-file"
                        else:
                            if pd.notna(expected_size_raw):
                                expected_size = int(expected_size_raw)
                                if local_size != expected_size:
                                    reason = "size-mismatch"

        if reason is not None:
            try:
                shutil.rmtree(file_id_path)
                deleted_ids.append(file_id)
                deleted_reasons[reason] = deleted_reasons.get(reason, 0) + 1
                logger.warning(
                    "Deleted incomplete folder (%s): %s", reason, file_id_path
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to delete incomplete folder: %s", file_id_path)
        else:
            downloaded_ids.append(file_id)

        if idx % progress_every == 0 or idx == total_entries:
            _log_scan_progress(idx, total_entries, "raw_data")
    # --- End of raw_data/ scan ---

    # --- Check ordered_data/ for any additional downloaded UUIDs ---
    ordered_data_dir = os.path.join(root_dir, "ordered_data")
    scan_entris = (
        os.listdir(ordered_data_dir) if os.path.isdir(ordered_data_dir) else []
    )
    total_entries = len(scan_entris)

    if total_entries == 0:
        logger.info("No entries found under ordered_data.")
    else:
        logger.info(
            "Scanning ordered_data for downloaded and ordered UUID folders: %s",
            ordered_data_dir,
        )
        progress_every = max(1, total_entries // 100) if total_entries else 1
        for idx, submitter_id in enumerate(scan_entris, start=1):
            submitter_path = os.path.join(ordered_data_dir, submitter_id)
            if not os.path.isdir(submitter_path) or os.path.islink(submitter_path):
                if idx % progress_every == 0 or idx == total_entries:
                    _log_scan_progress(idx, total_entries, "ordered_data")
                continue

            for subfolder in os.listdir(submitter_path):
                subfolder_path = os.path.join(submitter_path, subfolder)
                if not os.path.isdir(subfolder_path) or os.path.islink(subfolder_path):
                    continue

                for file_id in os.listdir(subfolder_path):
                    file_id_path = os.path.join(subfolder_path, file_id)
                    if not os.path.isdir(file_id_path) or os.path.islink(file_id_path):
                        continue

                    file_id_norm = file_id.strip().lower()
                    downloaded_ids.append(file_id_norm)

            if idx % progress_every == 0 or idx == total_entries:
                _log_scan_progress(idx, total_entries, "ordered_data")
    # --- End of ordered_data/ scan ---

    logger.info("Downloaded UUID folders detected: %d", len(downloaded_ids))
    logger.info("Incomplete UUID folders deleted: %d", len(deleted_ids))
    if deleted_reasons:
        reason_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(deleted_reasons.items())
        )
        logger.info("Deletion reasons summary: %s", reason_summary)

    downloaded_ids_norm = {x.strip().lower() for x in downloaded_ids}

    filtered_df = merged_df.loc[~merged_df.index.isin(downloaded_ids_norm)].copy()

    filtered_file = "gdc_manifest.filtered.txt"
    out_path = os.path.join(ready_dir, filtered_file)
    filtered_df.to_csv(out_path, sep="\t")

    logger.info(
        "Filtered manifest saved: %s (rows=%d -> %d)",
        out_path,
        len(merged_df),
        len(filtered_df),
    )

    if deleted_ids:
        log_dir = os.path.join(root_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        deleted_list_file = "deleted_incomplete_ids.txt"
        deleted_list_path = os.path.join(log_dir, deleted_list_file)
        with open(deleted_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(set(deleted_ids))) + "\n")
        logger.info("Deleted UUID list saved: %s", deleted_list_path)


def split_manifest(
    root_dir: str,
    filtered_manifest: str,
    n_files_per_command: int,
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """
    Split a (filtered) GDC manifest TSV into smaller manifest TSV files using pandas.

    Parameters
    ----------
    root_dir:
        Root folder containing manifests/ready_for_download/.
    filtered_manifest:
        The filtered manifest filename under <root_dir>/manifests/ready_for_download/.
    n_files_per_command:
        Number of rows (file UUIDs) per split manifest.
    logger:
        Optional logger. If None, a default logger will be created.

    Returns
    -------
    list[str]
        List of split manifest filenames saved under <root_dir>/manifests/ready_for_download/.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or _setup_module_logger(root_dir)

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    in_path = os.path.join(ready_dir, filtered_manifest)

    df = pd.read_table(in_path, index_col=0, dtype=str)
    logger.info(
        "Splitting manifest %s (rows=%d) with chunk size=%d",
        in_path,
        len(df),
        n_files_per_command,
    )

    splitted_files: list[str] = []

    part = 0
    for start in range(0, len(df), n_files_per_command):
        end = min(start + n_files_per_command, len(df))
        chunk_df = df.iloc[start:end]

        file = f"gdc_manifest.part{part:04d}.splitted.txt"
        out_path = os.path.join(ready_dir, file)

        chunk_df.to_csv(out_path, sep="\t")
        splitted_files.append(file)

        logger.info(
            "Wrote split manifest part %04d: %s (rows=%d)",
            part,
            out_path,
            len(chunk_df),
        )
        part += 1

    logger.info(
        "Finished splitting manifest into %d parts. Total rows: %d",
        len(splitted_files),
        len(df),
    )
    return splitted_files


def download_data(
    root_dir: str,
    splitted_manifests: list[str],
    n_processes: int = 8,
    tcga_user_token_file: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    fail_fast: bool = True,
):
    """
    Download GDC files using one or more prepared manifest TSVs via `gdc-client`.

    For each manifest in `splitted_manifests`, this function runs:
        gdc-client download -m <manifest_path> -n <n_processes> \
            -d <raw_data_dir> --log-file <gdc_log>

    If `tcga_user_token_file` is provided, it is passed as `-t <token>` for controlled-access data.

    Logging behavior
    ----------------
    - Python-level logs go to BOTH:
        - screen (stdout)
        - <root_dir>/logs/<timestamp>_download_data.log using standard logging handlers.
    - `gdc-client` also writes its own log file per manifest via `--log-file`.
    - Additionally, this function streams the subprocess output (stdout/stderr merged)
      into the Python logger.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/ready_for_download/
          - raw_data/
          - logs/
    splitted_manifests:
        List of manifest filenames under <root_dir>/manifests/ready_for_download/.
    n_processes:
        Number of parallel download processes/connections for `gdc-client`.
    tcga_user_token_file:
        Optional path to a GDC token file for controlled-access downloads.
    logger:
        Optional pre-configured logger. If None, a default one is created.
    fail_fast:
        If True, log on the first failed manifest download.
        If False, continue downloading the remaining manifests.

    Raises
    ------
    RuntimeError:
        If `gdc-client` returns a non-zero exit code and `fail_fast=True`.
    """

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or _setup_module_logger(root_dir)

    gdc_client = shutil.which("gdc-client")
    if not gdc_client:
        logger.error(
            "gdc-client not found in PATH. Install the GDC Data Transfer Tool binary."
        )
        raise RuntimeError(
            "gdc-client not found in PATH. Install the GDC Data Transfer Tool binary."
        )

    manifest_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    raw_data_dir = os.path.join(root_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Put gdc-client logs in a subfolder for cleanliness
    gdc_log_dir = os.path.join(log_dir, "gdc-client")
    os.makedirs(gdc_log_dir, exist_ok=True)

    logger.info(
        "Starting downloads: manifests=%d, n_processes=%d, raw_data_dir=%s",
        len(splitted_manifests),
        n_processes,
        str(raw_data_dir),
    )

    for manifest in splitted_manifests:
        manifest_path = os.path.join(manifest_dir, manifest)

        gdc_log_path = os.path.join(
            gdc_log_dir, f"{os.path.splitext(manifest)[0]}.gdc-client.log"
        )

        args = [
            gdc_client,
            "download",
            "-m",
            str(manifest_path),
            "-n",
            str(n_processes),
            "-d",
            str(raw_data_dir),
            "--log-file",
            str(gdc_log_path),
        ]
        if tcga_user_token_file is not None:
            args += ["-t", str(tcga_user_token_file)]

        logger.info("Running gdc-client for manifest=%s", manifest)
        logger.info("gdc-client log: %s", str(gdc_log_path))
        logger.info("Command: %s", " ".join(args))

        # Stream subprocess output into the logger (stdout+stderr merged).
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info("[gdc-client] %s", line.rstrip("\n"))

        rc = proc.wait()

        if rc != 0:
            logger.error(
                "gdc-client failed (exit=%d) for manifest=%s. See: %s",
                rc,
                manifest,
                str(gdc_log_path),
            )
            if fail_fast:
                logger.error(
                    "gdc-client failed (exit=%d) for manifest=%s. See: %s",
                    rc,
                    manifest,
                    str(gdc_log_path),
                )
                raise RuntimeError(
                    f"gdc-client failed (exit={rc}) for manifest={manifest}. See: {gdc_log_path}"
                )
            continue

        logger.info("Finished manifest=%s successfully.", manifest)


def reorder_data(
    root_dir: str, merged_manifest: str, logger: Optional[logging.Logger] = None
):
    """
    Reorder downloaded files from raw_data/ to ordered_data/ based on submitter_id and subfolder.

    Manifest is saved under <root_dir>/manifests/downloaded/gdc_manifest.moved.txt.

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
    merged_manifest:
        Path to the merged manifest file containing all the data to be reordered.
    logger:
        Optional pre-configured logger. If None, a default one is created.


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
    logger = logger or _setup_module_logger(root_dir)

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    raw_data_dir = os.path.join(root_dir, "raw_data")
    ordered_data_dir = os.path.join(root_dir, "ordered_data")
    os.makedirs(ordered_data_dir, exist_ok=True)
    downloaded_dir = os.path.join(root_dir, "manifests", "downloaded")
    os.makedirs(downloaded_dir, exist_ok=True)

    df = pd.read_table(os.path.join(ready_dir, merged_manifest), index_col=0, dtype=str)

    if df.empty:
        moved_file = "gdc_manifest.moved.txt"
        out_path = os.path.join(downloaded_dir, moved_file)
        df.to_csv(out_path, sep="\t")
        logger.info("No rows to reorder. Empty moved manifest saved: %s", out_path)
        return moved_file

    if "submitter_id" not in df.columns:
        raise ValueError("Manifest missing required column: submitter_id")
    if "subfolder" not in df.columns:
        df["subfolder"] = df.apply(
            lambda row: (
                "images"
                if str(row.get("data_format", "")).upper() == "SVS"
                else str(row.get("data_category", "unknown")).lower().replace(" ", "_")
            ),
            axis=1,
        )

    ids = df.index.astype(str).to_list()
    submitter_ids = df["submitter_id"].fillna("unknown").astype(str).to_list()
    subfolders = df["subfolder"].fillna("unknown").astype(str).to_list()

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

    manifest_moved_df = df.iloc[moved_positions].copy()
    moved_file = "gdc_manifest.moved.txt"
    out_path = os.path.join(downloaded_dir, moved_file)
    manifest_moved_df.to_csv(out_path, sep="\t")

    logger.info(
        "Reordered files: moved=%d, already_ordered=%d, missing=%d, manifest=%s",
        moved_count,
        already_ordered_count,
        missing_count,
        out_path,
    )


def remove_missing_mpp_metadata(
    root_dir: str, moved_manifest: str, logger: Optional[logging.Logger] = None
):
    """
    Remove samples from the manifest that do not have MPP metadata.

    The final manifest is saved under <root_dir>/manifests/downloaded/gdc_manifest.final.txt.

    Parameters
    ----------
    root_dir:
        Root folder containing:
            - manifests/downloaded/<moved_manifest>
            - ordered_data/<submitter_id>/<subfolder>/<id>
    moved_manifest:
        The reordered manifest file name.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or _setup_module_logger(root_dir)

    ordered_data_dir = os.path.join(root_dir, "ordered_data")
    os.makedirs(ordered_data_dir, exist_ok=True)
    downloaded_dir = os.path.join(root_dir, "manifests", "downloaded")
    os.makedirs(downloaded_dir, exist_ok=True)

    manifest_path = os.path.join(downloaded_dir, moved_manifest)
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

    saved_filename = "gdc_manifest.final.txt"
    out_path = os.path.join(downloaded_dir, saved_filename)
    manifest_df.to_csv(out_path, sep="\t")

    logger.info(
        "Removed %d samples missing MPP metadata. Updated manifest saved: %s",
        original_sample_count - len(manifest_df),
        out_path,
    )


def main():
    """
    Run the BCSS downloader from a YAML config file.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to data config file.")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide --config for the TCGA downloader.")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root_dir = config["root_dir"]
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(
            f"root_dir does not exist or is not a directory: {root_dir}"
        )

    # ----- Process manifests and prepare for download -----
    manifest_raw = config["manifest_raw"]
    raw_dir = os.path.join(root_dir, "manifests", "raw")
    if not os.path.exists(os.path.join(raw_dir, manifest_raw)):
        raise FileNotFoundError(
            f"Raw manifest file not found: {os.path.join(raw_dir, manifest_raw)}"
        )

    create_ready_for_download_manifest(root_dir, manifest_raw)
    filter_out_already_downloaded(root_dir, "gdc_manifest.merged.txt")

    n_files_per_command = config["n_files_per_command"]
    if not isinstance(n_files_per_command, int) or n_files_per_command <= 0:
        raise ValueError("n_files_per_command must be a positive integer")

    splitted_manifests = split_manifest(
        root_dir, "gdc_manifest.filtered.txt", n_files_per_command
    )

    # ----- Download using gdc-client and reorder data -----
    tcga_user_token_file = config["tcga_user_token_file"]
    if tcga_user_token_file is not False:
        token_path = os.path.expanduser(tcga_user_token_file)
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found: {token_path}")
    else:
        token_path = None

    n_processes = config["n_processes"]
    if not isinstance(n_processes, int) or n_processes <= 0:
        raise ValueError("n_processes must be a positive integer")
    download_data(
        root_dir,
        splitted_manifests,
        n_processes=n_processes,
        tcga_user_token_file=token_path,
    )
    reorder_data(root_dir, "gdc_manifest.merged.txt")

    # ----- Remove samples missing MPP metadata -----
    remove_missing_mpp_metadata(root_dir, "gdc_manifest.moved.txt")

    # ----- Create atlas metadata file -----
    atlas_df = pd.DataFrame(
        {
            "type": ["final_manifest"],
            "path": [
                os.path.join(
                    root_dir, "manifests", "downloaded", "gdc_manifest.final.txt"
                )
            ],
        }
    )
    atlas_dir = os.path.join(root_dir, "atlases")
    atlas_path = os.path.join(atlas_dir, "manifest_atlas.txt")
    os.makedirs(atlas_dir, exist_ok=True)
    atlas_df.to_csv(atlas_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
