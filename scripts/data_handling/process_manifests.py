"""Process GDC manifest files: merge, filter, and split for download preparation."""

import io
import logging
import os
import re
import shutil
import time
from typing import Optional

import pandas as pd
import requests

from VexDR.utils.logger import setup_logger


def create_ready_for_download_manifest(
    root_dir: str, manifests_raw: list[str], logger: Optional[logging.Logger] = None
) -> list[str]:
    """
    Merge raw GDC manifest TSV file(s) and augment them with selected metadata from the GDC API.

    Parameters
    ----------
    root_dir:
        Root folder containing:
            - manifests/raw/  (input manifests)
            - manifests/ready_for_download/ (outputs will be saved here)
            - logs/ (logs)
    manifests_raw:
        List of raw manifest filenames under <root_dir>/manifests/raw/.
        Each manifest is expected to be a TSV with the first column as file UUID (id).
    logger:
        Optional logger. If None, a default logger will be created.

    Returns
    -------
    list[str]
        A one-element list containing the merged manifest filename saved under
        <root_dir>/manifests/ready_for_download/.

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
    logger = logger or setup_logger(log_dir, name="process_manifests")
    logger.info(
        "--- START create_ready_for_download_manifest WITH %d RAW MANIFEST(S) ---",
        len(manifests_raw),
    )

    t0 = time.perf_counter()

    # logger.info("Start reading %d raw manifest file(s)...", len(manifests_raw))

    raw_dir = os.path.join(root_dir, "manifests", "raw")

    manifests_df = pd.concat(
        [
            pd.read_table(os.path.join(raw_dir, file), index_col=0, dtype=str)
            for file in manifests_raw
        ],
        axis=0,
    )
    logger.info("Finished reading manifests. Total rows: %d", len(manifests_df))

    file_ids = list(manifests_df.index)

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

    manifest_merged_df = manifests_df.join(metadata_df, how="left")
    manifest_merged_df["subfolder"] = manifest_merged_df.apply(
        lambda row: (
            "images"
            if row.get("data_format", "").upper() == "SVS"
            else str(row.get("data_category", "unknown")).lower().replace(" ", "_")
        ),
        axis=1,
    )

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    os.makedirs(ready_dir, exist_ok=True)

    merged_file = time.strftime(
        "gdc_manifest.%Y-%m-%d_%H-%M-%S.merged.txt", time.localtime()
    )
    merged_path = os.path.join(ready_dir, merged_file)
    manifest_merged_df.to_csv(merged_path, sep="\t")

    logger.info("Merged manifest saved: %s", merged_path)
    logger.info(
        "--- END OF create_ready_for_download_manifest. TIME ELAPSED: %.2fs ---",
        time.perf_counter() - t0,
    )

    return [merged_file]


def filter_out_already_downloaded(
    root_dir: str, manifests_merged: list[str], logger: Optional[logging.Logger] = None
) -> str:
    """
    Filter merged manifest(s) by removing file UUIDs that are already downloaded.

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
    manifests_merged:
        List of merged manifest filenames (under manifests/ready_for_download/)
        to concatenate and filter.
    logger:
        Optional logger. If None, a default logger will be created.

    Returns
    -------
    str
        Filename of the filtered manifest saved under <root_dir>/manifests/ready_for_download/.

    Raises
    ------
    ValueError:
        If the merged manifest is missing required columns or if expected metadata fields
        are missing for validation.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="process_manifests")
    logger.info(
        "--- STARTING filter_out_already_downloaded WITH %d MERGED MANIFEST(S) ---",
        len(manifests_merged),
    )

    t0 = time.perf_counter()
    raw_data_dir = os.path.join(root_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    manifest_merged_df = pd.concat(
        [
            pd.read_table(os.path.join(ready_dir, file), index_col=0, dtype=str)
            for file in manifests_merged
        ],
        axis=0,
    )
    manifest_merged_df.index = (
        manifest_merged_df.index.astype(str).str.strip().str.lower()
    )

    required_columns = {"filename", "size"}
    missing_columns = required_columns.difference(manifest_merged_df.columns)
    if missing_columns:
        raise ValueError(
            f"Merged manifest missing required columns: {sorted(missing_columns)}"
        )

    duplicate_mask = manifest_merged_df.index.duplicated(keep="first")
    if duplicate_mask.any():
        n_dup = int(duplicate_mask.sum())
        logger.warning(
            "Found %d duplicate file_id rows in merged manifest; keeping first occurrence per UUID.",
            n_dup,
        )
        manifest_merged_df = manifest_merged_df.loc[~duplicate_mask].copy()

    expected_df = manifest_merged_df[["filename", "size"]].copy()
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

    manifest_filtered_df = manifest_merged_df.loc[
        ~manifest_merged_df.index.isin(downloaded_ids_norm)
    ].copy()

    filtered_file = time.strftime(
        "gdc_manifest.%Y-%m-%d_%H-%M-%S.filtered.txt", time.localtime()
    )
    out_path = os.path.join(ready_dir, filtered_file)
    manifest_filtered_df.to_csv(out_path, sep="\t")

    logger.info(
        "Filtered manifest saved: %s (rows=%d -> %d)",
        out_path,
        len(manifest_merged_df),
        len(manifest_filtered_df),
    )

    if deleted_ids:
        log_dir = os.path.join(root_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        deleted_list_file = time.strftime(
            "%Y-%m-%d_%H-%M-%S_deleted_incomplete_ids.txt", time.localtime()
        )
        deleted_list_path = os.path.join(log_dir, deleted_list_file)
        with open(deleted_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(set(deleted_ids))) + "\n")
        logger.info("Deleted UUID list saved: %s", deleted_list_path)

    logger.info(
        "--- END OF filter_out_already_downloaded. TIME ELAPSED: %.2fs ---",
        time.perf_counter() - t0,
    )
    return filtered_file


def split_manifest(
    root_dir: str,
    manifest_filtered: str,
    n_files_per_command: int,
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """
    Split a (filtered) GDC manifest TSV into smaller manifest TSV files using pandas.

    Parameters
    ----------
    root_dir:
        Root folder containing manifests/ready_for_download/.
    manifest_filtered:
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
    logger = logger or setup_logger(log_dir, name="process_manifests")
    logger.info(
        "--- STARTING split_manifest WITH %d FILTERED ROWS AND CHUNK SIZE %d ---",
        len(
            pd.read_table(
                os.path.join(
                    root_dir, "manifests", "ready_for_download", manifest_filtered
                )
            )
        ),
        n_files_per_command,
    )

    t0 = time.perf_counter()
    ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
    in_path = os.path.join(ready_dir, manifest_filtered)

    manifest_df = pd.read_table(in_path, index_col=0, dtype=str)
    logger.info(
        "Splitting manifest %s (rows=%d) with chunk size=%d",
        in_path,
        len(manifest_df),
        n_files_per_command,
    )

    splitted_files: list[str] = []
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    part = 0
    for start in range(0, len(manifest_df), n_files_per_command):
        end = min(start + n_files_per_command, len(manifest_df))
        chunk_df = manifest_df.iloc[start:end]

        file = f"gdc_manifest.{ts}.part{part:04d}.splitted.txt"
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
        len(manifest_df),
    )
    logger.info(
        "--- END OF split_manifest. TIME ELAPSED: %.2fs ---", time.perf_counter() - t0
    )
    return splitted_files
