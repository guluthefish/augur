"""Download GDC files using gdc-client based on prepared manifest TSVs."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Optional

from augur.utils.logger import setup_logger


def download_data(
    root_dir: str,
    manifests_ready: list[str],
    n_processes: int = 8,
    tcga_user_token_file: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    fail_fast: bool = True,
):
    """
    Download GDC files using one or more prepared manifest TSVs via `gdc-client`.

    For each manifest in `manifests_ready`, this function runs:
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
    manifests_ready:
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
    t0 = time.perf_counter()

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="download_data")

    logger.info(
        "--- STARTING download_data WITH %d MANIFESTS READY FOR DOWNLOAD ---",
        len(manifests_ready),
    )

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
        len(manifests_ready),
        n_processes,
        str(raw_data_dir),
    )

    for manifest in manifests_ready:
        manifest_path = os.path.join(manifest_dir, manifest)

        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        gdc_log_path = os.path.join(
            gdc_log_dir, f"{ts}_{os.path.splitext(manifest)[0]}.gdc-client.log"
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

        start = time.perf_counter()

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
        elapsed = time.perf_counter() - start

        if rc != 0:
            logger.error(
                "gdc-client failed (exit=%d) for manifest=%s (%.2fs). See: %s",
                rc,
                manifest,
                elapsed,
                str(gdc_log_path),
            )
            if fail_fast:
                logger.error(
                    "gdc-client failed (exit=%d) for manifest=%s (%.2fs). See: %s",
                    rc,
                    manifest,
                    elapsed,
                    str(gdc_log_path),
                )
                raise RuntimeError(
                    f"gdc-client failed (exit={rc}) for manifest={manifest}. See: {gdc_log_path}"
                )
            continue

        logger.info("Finished manifest=%s successfully (%.2fs).", manifest, elapsed)

    logger.info(
        "--- END OF download_data. TIME ELAPSED: %.2fs ---", time.perf_counter() - t0
    )
