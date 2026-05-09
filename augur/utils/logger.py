"""Download and process GDC data."""

from __future__ import annotations
import logging
import os
import sys
import time
from typing import Optional


def _current_process_rank() -> int:
    """Infer the current process rank from common distributed env vars."""
    for env_var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return 0


def setup_logger(
    log_dir: str,
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    *,
    rank_zero_only: bool = False,
) -> logging.Logger:
    """
    Create (or reuse) a logger that logs to BOTH:
        - console (stdout)
        - a log file under log_dir

    Parameters
    ----------
    log_dir:
        Directory where log files will be stored (must exist or will be created).
    name:
        Name of the logger (use one per pipeline to avoid duplicate handlers).
    level:
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_file:
        Optional explicit log filename. If None, a timestamped filename is created.
    rank_zero_only:
        If True, only rank 0 emits custom logs. Non-zero ranks receive a
        ``NullHandler`` to keep distributed training logs readable.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    - Uses StreamHandler + FileHandler (standard approach).
    - Prevents duplicate logs by configuring handlers only once.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent adding handlers multiple times (common cause of duplicate logs).
    if getattr(logger, "_configured", False):
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False  # don't bubble up to root logger (can duplicate)

    if rank_zero_only and _current_process_rank() != 0:
        logger.addHandler(logging.NullHandler())
        logger._configured = True  # type: ignore[attr-defined] pylint: disable=protected-access
        return logger

    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if log_file is None:
        log_file = f"{ts}_{name}.log"

    log_path = os.path.join(log_dir, log_file)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Console handler (stdout)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger._configured = True  # type: ignore[attr-defined] pylint: disable=protected-access
    logger.info("Logger initialized. Log file: %s", log_path)
    return logger
