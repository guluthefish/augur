"""Analyzing tile encoder outputs."""

import os
from typing import Any

import pandas as pd


def analyze_metrics(training_cfg: dict[str, Any]):
    """Analyze and print metrics from the tile encoder training configuration."""
    logger_cfg = training_cfg.get("logger", None)
    assert isinstance(
        logger_cfg, dict
    ), "Expected logger configuration to be a dictionary."

    logger_dir = logger_cfg.get("save_dir", None)
    assert isinstance(logger_dir, str), "Expected logger save directory to be a string."
    logger_name = logger_cfg.get("name", None)
    assert isinstance(logger_name, str), "Expected logger name to be a string."

    logger_path = os.path.join(logger_dir, logger_name)

    versions = os.listdir(logger_path)
    assert len(versions) > 0, "No versions found in logger directory."
