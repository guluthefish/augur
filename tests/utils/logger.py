"""Unit tests for the logging utilities."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid

from augur.utils.logger import setup_logger


def _restore_env(env_var: str, previous_value: str | None) -> None:
    """Restore an environment variable after a test mutation."""
    if previous_value is None:
        os.environ.pop(env_var, None)
        return
    os.environ[env_var] = previous_value


def _test_setup_logger_rank_zero_only():
    """Non-zero ranks should stay silent when rank_zero_only is enabled."""
    print("Testing setup_logger() rank-zero-only behavior...")
    previous_rank = os.environ.get("RANK")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["RANK"] = "1"
            logger_name = f"ranked_logger_{uuid.uuid4().hex}"
            logger = setup_logger(tmpdir, logger_name, rank_zero_only=True)

            assert len(logger.handlers) == 1, "Expected a single NullHandler."
            assert isinstance(logger.handlers[0], logging.NullHandler), (
                "Non-zero ranks should not attach stream/file handlers."
            )
            assert os.listdir(tmpdir) == [], (
                "Non-zero ranks should not create log files."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["RANK"] = "0"
            logger_name = f"ranked_logger_{uuid.uuid4().hex}"
            logger = setup_logger(tmpdir, logger_name, rank_zero_only=True)

            assert any(
                isinstance(handler, logging.FileHandler) for handler in logger.handlers
            ), "Rank zero should attach a FileHandler."
            assert any(
                isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                for handler in logger.handlers
            ), "Rank zero should attach a StreamHandler."
            assert len(os.listdir(tmpdir)) == 1, "Rank zero should create one log file."
    finally:
        _restore_env("RANK", previous_rank)

    print("[OK] setup_logger() rank-zero-only test passed.")


def test_all_utils_logger():
    """Run all logger utility tests."""
    print("Running logger utility tests...")
    _test_setup_logger_rank_zero_only()
    print("All logger utility tests passed!")


if __name__ == "__main__":
    test_all_utils_logger()
