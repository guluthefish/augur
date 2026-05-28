"""Mutational-signature subtask label utilities for slide-level datasets.

Supports per-submitter COSMIC signature exposure vectors produced by
:mod:`scripts.data_handling.extract_signatures` for the SBS, DBS, ID, and CN
(CNV48) signature classes. Each subtask corresponds to one such contribution
table on disk.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

SUPPORTED_SUBTASKS: tuple[str, ...] = (
    "sbs_regression",
    "dbs_regression",
    "id_regression",
    "cnv_regression",
)


def load_signature_labels(
    labels_path: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """
    Load a COSMIC signature exposure table into ``(submitter_id -> vector, signature_names)``.

    The table is a TSV with one row per submitter and one column per COSMIC
    signature. The first column is treated as the submitter id (renamed to
    ``submitter_id`` if necessary); the remaining columns are stored as a
    per-submitter ``float32`` vector.

    Works generically for any signature class produced by
    :func:`scripts.data_handling.extract_signatures.extract_signature_labels`
    (SBS, DBS, ID, CN).

    Parameters
    ----------
    labels_path:
        Path to the contribution TSV.
    logger:
        Optional logger; a row count summary is emitted when provided.

    Returns
    -------
    tuple[dict[str, np.ndarray], tuple[str, ...]]
        Mapping from ``submitter_id`` to exposure vector, and a tuple of the
        signature column names in vector order.
    """
    labels_df = pd.read_table(labels_path)
    if "submitter_id" not in labels_df.columns:
        labels_df = labels_df.rename(columns={labels_df.columns[0]: "submitter_id"})
    labels_df["submitter_id"] = labels_df["submitter_id"].astype(str).str.strip()

    signature_columns = [
        column for column in labels_df.columns if column != "submitter_id"
    ]
    if not signature_columns:
        raise ValueError(f"Signature labels file has no label columns: {labels_path}")

    label_matrix = labels_df[signature_columns].to_numpy(dtype=np.float32)
    submitter_ids = labels_df["submitter_id"].tolist()
    submitter_labels = {
        submitter_id: label_matrix[idx]
        for idx, submitter_id in enumerate(submitter_ids)
    }
    if logger is not None:
        logger.info(
            "Loaded signature labels from %s: %d submitter(s), %d signature column(s).",
            labels_path,
            len(submitter_labels),
            len(signature_columns),
        )
    return submitter_labels, tuple(signature_columns)
