"""Cancer subtyping label utilities for slide-level classification."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

UNKNOWN_SUBTYPE_CLASS = "Unknown"


def normalize_subtype_label(value: Any) -> str:
    """
    Normalize a TCGA histologic-subtype value into a class label.

    Parameters
    ----------
    value:
        Raw label from the subtype column. Missing / placeholder values are
        coerced to :data:`UNKNOWN_SUBTYPE_CLASS`.

    Returns
    -------
    str
        Canonical subtype label.
    """
    label = str(value).strip()
    if label.lower() == "nan" or label in {"", "[Not Available]", "[Not Applicable]"}:
        return UNKNOWN_SUBTYPE_CLASS

    parts = [part.strip() for part in label.split("|") if part.strip()]
    unique_parts = list(dict.fromkeys(parts))
    if len(unique_parts) == 1:
        label = unique_parts[0]

    if label == UNKNOWN_SUBTYPE_CLASS:
        return UNKNOWN_SUBTYPE_CLASS
    return label


def load_subtyping_labels(
    labels_path: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """
    Load slide-level subtyping labels.

    Rows whose normalized subtype is :data:`UNKNOWN_SUBTYPE_CLASS` (missing /
    placeholder values, or genuinely labelled ``Unknown``) are skipped, so
    those submitters are absent from the returned mapping and their slides
    are dropped downstream by ``_get_slide_splits``. Submitters with
    conflicting real subtype labels across rows are likewise dropped.

    Parameters
    ----------
    labels_path:
        Path to a TSV file with at least ``submitter_id`` and ``subtype``
        columns.
    logger:
        Optional logger; warnings are emitted when a submitter has conflicting
        labels in the table.

    Returns
    -------
    tuple[dict[str, int], tuple[str, ...]]
        Mapping from ``submitter_id`` to class index, and a tuple of class
        names in index order. Indices are assigned in the order each class
        is first observed; ``Unknown`` is not represented.
    """
    labels_df = pd.read_table(labels_path, dtype=str)
    required_columns = {"submitter_id", "subtype"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        raise ValueError(
            f"Subtyping label table is missing column(s): {sorted(missing_columns)}"
        )

    submitter_ids = labels_df["submitter_id"].astype(str).str.strip()
    raw_values = labels_df["subtype"]

    class_to_index: dict[str, int] = {}
    submitter_labels: dict[str, int] = {}
    raw_submitter_labels: dict[str, str] = {}
    dropped_submitters: set[str] = set()

    for submitter_id, raw_label in zip(submitter_ids, raw_values):
        if submitter_id in dropped_submitters:
            continue

        label = normalize_subtype_label(raw_label)
        if label == UNKNOWN_SUBTYPE_CLASS:
            # No informative label in this row; don't add the submitter. If a
            # prior row already gave this submitter a real label, that label
            # stays untouched.
            continue

        if label not in class_to_index:
            class_to_index[label] = len(class_to_index)
        class_index = class_to_index[label]

        previous_label = raw_submitter_labels.get(submitter_id)
        if previous_label is not None and previous_label != label:
            if logger is not None:
                logger.warning(
                    "Conflicting subtyping labels for submitter %s: %s vs %s. "
                    "Dropping this submitter from the label table.",
                    submitter_id,
                    previous_label,
                    label,
                )
            submitter_labels.pop(submitter_id, None)
            raw_submitter_labels.pop(submitter_id, None)
            dropped_submitters.add(submitter_id)
            continue

        submitter_labels[submitter_id] = class_index
        raw_submitter_labels[submitter_id] = label

    if not submitter_labels:
        raise RuntimeError(f"No TCGA subtyping labels were found in: {labels_path}")

    # Drop classes that ended up with zero surviving submitters (introduced
    # only via rows that were later dropped via conflict resolution), and
    # densely renumber the survivors in first-seen order.
    surviving_indices = set(submitter_labels.values())
    if len(surviving_indices) != len(class_to_index):
        ordered_names = [
            name for name, idx in class_to_index.items() if idx in surviving_indices
        ]
        index_remap = {
            class_to_index[name]: new_idx for new_idx, name in enumerate(ordered_names)
        }
        submitter_labels = {
            sid: index_remap[idx] for sid, idx in submitter_labels.items()
        }
        class_to_index = {name: i for i, name in enumerate(ordered_names)}

    return submitter_labels, tuple(class_to_index)
