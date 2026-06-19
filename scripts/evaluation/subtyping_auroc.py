"""Compute one-vs-rest AUROC for slide-level subtyping predictions.

Reads one or more ``subtyping.txt`` prediction tables produced by
``scripts/evaluation/slide_results.py`` (via ``cloud/07_predict_slide.sh``).
Each file is a TSV with columns::

    slide_id  true_label  pred_label  <class_0>  <class_1>  ...  <class_n>

where the trailing columns are the per-class probabilities (e.g.
``BRCA_LumA``, ``BRCA_Her2``, ...) and sum to 1 per row. ``true_label`` /
``pred_label`` hold one of those class names.

Passing several files pools their rows first (e.g. to aggregate out-of-fold
test predictions across CV folds), then computes:

  * per-class one-vs-rest AUROC + its positive support,
  * macro AUROC (unweighted mean over classes with both pos/neg present),
  * support-weighted AUROC,
  * accuracy and balanced accuracy (from ``pred_label``, for context).

AUROC is the rank statistic (Mann-Whitney U with average-rank tie handling),
identical to ``sklearn.metrics.roc_auc_score`` for the binary OvR case, but
implemented with numpy so this script needs no scikit-learn.

Examples
--------
    python scripts/evaluation/subtyping_auroc.py \
        outputs/evaluation/clam-mb-resnet50-full-fold0/subtyping.txt

    # pool all 5 folds and bootstrap 95% CIs
    python scripts/evaluation/subtyping_auroc.py \
        outputs/evaluation/clam-mb-resnet50-full-fold*/subtyping.txt \
        --bootstrap 1000

    # 4-class view: drop Normal-like rows + column and renormalize
    python scripts/evaluation/subtyping_auroc.py SUBTYPING.txt \
        --exclude-classes Normal
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np
import pandas as pd

_META_COLUMNS = ("slide_id", "true_label", "pred_label")


def _average_ranks(sorted_values: np.ndarray) -> np.ndarray:
    """1-based average ranks for an already-sorted 1-D array (ties share rank)."""
    n = sorted_values.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """One-vs-rest AUROC via the rank statistic; NaN if a class is degenerate."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = _average_ranks(scores[order])
    sum_ranks_pos = ranks[y_true].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_predictions(paths: Sequence[str]) -> pd.DataFrame:
    """Read and row-concatenate one or more prediction TSVs."""
    frames: list[pd.DataFrame] = []
    columns: tuple[str, ...] | None = None
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        if columns is None:
            columns = tuple(frame.columns)
        elif tuple(frame.columns) != columns:
            raise ValueError(
                f"Column mismatch in {path}.\n  expected: {columns}\n  found:    "
                f"{tuple(frame.columns)}"
            )
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    missing = [c for c in _META_COLUMNS if c not in pooled.columns]
    if missing:
        raise ValueError(f"Prediction file(s) missing required column(s): {missing}")
    return pooled


def class_columns(frame: pd.DataFrame) -> list[str]:
    """Probability columns = everything that isn't a metadata column."""
    return [c for c in frame.columns if c not in _META_COLUMNS]


def _resolve_excluded(classes: Sequence[str], tokens: Sequence[str]) -> set[str]:
    """Match exclude tokens against class columns (exact, suffix, or BRCA_ prefix)."""
    def _matches(token: str, cls: str) -> bool:
        return token == cls or cls.split("_")[-1] == token or f"BRCA_{token}" == cls

    excluded = {cls for token in tokens for cls in classes if _matches(token, cls)}
    unmatched = [t for t in tokens if not any(_matches(t, c) for c in classes)]
    if unmatched:
        raise ValueError(
            f"--exclude-classes value(s) {unmatched} matched no class in {list(classes)}"
        )
    return excluded


def apply_exclusion(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    """Drop excluded-class rows + columns, renormalize remaining probabilities."""
    if not excluded:
        return frame
    kept_classes = [c for c in class_columns(frame) if c not in excluded]
    out = frame[~frame["true_label"].isin(excluded)].copy()
    probs = out[kept_classes].to_numpy(dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out[kept_classes] = probs / row_sums
    return out[list(_META_COLUMNS) + kept_classes]


def per_class_auroc(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-class OvR AUROC and positive support."""
    true = frame["true_label"].to_numpy()
    result: dict[str, dict[str, float]] = {}
    for cls in class_columns(frame):
        y_true = true == cls
        result[cls] = {
            "support": int(y_true.sum()),
            "auroc": binary_auroc(y_true, frame[cls].to_numpy()),
        }
    return result


def macro_weighted(per_class: dict[str, dict[str, float]]) -> tuple[float, float]:
    """Macro (unweighted) and support-weighted AUROC over valid classes."""
    valid = {c: v for c, v in per_class.items() if not np.isnan(v["auroc"])}
    if not valid:
        return float("nan"), float("nan")
    aurocs = np.array([v["auroc"] for v in valid.values()])
    supports = np.array([v["support"] for v in valid.values()], dtype=np.float64)
    macro = float(aurocs.mean())
    weighted = (
        float(np.average(aurocs, weights=supports))
        if supports.sum() > 0
        else float("nan")
    )
    return macro, weighted


def argmax_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    """Accuracy and balanced accuracy from pred_label (argmax decision)."""
    true = frame["true_label"].to_numpy()
    pred = frame["pred_label"].to_numpy()
    accuracy = float((true == pred).mean())
    recalls = [
        float((pred[true == cls] == cls).mean())
        for cls in class_columns(frame)
        if (true == cls).any()
    ]
    balanced = float(np.mean(recalls)) if recalls else float("nan")
    return accuracy, balanced


def _summarize_samples(values: list[float]) -> dict | None:
    """Std (standard error of the estimate) + 95% percentile CI of bootstrap draws."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return {
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
    }


def bootstrap_stats(
    frame: pd.DataFrame, classes: list[str], n_boot: int, seed: int
) -> dict | None:
    """Bootstrap std + 95% CI for each AUROC metric by resampling rows.

    ``std`` is the bootstrap standard deviation (the standard error of the
    estimate, i.e. the ``±`` in ``0.65 ±0.02``); ``ci95`` is the 2.5/97.5
    percentile interval. Returns ``None`` when bootstrapping is disabled.
    """
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(frame)
    per_class_samples: dict[str, list[float]] = {c: [] for c in classes}
    macro_samples: list[float] = []
    weighted_samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        per_class = per_class_auroc(frame.iloc[idx])
        for cls in classes:
            per_class_samples[cls].append(per_class[cls]["auroc"])
        macro, weighted = macro_weighted(per_class)
        macro_samples.append(macro)
        weighted_samples.append(weighted)
    return {
        "per_class": {c: _summarize_samples(per_class_samples[c]) for c in classes},
        "macro": _summarize_samples(macro_samples),
        "weighted": _summarize_samples(weighted_samples),
    }


def build_report(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Assemble the metrics dict for one (possibly pooled) prediction set."""
    classes = class_columns(frame)
    per_class = per_class_auroc(frame)
    macro, weighted = macro_weighted(per_class)
    accuracy, balanced = argmax_metrics(frame)
    report = {
        "n_slides": int(len(frame)),
        "classes": classes,
        "per_class": per_class,
        "macro_auroc": macro,
        "weighted_auroc": weighted,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }
    stats = bootstrap_stats(frame, classes, n_boot, seed)
    if stats is not None:
        report["n_bootstrap"] = n_boot
        for cls in classes:
            summary = stats["per_class"].get(cls)
            if summary is not None:
                report["per_class"][cls]["std"] = summary["std"]
                report["per_class"][cls]["ci95"] = summary["ci95"]
        if stats["macro"] is not None:
            report["macro_auroc_std"] = stats["macro"]["std"]
            report["macro_auroc_ci95"] = stats["macro"]["ci95"]
        if stats["weighted"] is not None:
            report["weighted_auroc_std"] = stats["weighted"]["std"]
            report["weighted_auroc_ci95"] = stats["weighted"]["ci95"]
    return report


def print_report(report: dict) -> None:
    """Human-readable table to stdout."""
    has_std = "macro_auroc_std" in report
    print(f"\nslides: {report['n_slides']}   classes: {len(report['classes'])}\n")
    auroc_header = "AUROC ±std" if has_std else "AUROC"
    header = f"{'class':<16}{'support':>9}{auroc_header:>18}"
    print(header)
    print("-" * len(header))
    for cls, vals in report["per_class"].items():
        auroc = vals["auroc"]
        if np.isnan(auroc):
            cell = "n/a"
        elif has_std and vals.get("std") is not None:
            cell = f"{auroc:.4f} ±{vals['std']:.4f}"
        else:
            cell = f"{auroc:.4f}"
        print(f"{cls:<16}{vals['support']:>9}{cell:>18}")
    print("-" * len(header))
    macro_line = f"macro AUROC      : {report['macro_auroc']:.4f}"
    if has_std:
        lo, hi = report["macro_auroc_ci95"]
        macro_line += f" ±{report['macro_auroc_std']:.4f}   (95% CI {lo:.4f}-{hi:.4f})"
    print(macro_line)
    weighted_line = f"weighted AUROC   : {report['weighted_auroc']:.4f}"
    if has_std and report.get("weighted_auroc_std") is not None:
        weighted_line += f" ±{report['weighted_auroc_std']:.4f}"
    print(weighted_line)
    print(f"accuracy         : {report['accuracy']:.4f}")
    print(f"balanced accuracy: {report['balanced_accuracy']:.4f}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute one-vs-rest AUROC for subtyping predictions.",
    )
    parser.add_argument(
        "pred_paths",
        nargs="+",
        help="One or more subtyping.txt prediction TSVs (pooled if multiple).",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=[],
        metavar="CLASS",
        help="Class name(s) to drop (rows + column), e.g. 'Normal' for a 4-class view.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for +-std and 95%% CI on the AUROC metrics (0 = off).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--save", default=None, help="Optional path to write the metrics as JSON."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_predictions(args.pred_paths)
    if args.exclude_classes:
        excluded = _resolve_excluded(class_columns(frame), args.exclude_classes)
        frame = apply_exclusion(frame, excluded)
        print(f"excluded classes: {sorted(excluded)}")
    report = build_report(frame, args.bootstrap, args.seed)
    print_report(report)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
