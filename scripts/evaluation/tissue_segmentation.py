"""Tissue-segmentation metrics across one or more tile-encoder configurations.

Loads each requested tile model from a YAML config + checkpoint, runs its
``tissue_segmentation`` decoder over the held-out BCSS test split, and reports
macro-averaged and per-class Dice / IoU / recall over BCSS tissue classes.
Class indices listed in ``--exclude-class-indices`` (default ``[0]``, which is
typically ``outside_roi``) are dropped from the macro average; the per-class
table still reports every class so their behaviour stays visible.

Model and dataset loading mirror :mod:`scripts.evaluation.tile_feature_space`
-- helpers are duplicated to keep each script self-contained. The same test
split and ``T * portion_per_sample`` sampling rule are used so models can be
compared on identical tiles.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from openslide import OpenSlide
from torch import Tensor
from torch.utils.data import DataLoader

from augur.datasets.factory import get_dataset_from_config
from augur.datasets.tcga_tile_dataset import (  # noqa: F401  (private inner Dataset reused for test sampling)
    TCGATileDataset,
    _TileDataset,
)
from augur.datasets.utils import (
    SlideRecord,
    TileRecord,
    derive_tissue_slide_name,
    get_slide_mpp,
    resolve_tissue_label_metadata_path,
)
from augur.models.tile_level.tile_model import TileModel
from augur.utils.config import load_yaml_config


def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the evaluation run."""
    log_dir = os.path.join("logs", "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "tissue_segmentation.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("tissue_segmentation")
    logger.handlers = [handler, logging.StreamHandler()]
    logger.setLevel(logging.INFO)
    return logger


def _resolve_relative_config_path(value: Any, *, config_dir: str) -> Any:
    """Resolve a relative config path against the parent config directory."""
    if not isinstance(value, str) or os.path.isabs(value):
        return value
    candidate = os.path.join(config_dir, value)
    return candidate if os.path.exists(candidate) else value


def _resolve_model_component_paths(
    config: dict[str, Any], *, config_path: str
) -> dict[str, Any]:
    """Resolve nested encoder/decoder config paths relative to the model config."""
    resolved = dict(config)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    params = resolved.get("params", {})
    if not isinstance(params, dict):
        return resolved
    resolved_params = dict(params)
    if "encoder_config" in resolved_params:
        resolved_params["encoder_config"] = _resolve_relative_config_path(
            resolved_params["encoder_config"], config_dir=config_dir
        )
    decoders_config = resolved_params.get("decoders_config")
    if isinstance(decoders_config, dict):
        resolved_params["decoders_config"] = {
            task: _resolve_relative_config_path(spec, config_dir=config_dir)
            for task, spec in decoders_config.items()
        }
    resolved["params"] = resolved_params
    return resolved


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Normalize a Lightning checkpoint or plain state dict into a parameter mapping."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must deserialize into a dict-like object.")
    if isinstance(checkpoint.get("state_dict"), dict):
        return dict(checkpoint["state_dict"])
    if isinstance(checkpoint.get("model_state_dict"), dict):
        return dict(checkpoint["model_state_dict"])
    return dict(checkpoint)


def _create_tile_model(
    *,
    model_config_path: str,
    checkpoint_path: str | None,
    device: torch.device,  # type: ignore
    logger: logging.Logger,
) -> TileModel:
    """Build a ``TileModel`` from a YAML config and load weights from a checkpoint."""
    config = _resolve_model_component_paths(
        load_yaml_config(model_config_path),
        config_path=model_config_path,
    )
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Model config 'params' must be a dict.")
    model = TileModel.from_config(params)

    resolved_ckpt = checkpoint_path or config.get("checkpoint_path")
    if not resolved_ckpt:
        raise ValueError(
            "No checkpoint path was provided via --checkpoint-paths or model config."
        )
    resolved_ckpt = os.path.abspath(resolved_ckpt)
    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    checkpoint = torch.load(resolved_ckpt, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded weights from %s", resolved_ckpt)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    return model.to(device).eval()


def _load_class_names(root_dir: str, logger: logging.Logger) -> list[str]:
    """Return BCSS tissue class labels indexed by their integer GT codes."""
    gt_path, _, _ = resolve_tissue_label_metadata_path(root_dir=root_dir, logger=logger)
    df = pd.read_csv(gt_path, sep="\t", dtype=str)
    code_col = "GT_code" if "GT_code" in df.columns else "gt_code"
    df[code_col] = df[code_col].astype(int)
    df = df.sort_values(code_col).reset_index(drop=True)
    n_classes = int(df[code_col].max()) + 1  # type: ignore
    names: list[str] = [f"class_{i}" for i in range(n_classes)]
    for _, row in df.iterrows():
        names[int(row[code_col])] = str(row["label"])  # type: ignore
    return names


def _derive_model_name(model_config_path: str) -> str:
    """Extract a short model name from the model-config filename.

    ``configs/model-resnet50-magnification.yaml`` -> ``resnet50-magnification``.
    Filenames not following the ``model-<name>.yaml`` convention return their
    stem unchanged.
    """
    stem = os.path.splitext(os.path.basename(model_config_path))[0]
    return stem[len("model-") :] if stem.startswith("model-") else stem


def _enumerate_bcss_grid_records(
    *,
    slide_record: SlideRecord,
    slide_rois: pd.DataFrame,
    base_mpp: float,
    output_size: int,
    stride: int,
    logger: logging.Logger,
) -> list[TileRecord]:
    """Enumerate non-overlapping BCSS-ROI tile records on a regular grid."""
    slide = OpenSlide(slide_record.slide_path)
    try:
        slide_mpp = get_slide_mpp(slide, logger)
        target_downsample = base_mpp / slide_mpp
        level = slide.get_best_level_for_downsample(target_downsample)
        actual_downsample = float(slide.level_downsamples[level])
        read_size = max(
            int(round(output_size * target_downsample / actual_downsample)), 1
        )
        tile_extent_l0 = read_size * actual_downsample
        stride_l0 = max(int(round(stride * (base_mpp / slide_mpp))), 1)
        width_l0, height_l0 = slide.dimensions
        max_x = max(int(round(width_l0 - tile_extent_l0)), 0)
        max_y = max(int(round(height_l0 - tile_extent_l0)), 0)
        half_extent = tile_extent_l0 / 2.0

        records: list[TileRecord] = []
        for _, row in slide_rois.iterrows():
            xmin = int(row["xmin"])  # type: ignore
            ymin = int(row["ymin"])  # type: ignore
            xmax = int(row["xmax"])  # type: ignore
            ymax = int(row["ymax"])  # type: ignore
            roi_name = str(row["slide_name"])

            cx_start = int(math.ceil(xmin + half_extent))
            cx_end = int(math.floor(xmax - half_extent))
            cy_start = int(math.ceil(ymin + half_extent))
            cy_end = int(math.floor(ymax - half_extent))
            if cx_end < cx_start or cy_end < cy_start:
                continue

            for cy in range(cy_start, cy_end + 1, stride_l0):
                for cx in range(cx_start, cx_end + 1, stride_l0):
                    x = min(max(int(round(cx - half_extent)), 0), max_x)
                    y = min(max(int(round(cy - half_extent)), 0), max_y)
                    records.append(
                        TileRecord(
                            slide_id=slide_record.slide_id,
                            submitter_id=slide_record.submitter_id,
                            slide_path=slide_record.slide_path,
                            x=x,
                            y=y,
                            level=level,
                            size=read_size,
                            roi_name=roi_name,
                            roi_xmin=xmin,
                            roi_ymin=ymin,
                            roi_xmax=xmax,
                            roi_ymax=ymax,
                        )
                    )
        return records
    finally:
        slide.close()


def _build_test_records_with_portion(
    *,
    datamodule: TCGATileDataset,
    portion_per_sample: float,
    stride: int,
    seed: int,
    logger: logging.Logger,
) -> list[TileRecord]:
    """Sample ``floor(T * portion_per_sample)`` BCSS tile records per test slide."""
    if not 0.0 < portion_per_sample <= 1.0:
        raise ValueError(
            f"portion_per_sample must be in (0, 1]. Got: {portion_per_sample}"
        )
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer. Got: {stride}")

    splits = datamodule._get_slide_splits()
    test_slides = splits["test"]
    _, _, roi_df = datamodule._load_tissue_metadata()
    roi_groups: dict[str, pd.DataFrame] = {
        str(slide_name): group.reset_index(drop=True)
        for slide_name, group in roi_df.groupby("slide_name", sort=False)
    }

    rng = random.Random(seed)
    all_records: list[TileRecord] = []
    total = len(test_slides)
    for index, slide_record in enumerate(test_slides, start=1):
        slide_name = derive_tissue_slide_name(
            slide_record.slide_path, slide_record.submitter_id
        )
        slide_rois = roi_groups.get(slide_name)
        if slide_rois is None or slide_rois.empty:
            logger.warning(
                "No BCSS ROI bounds for test slide %s; skipping.",
                slide_record.slide_id,
            )
            continue

        candidates = _enumerate_bcss_grid_records(
            slide_record=slide_record,
            slide_rois=slide_rois,
            base_mpp=datamodule.base_mpp,
            output_size=datamodule.tile_size,
            stride=stride,
            logger=logger,
        )
        candidate_count = len(candidates)
        if candidate_count == 0:
            logger.warning(
                "No BCSS grid centers fit for test slide %s; skipping.",
                slide_record.slide_id,
            )
            continue
        sample_size = max(1, int(candidate_count * portion_per_sample))
        sample_size = min(sample_size, candidate_count)
        sampled = rng.sample(candidates, sample_size)
        all_records.extend(sampled)

        if index == 1 or index == total or index % 25 == 0:
            logger.info(
                "Test slide %d/%d (%s): T=%d, sampled %d.",
                index,
                total,
                slide_record.slide_id,
                candidate_count,
                sample_size,
            )

    if not all_records:
        raise RuntimeError("No tile records sampled from the held-out test set.")
    return all_records


def _accumulate_confusion(
    *,
    model: TileModel,
    test_loader: DataLoader,
    device: torch.device,  # type: ignore
    n_classes: int,
    max_batches: int | None,
    logger: logging.Logger,
) -> np.ndarray:
    """Run the model over the test loader and return a ``(C, C)`` pixel confusion matrix.

    Rows are GT classes (argmax of the one-hot target mask), columns are
    predicted classes (argmax of the decoder logits). One increment per
    pixel across every tile.
    """
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            task_batch = batch["tissue_segmentation"]
            image_batch = task_batch["image"].to(device, non_blocking=True)
            target_batch = task_batch["target"].to(device, non_blocking=True)

            encoded = model.encoder(image_batch)
            logits = model.decoders["tissue_segmentation"](encoded)
            if not isinstance(logits, Tensor) or logits.ndim != 4:
                raise ValueError(
                    "tissue_segmentation decoder must return a (B, C, H, W) tensor; "
                    f"got {type(logits).__name__} with shape "
                    f"{tuple(logits.shape) if isinstance(logits, Tensor) else 'n/a'}."
                )

            gt_pixels = target_batch.argmax(dim=1).reshape(-1)
            pred_pixels = logits.argmax(dim=1).reshape(-1)
            pixel_idx = (gt_pixels * n_classes + pred_pixels).to(torch.int64)
            batch_counts = torch.bincount(pixel_idx, minlength=n_classes * n_classes)
            confusion += batch_counts.cpu().numpy().reshape(n_classes, n_classes)

            if batch_idx % 10 == 0:
                logger.info("Processed batch %d", batch_idx)
    return confusion


def _compute_segmentation_metrics(
    confusion: np.ndarray,
    class_names: list[str],
    *,
    exclude_indices: set[int],
) -> dict[str, Any]:
    """Derive per-class Dice / IoU / recall and macro averages from a confusion matrix.

    A class with zero GT pixels has ``recall=None`` (and is skipped in the
    macro average for that metric). A class with zero GT and zero predicted
    pixels has ``dice=None`` and ``iou=None`` (also skipped).
    Classes in ``exclude_indices`` are dropped from every macro average,
    regardless of GT support.
    """
    n_classes = confusion.shape[0]
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    gt_total = tp + fn
    pred_total = tp + fp

    per_class: dict[str, dict[str, Any]] = {}
    dice_values: list[float] = []
    iou_values: list[float] = []
    recall_values: list[float] = []
    for c in range(n_classes):
        denom_dice = 2 * tp[c] + fp[c] + fn[c]
        denom_iou = tp[c] + fp[c] + fn[c]
        dice = float(2 * tp[c] / denom_dice) if denom_dice > 0 else None
        iou = float(tp[c] / denom_iou) if denom_iou > 0 else None
        recall = float(tp[c] / gt_total[c]) if gt_total[c] > 0 else None
        per_class[class_names[c]] = {
            "index": c,
            "n_gt_pixels": int(gt_total[c]),
            "n_pred_pixels": int(pred_total[c]),
            "dice": dice,
            "iou": iou,
            "recall": recall,
        }
        if c in exclude_indices:
            continue
        if dice is not None:
            dice_values.append(dice)
        if iou is not None:
            iou_values.append(iou)
        if recall is not None:
            recall_values.append(recall)

    macro = {
        "dice": float(np.mean(dice_values)) if dice_values else None,
        "iou": float(np.mean(iou_values)) if iou_values else None,
        "recall": float(np.mean(recall_values)) if recall_values else None,
        "n_classes_in_average": len(dice_values),
        "excluded_indices": sorted(exclude_indices),
    }
    return {"macro_average": macro, "per_class": per_class}


def _fmt_metric(value: float | None) -> str:
    """Format a metric for log alignment, returning ``" n/a "`` when ``None``."""
    return f"{value:.4f}" if value is not None else " n/a  "


def _log_model_summary(
    model_name: str, report: dict[str, Any], logger: logging.Logger
) -> None:
    """Pretty-print one model's per-class table and macro headline."""
    macro = report["macro_average"]
    logger.info("=" * 72)
    logger.info("Model: %s", model_name)
    logger.info(
        "Macro avg (excl. indices %s, n=%d): dice=%s  iou=%s  recall=%s",
        macro["excluded_indices"],
        macro["n_classes_in_average"],
        _fmt_metric(macro["dice"]),
        _fmt_metric(macro["iou"]),
        _fmt_metric(macro["recall"]),
    )
    logger.info(
        "  %-30s  %-12s  %-7s  %-7s  %-7s",
        "class",
        "n_gt_pixels",
        "dice",
        "iou",
        "recall",
    )
    for name, entry in report["per_class"].items():
        logger.info(
            "  %-30s  %-12d  %s  %s  %s",
            name,
            entry["n_gt_pixels"],
            _fmt_metric(entry["dice"]),
            _fmt_metric(entry["iou"]),
            _fmt_metric(entry["recall"]),
        )


def _log_comparison_table(
    all_reports: dict[str, dict[str, Any]], logger: logging.Logger
) -> None:
    """Log a side-by-side macro comparison of every evaluated model."""
    logger.info("=" * 72)
    logger.info("Cross-model macro comparison:")
    logger.info("  %-35s  %-7s  %-7s  %-7s", "model", "dice", "iou", "recall")
    for model_name, report in all_reports.items():
        macro = report["macro_average"]
        logger.info(
            "  %-35s  %s  %s  %s",
            model_name,
            _fmt_metric(macro["dice"]),
            _fmt_metric(macro["iou"]),
            _fmt_metric(macro["recall"]),
        )


def evaluate(
    *,
    model_config_paths: list[str],
    dataset_config_path: str,
    checkpoint_paths: list[str] | None,
    exclude_class_indices: list[int],
    device_str: str,
    output_dir: str,
    output_basename: str,
    max_batches: int | None,
    portion_per_sample: float,
    stride: int | None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate every model on a shared test split and persist per-model + summary reports."""
    logger = _setup_logger()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        device = torch.device(device_str)  # type: ignore
    logger.info("Using device: %s", device)

    if not model_config_paths:
        raise ValueError("At least one --model-configs entry is required.")
    if checkpoint_paths is not None and len(checkpoint_paths) != len(
        model_config_paths
    ):
        raise ValueError(
            "--checkpoint-paths must have the same number of entries as "
            f"--model-configs (got {len(checkpoint_paths)} vs "
            f"{len(model_config_paths)})."
        )
    logger.info(
        "Evaluating %d model(s): %s", len(model_config_paths), model_config_paths
    )

    dataset_config = load_yaml_config(dataset_config_path)
    datamodule = get_dataset_from_config(dataset_config)
    if "tissue_segmentation" not in getattr(datamodule, "tasks", ()):
        raise ValueError(
            "Dataset must include 'tissue_segmentation' to provide GT masks."
        )
    if not isinstance(datamodule, TCGATileDataset):
        raise TypeError(
            "Expected a TCGATileDataset; the T*portion_per_sample sampling reuses "
            "its BCSS metadata helpers."
        )
    datamodule.prepare_data()

    effective_stride = stride if stride is not None else datamodule.tile_size
    test_records = _build_test_records_with_portion(
        datamodule=datamodule,
        portion_per_sample=portion_per_sample,
        stride=effective_stride,
        seed=seed,
        logger=logger,
    )
    datamodule.test_dataset = _TileDataset(
        records=test_records,
        tasks=["tissue_segmentation"],
        tile_size=datamodule.tile_size,
        image_size=datamodule.image_size,
        base_mpp=datamodule.base_mpp,
        root_dir=datamodule.root_dir,
        tissue_segmentation_n_classes=len(datamodule._tissue_gt_codes or {}),
        random_seed=datamodule.random_seed,
        logger=logger,
    )
    logger.info(
        "Test dataset size: %d tiles (portion=%.3f, stride=%d)",
        len(datamodule.test_dataset),  # type: ignore
        portion_per_sample,
        effective_stride,
    )

    class_names = _load_class_names(datamodule.root_dir, logger)  # type: ignore
    n_classes = len(class_names)
    exclude_set = set(exclude_class_indices)
    invalid = [i for i in exclude_set if i < 0 or i >= n_classes]
    if invalid:
        raise ValueError(
            f"--exclude-class-indices entries {invalid} are out of range for "
            f"n_classes={n_classes}."
        )
    logger.info(
        "Loaded %d tissue classes; excluding indices %s from macro averages.",
        n_classes,
        sorted(exclude_set),
    )

    os.makedirs(output_dir, exist_ok=True)
    all_reports: dict[str, dict[str, Any]] = {}
    ckpt_iter = (
        checkpoint_paths
        if checkpoint_paths is not None
        else [None] * len(model_config_paths)
    )
    for mc_path, ckpt in zip(model_config_paths, ckpt_iter):
        model_name = _derive_model_name(mc_path)
        logger.info("--- Evaluating model: %s ---", model_name)
        model = _create_tile_model(
            model_config_path=mc_path,
            checkpoint_path=ckpt,
            device=device,
            logger=logger,
        )
        if "tissue_segmentation" not in model.decoders:
            raise ValueError(
                f"Model {model_name} must expose a 'tissue_segmentation' decoder. "
                f"Found: {sorted(model.decoders.keys())}"
            )
        test_loader = datamodule.test_dataloader()
        confusion = _accumulate_confusion(
            model=model,
            test_loader=test_loader,
            device=device,
            n_classes=n_classes,
            max_batches=max_batches,
            logger=logger,
        )
        metrics = _compute_segmentation_metrics(
            confusion, class_names, exclude_indices=exclude_set
        )
        report: dict[str, Any] = {
            "config": {
                "model_config": os.path.abspath(mc_path),
                "dataset_config": os.path.abspath(dataset_config_path),
                "checkpoint_path": os.path.abspath(ckpt) if ckpt else None,
                "portion_per_sample": float(portion_per_sample),
                "stride": int(effective_stride),
                "seed": int(seed),
                "n_classes": int(n_classes),
                "excluded_class_indices": sorted(exclude_set),
            },
            "macro_average": metrics["macro_average"],
            "per_class": metrics["per_class"],
            "confusion_matrix": confusion.astype(np.int64).tolist(),
            "class_names": list(class_names),
        }
        per_model_path = os.path.join(
            output_dir, f"{output_basename}_{model_name}.json"
        )
        with open(per_model_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Wrote report to %s", per_model_path)
        _log_model_summary(model_name, report, logger)
        all_reports[model_name] = report

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "dataset_config": os.path.abspath(dataset_config_path),
        "portion_per_sample": float(portion_per_sample),
        "stride": int(effective_stride),
        "seed": int(seed),
        "excluded_class_indices": sorted(exclude_set),
        "models": {
            model_name: {
                "model_config": report["config"]["model_config"],
                "checkpoint_path": report["config"]["checkpoint_path"],
                "macro_average": report["macro_average"],
            }
            for model_name, report in all_reports.items()
        },
    }
    summary_path = os.path.join(output_dir, f"{output_basename}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote cross-model summary to %s", summary_path)
    _log_comparison_table(all_reports, logger)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the evaluation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-configs",
        nargs="+",
        required=True,
        help="One or more model YAML configs to evaluate on the same test split.",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Single dataset YAML config; must include 'tissue_segmentation' as a task.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="*",
        default=None,
        help=(
            "Optional per-model checkpoint overrides. If supplied, must have "
            "exactly one entry per --model-configs entry."
        ),
    )
    parser.add_argument(
        "--exclude-class-indices",
        nargs="*",
        type=int,
        default=[0],
        help=(
            "Class indices to exclude from the macro averages. Defaults to "
            "[0] since BCSS class 0 ('outside_roi') is not a tissue class."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument(
        "--output-basename",
        default="tissue_segmentation",
        help=(
            "Outputs are written as <basename>_<model-name>.json per model "
            "plus a combined <basename>_summary.json. <model-name> is derived "
            "from each --model-configs entry (e.g. "
            "'configs/model-resnet50-magnification.yaml' -> "
            "'resnet50-magnification')."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of test batches processed per model.",
    )
    parser.add_argument(
        "--portion-per-sample",
        type=float,
        default=1.0,
        help=(
            "Fraction of grid centers per held-out test slide to keep. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Spacing between grid centers in pixels at base_mpp. "
            "Defaults to the dataset's tile_size (non-overlapping)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """CLI entrypoint: parse arguments and run :func:`evaluate`."""
    args = _build_arg_parser().parse_args()
    evaluate(
        model_config_paths=args.model_configs,
        dataset_config_path=args.dataset_config,
        checkpoint_paths=args.checkpoint_paths,
        exclude_class_indices=args.exclude_class_indices,
        device_str=args.device,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        max_batches=args.max_batches,
        portion_per_sample=args.portion_per_sample,
        stride=args.stride,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
