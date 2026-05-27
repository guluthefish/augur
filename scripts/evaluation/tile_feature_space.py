"""Tile-level feature-space metrics on the held-out test split.

Reuses the embedding-extraction and tissue-class assignment pipeline from
:mod:`scripts.visualization.tile_embeddings` (helpers are duplicated to keep
each script self-contained) but operates on the full-dimensional embeddings
(no PCA/UMAP) and reports four metrics per configuration:

1. Class separation -- silhouette score (cosine) and Davies-Bouldin index
   over tile embeddings, using the dominant GT tissue class as the label.
2. Within-class compactness -- mean intra-class cosine distance per class,
   plus the macro average.
3. Agnostic tile fraction -- share of tiles whose dominant-class portion
   in the GT mask falls below ``--dominance-threshold`` (default 0.5).
4. Failure modes -- top tissue-class confusion pairs from both pixel-level
   and tile-level confusion matrices between GT and model segmentation
   predictions.

Per-tile selection mirrors :mod:`scripts.visualization.tile_embeddings`:
candidates within each held-out slide's BCSS ROIs are enumerated on a regular
grid and ``floor(T * portion_per_sample)`` of them are sampled without
replacement.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openslide import OpenSlide
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch import Tensor

from augur.datasets.factory import get_dataset_from_config
from augur.datasets.tcga_tile_dataset import (  # noqa: F401  (private inner Dataset reused for test sampling)
    TCGATileDataset,
    _TileDataset,
)
from augur.datasets.utils import (
    SlideRecord,
    TileRecord,
    derive_bcss_slide_name,
    get_slide_mpp,
    resolve_tissue_label_metadata_path,
)
from augur.models.tile_level.tile_model import TileModel
from augur.utils.config import load_yaml_config

SUPPORTED_EMBEDDING_SOURCES = ("encoder", "decoder", "pre-logits")


def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the evaluation run."""
    log_dir = os.path.join("logs", "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "tile_feature_space.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("tile_feature_space")
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
            "No checkpoint path was provided via --checkpoint-path or model config."
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


def _pool_encoder_features(model: TileModel, encoded: Any) -> Tensor:
    """Reduce encoder outputs to one ``[B, D]`` embedding vector per tile."""
    encoder_name = model.encoder.__class__.__name__
    if encoder_name == "ViTEncoder":
        return model._get_last_features_vit(encoded)  # pylint: disable=protected-access
    if encoder_name in {"ResNetEncoder", "UNetEncoder"}:
        if not (isinstance(encoded, (list, tuple)) and len(encoded) == 5):
            raise ValueError(
                f"Expected 5 feature maps from {encoder_name}, got {type(encoded)}."
            )
        c4 = encoded[-1]
        return torch.nn.functional.adaptive_avg_pool2d(c4, 1).flatten(1)
    raise ValueError(f"Unsupported encoder for embedding extraction: {encoder_name}")


def _run_decoder(
    model: TileModel,
    encoded: Any,
    *,
    capture_pre_logits: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Run the ``tissue_segmentation`` decoder, optionally capturing pre-logits.

    Returns the segmentation ``logits`` of shape ``[B, n_classes, H, W]``. When
    ``capture_pre_logits`` is set, also returns the spatial-mean-pooled
    ``[B, C]`` input to the decoder's final classification layer (captured
    via a forward pre-hook on ``decoder.head`` -- ``head[-1]`` for ``DPTDecoder``
    and ``head`` itself for ``UNetDecoder``).
    """
    if "tissue_segmentation" not in model.decoders:
        raise ValueError("tile_feature_space requires a 'tissue_segmentation' decoder.")
    decoder = model.decoders["tissue_segmentation"]
    captured: dict[str, Tensor] = {}
    handle = None
    if capture_pre_logits:
        head = getattr(decoder, "head", None)
        if head is None:
            raise AttributeError(
                f"Decoder {type(decoder).__name__} has no 'head' attribute."
            )
        final_layer = head[-1] if isinstance(head, torch.nn.Sequential) else head

        def _capture_input(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            del module
            if not args or not isinstance(args[0], Tensor):
                raise RuntimeError(
                    "Expected the decoder's final layer to receive a tensor input."
                )
            captured["pre_logits"] = args[0]

        handle = final_layer.register_forward_pre_hook(_capture_input)
    try:
        logits = decoder(encoded)
    finally:
        if handle is not None:
            handle.remove()

    if not isinstance(logits, Tensor) or logits.ndim != 4:
        raise ValueError(
            "tissue_segmentation decoder must return a (B, C, H, W) tensor; "
            f"got {type(logits).__name__}."
        )

    pre_logits_pooled: Tensor | None = None
    if capture_pre_logits:
        pre_logits = captured.get("pre_logits")
        if pre_logits is None or pre_logits.ndim != 4:
            raise RuntimeError(
                "Failed to capture (B, C, H, W) pre-logits via forward hook."
            )
        pre_logits_pooled = pre_logits.flatten(2).mean(dim=2)
    return logits, pre_logits_pooled


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

    splits = datamodule._get_slide_splits()  # pylint: disable=protected-access
    test_slides = splits["test"]
    _, _, roi_df = datamodule._load_bcss_metadata()  # pylint: disable=protected-access
    roi_groups: dict[str, pd.DataFrame] = {
        str(slide_name): group.reset_index(drop=True)
        for slide_name, group in roi_df.groupby("slide_name", sort=False)
    }

    rng = random.Random(seed)
    all_records: list[TileRecord] = []
    total = len(test_slides)
    for index, slide_record in enumerate(test_slides, start=1):
        slide_name = derive_bcss_slide_name(
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


# ----------------------------- metric computations -----------------------------


def _compute_class_separation(
    embeddings: np.ndarray,
    classes: np.ndarray,
    *,
    silhouette_sample_size: int,
    seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Silhouette (cosine) + Davies-Bouldin (euclidean) on tile embeddings.

    Both metrics require at least 2 distinct cluster labels; if that
    condition is not met they are reported as ``None``.
    """
    n_samples = int(embeddings.shape[0])
    unique = np.unique(classes)
    n_classes_present = int(unique.size)
    result: dict[str, Any] = {
        "n_samples": n_samples,
        "n_classes_present": n_classes_present,
        "silhouette_cosine": None,
        "silhouette_sample_size": None,
        "davies_bouldin": None,
    }
    if n_classes_present < 2 or n_samples < 3:
        logger.warning(
            "Not enough samples/classes for class-separation metrics "
            "(n_samples=%d, n_classes_present=%d).",
            n_samples,
            n_classes_present,
        )
        return result

    effective_sample = (
        min(silhouette_sample_size, n_samples) if silhouette_sample_size > 0 else None
    )
    try:
        sil = silhouette_score(
            embeddings,
            classes,
            metric="cosine",
            sample_size=effective_sample,
            random_state=seed,
        )
        result["silhouette_cosine"] = float(sil)
        result["silhouette_sample_size"] = (
            int(effective_sample) if effective_sample is not None else n_samples
        )
    except ValueError as exc:
        logger.warning("Silhouette computation failed: %s", exc)

    try:
        result["davies_bouldin"] = float(davies_bouldin_score(embeddings, classes))
    except ValueError as exc:
        logger.warning("Davies-Bouldin computation failed: %s", exc)
    return result


def _mean_intra_class_cosine_distance(features: np.ndarray) -> float:
    """Closed-form mean pairwise cosine distance over rows of ``features``.

    For L2-normalized vectors ``u_i`` (``i = 1..n``),
    ``sum_{i != j} u_i . u_j = ||sum_i u_i||^2 - n``.
    The mean pairwise cosine similarity is therefore
    ``(||sum||^2 - n) / (n * (n - 1))`` and the mean cosine distance is
    ``1 - <that>``. This avoids materializing an ``O(n^2)`` distance matrix.
    """
    n = features.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    u = features / safe
    sum_vec = u.sum(axis=0)
    sum_sq = float(np.dot(sum_vec, sum_vec))
    mean_cos_sim = (sum_sq - n) / (n * (n - 1))
    return float(1.0 - mean_cos_sim)


def _compute_within_class_compactness(
    embeddings: np.ndarray,
    classes: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Mean intra-class cosine distance per class, plus macro average."""
    per_class: dict[str, dict[str, Any]] = {}
    valid_means: list[float] = []
    for cls_idx in np.unique(classes):
        mask = classes == cls_idx
        count = int(mask.sum())
        if count < 2:
            per_class[class_names[int(cls_idx)]] = {
                "count": count,
                "mean_cosine_distance": None,
            }
            continue
        mean_d = _mean_intra_class_cosine_distance(embeddings[mask])
        per_class[class_names[int(cls_idx)]] = {
            "count": count,
            "mean_cosine_distance": mean_d,
        }
        if np.isfinite(mean_d):
            valid_means.append(mean_d)
    macro = float(np.mean(valid_means)) if valid_means else None
    return {
        "metric": "cosine",
        "per_class": per_class,
        "macro_avg_cosine_distance": macro,
    }


def _compute_agnostic_fraction(
    max_portions: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """Share of tiles whose dominant-class portion falls below ``threshold``."""
    n = int(max_portions.size)
    n_agnostic = int(np.sum(max_portions < threshold))
    return {
        "threshold": float(threshold),
        "n_tiles": n,
        "n_agnostic": n_agnostic,
        "fraction": float(n_agnostic / n) if n > 0 else None,
        "mean_dominant_portion": float(max_portions.mean()) if n > 0 else None,
    }


def _top_confusion_pairs(
    matrix: np.ndarray,
    class_names: list[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the ``top_k`` highest off-diagonal entries of ``matrix``.

    Each entry reports the absolute count, the GT-row rate, and the
    prediction-column rate so confusion can be read as either recall or
    precision style. Symmetric pairs ``(gt=A, pred=B)`` and ``(gt=B, pred=A)``
    are reported separately because the directional information matters.
    """
    n_classes = matrix.shape[0]
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    cm = matrix.copy()
    np.fill_diagonal(cm, 0)
    flat = cm.flatten()
    k = int(min(top_k, np.count_nonzero(flat)))
    if k <= 0:
        return []
    idx_flat = np.argpartition(flat, -k)[-k:]
    idx_flat = idx_flat[np.argsort(-flat[idx_flat])]
    pairs: list[dict[str, Any]] = []
    for idx in idx_flat:
        gt = int(idx // n_classes)
        pred = int(idx % n_classes)
        count = int(matrix[gt, pred])
        gt_total = int(row_totals[gt])
        pred_total = int(col_totals[pred])
        pairs.append(
            {
                "gt": class_names[gt],
                "pred": class_names[pred],
                "count": count,
                "row_rate": float(count / gt_total) if gt_total > 0 else None,
                "col_rate": float(count / pred_total) if pred_total > 0 else None,
            }
        )
    return pairs


def _compute_failure_modes(
    *,
    pixel_confusion: np.ndarray,
    tile_confusion: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Pixel-level and tile-level confusion summaries (top-K off-diagonal pairs).

    The raw confusion matrices are embedded in the result so the JSON report
    is self-contained and can be re-plotted without re-running inference.
    """
    return {
        "pixel_level": {
            "n_pixels": int(pixel_confusion.sum()),
            "top_pairs": _top_confusion_pairs(
                pixel_confusion, class_names, top_k=top_k
            ),
            "matrix": pixel_confusion.astype(np.int64).tolist(),
        },
        "tile_level": {
            "n_tiles": int(tile_confusion.sum()),
            "top_pairs": _top_confusion_pairs(tile_confusion, class_names, top_k=top_k),
            "matrix": tile_confusion.astype(np.int64).tolist(),
        },
        "class_names": list(class_names),
    }


def _format_class_label(name: str) -> str:
    """Convert ``snake_case`` BCSS class names to a human-readable axis label."""
    return name.replace("_", " ")


def _derive_model_name(model_config_path: str) -> str:
    """Extract a short model name from the model-config filename.

    For configs that follow the repo convention ``model-<name>.yaml`` (e.g.
    ``configs/model-resnet50-magnification.yaml``) the ``model-`` prefix is
    stripped so the returned name is ``resnet50-magnification``. Other
    filenames are returned as their stem unchanged.
    """
    stem = os.path.splitext(os.path.basename(model_config_path))[0]
    return stem[len("model-"):] if stem.startswith("model-") else stem


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    *,
    title: str,
    output_path: str,
    normalize: str = "row",
    annotate_threshold: float = 0.01,
) -> None:
    """Render a confusion matrix as a labelled heatmap and save to ``output_path``.

    ``normalize='row'`` divides each row by its sum (recall-style: for each
    GT class, the share of its pixels/tiles predicted as each class).
    ``normalize='col'`` divides each column by its sum (precision-style).
    ``normalize='none'`` plots raw counts on a log scale. Cells whose
    normalized value is below ``annotate_threshold`` are left unannotated to
    avoid noise.
    """
    n_classes = matrix.shape[0]
    mat = matrix.astype(np.float64)
    if normalize == "row":
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.where(row_sums > 0, mat / row_sums, 0.0)
        vmin, vmax = 0.0, 1.0
        cbar_label = "Row-normalized rate (recall)"
    elif normalize == "col":
        col_sums = mat.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.where(col_sums > 0, mat / col_sums, 0.0)
        vmin, vmax = 0.0, 1.0
        cbar_label = "Column-normalized rate (precision)"
    else:
        display = mat
        vmin, vmax = None, None
        cbar_label = "Count"

    side = max(7.0, 0.42 * n_classes)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    im = ax.imshow(display, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")

    labels = [_format_class_label(name) for name in class_names]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title)
    ax.tick_params(axis="x", which="both", top=False, bottom=True)

    if vmax is not None:
        upper = vmax
    else:
        upper = float(display.max()) if display.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            v = float(display[i, j])
            if v >= annotate_threshold:
                color = "white" if upper > 0 and v / upper > 0.5 else "black"
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=6,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _log_summary(report: dict[str, Any], logger: logging.Logger) -> None:
    """Pretty-print the report's headline numbers to the log."""
    sep = report["class_separation"]
    comp = report["within_class_compactness"]
    agn = report["agnostic_fraction"]
    fm = report["failure_modes"]
    logger.info("=" * 60)
    logger.info("Class separation:")
    logger.info(
        "  silhouette (cosine) = %s  [n_samples=%d, n_classes_present=%d, "
        "sample_size=%s]",
        (
            f"{sep['silhouette_cosine']:.4f}"
            if sep["silhouette_cosine"] is not None
            else "n/a"
        ),
        sep["n_samples"],
        sep["n_classes_present"],
        sep["silhouette_sample_size"],
    )
    logger.info(
        "  Davies-Bouldin     = %s",
        f"{sep['davies_bouldin']:.4f}" if sep["davies_bouldin"] is not None else "n/a",
    )
    logger.info("Within-class compactness (cosine distance):")
    logger.info(
        "  macro avg          = %s",
        (
            f"{comp['macro_avg_cosine_distance']:.4f}"
            if comp["macro_avg_cosine_distance"] is not None
            else "n/a"
        ),
    )
    for name, entry in comp["per_class"].items():
        if entry["mean_cosine_distance"] is None:
            logger.info("    %-30s n=%d  (skipped)", name, entry["count"])
        else:
            logger.info(
                "    %-30s n=%d  d=%.4f",
                name,
                entry["count"],
                entry["mean_cosine_distance"],
            )
    logger.info("Agnostic tile fraction:")
    logger.info(
        "  threshold=%.2f  fraction=%s  (n_agnostic=%d / %d, mean_dominant=%s)",
        agn["threshold"],
        f"{agn['fraction']:.4f}" if agn["fraction"] is not None else "n/a",
        agn["n_agnostic"],
        agn["n_tiles"],
        (
            f"{agn['mean_dominant_portion']:.4f}"
            if agn["mean_dominant_portion"] is not None
            else "n/a"
        ),
    )
    logger.info(
        "Failure modes (top off-diagonal pairs, pixel-level): %d pixels",
        fm["pixel_level"]["n_pixels"],
    )
    for pair in fm["pixel_level"]["top_pairs"]:
        logger.info(
            "  gt=%-22s pred=%-22s count=%-12d row_rate=%s col_rate=%s",
            pair["gt"],
            pair["pred"],
            pair["count"],
            f"{pair['row_rate']:.4f}" if pair["row_rate"] is not None else "n/a",
            f"{pair['col_rate']:.4f}" if pair["col_rate"] is not None else "n/a",
        )
    logger.info(
        "Failure modes (top off-diagonal pairs, tile-level): %d tiles",
        fm["tile_level"]["n_tiles"],
    )
    for pair in fm["tile_level"]["top_pairs"]:
        logger.info(
            "  gt=%-22s pred=%-22s count=%-12d row_rate=%s col_rate=%s",
            pair["gt"],
            pair["pred"],
            pair["count"],
            f"{pair['row_rate']:.4f}" if pair["row_rate"] is not None else "n/a",
            f"{pair['col_rate']:.4f}" if pair["col_rate"] is not None else "n/a",
        )
    logger.info("=" * 60)


def evaluate(
    *,
    model_config_path: str,
    dataset_config_path: str,
    checkpoint_path: str | None,
    class_threshold: float,
    dominance_threshold: float,
    device_str: str,
    output_dir: str,
    output_basename: str,
    max_batches: int | None,
    portion_per_sample: float,
    stride: int | None,
    embedding_source: str,
    silhouette_sample_size: int,
    confusion_top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Run the full evaluation pipeline and return / persist the metric report."""
    logger = _setup_logger()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        device = torch.device(device_str)  # type: ignore
    logger.info("Using device: %s", device)

    if embedding_source not in SUPPORTED_EMBEDDING_SOURCES:
        raise ValueError(
            f"Unsupported embedding_source '{embedding_source}'. "
            f"Expected one of {SUPPORTED_EMBEDDING_SOURCES}."
        )
    logger.info("Embedding source: %s", embedding_source)

    model = _create_tile_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
    )
    if "tissue_segmentation" not in model.decoders:
        raise ValueError(
            "Model must have a 'tissue_segmentation' decoder. Found: "
            f"{sorted(model.decoders.keys())}"
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
        tissue_segmentation_n_classes=len(datamodule._bcss_gt_codes or {}),
        random_seed=datamodule.random_seed,
        logger=logger,
    )
    test_loader = datamodule.test_dataloader()
    logger.info(
        "Test dataset size: %d tiles (portion=%.3f, stride=%d)",
        len(datamodule.test_dataset),  # type: ignore
        portion_per_sample,
        effective_stride,
    )

    class_names = _load_class_names(datamodule.root_dir, logger)  # type: ignore
    n_classes = len(class_names)
    logger.info("Loaded %d tissue classes", n_classes)

    embeddings: list[np.ndarray] = []
    gt_dominant: list[np.ndarray] = []
    gt_max_portion: list[np.ndarray] = []
    pred_dominant: list[np.ndarray] = []
    pixel_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            task_batch = batch["tissue_segmentation"]
            image_batch = task_batch["image"].to(device, non_blocking=True)
            target_batch = task_batch["target"].to(device, non_blocking=True)

            encoded = model.encoder(image_batch)
            logits, pre_logits_pooled = _run_decoder(
                model, encoded, capture_pre_logits=(embedding_source == "pre-logits")
            )

            if embedding_source == "encoder":
                feats = _pool_encoder_features(model, encoded)
            elif embedding_source == "decoder":
                feats = torch.softmax(logits, dim=1).flatten(2).mean(dim=2)
            else:  # pre-logits
                assert pre_logits_pooled is not None
                feats = pre_logits_pooled

            n_pixels = target_batch.shape[2] * target_batch.shape[3]
            gt_portions = target_batch.float().sum(dim=(2, 3)) / n_pixels
            gt_max_p, gt_dom = gt_portions.max(dim=1)

            pred_probs = torch.softmax(logits, dim=1).flatten(2).mean(dim=2)
            _, pred_dom = pred_probs.max(dim=1)

            gt_pixels = target_batch.argmax(dim=1).reshape(-1)
            pred_pixels = logits.argmax(dim=1).reshape(-1)
            pixel_idx = (gt_pixels * n_classes + pred_pixels).to(torch.int64)
            batch_counts = torch.bincount(pixel_idx, minlength=n_classes * n_classes)
            pixel_confusion += batch_counts.cpu().numpy().reshape(n_classes, n_classes)

            embeddings.append(feats.cpu().numpy())
            gt_dominant.append(gt_dom.cpu().numpy())
            gt_max_portion.append(gt_max_p.cpu().numpy())
            pred_dominant.append(pred_dom.cpu().numpy())

            if batch_idx % 10 == 0:
                logger.info("Processed batch %d", batch_idx)

    embeddings_np = np.concatenate(embeddings, axis=0)
    gt_dominant_np = np.concatenate(gt_dominant, axis=0)
    gt_max_portion_np = np.concatenate(gt_max_portion, axis=0)
    pred_dominant_np = np.concatenate(pred_dominant, axis=0)
    n_tiles = int(embeddings_np.shape[0])
    logger.info(
        "Collected %d tiles (embedding dim=%d). Mean dominant portion: %.4f.",
        n_tiles,
        int(embeddings_np.shape[1]),
        float(gt_max_portion_np.mean()),
    )

    sure_mask = gt_max_portion_np >= class_threshold
    sure_emb = embeddings_np[sure_mask]
    sure_classes = gt_dominant_np[sure_mask]
    logger.info(
        "Sure tiles for clustering metrics: %d / %d (class_threshold=%.2f).",
        int(sure_mask.sum()),
        n_tiles,
        class_threshold,
    )

    class_separation = _compute_class_separation(
        sure_emb,
        sure_classes,
        silhouette_sample_size=silhouette_sample_size,
        seed=seed,
        logger=logger,
    )
    compactness = _compute_within_class_compactness(sure_emb, sure_classes, class_names)
    agnostic = _compute_agnostic_fraction(
        gt_max_portion_np, threshold=dominance_threshold
    )

    tile_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(tile_confusion, (gt_dominant_np, pred_dominant_np), 1)
    failure_modes = _compute_failure_modes(
        pixel_confusion=pixel_confusion,
        tile_confusion=tile_confusion,
        class_names=class_names,
        top_k=confusion_top_k,
    )

    class_counts = {
        class_names[int(c)]: int((sure_classes == c).sum())
        for c in np.unique(sure_classes)
    }

    report: dict[str, Any] = {
        "config": {
            "model_config": os.path.abspath(model_config_path),
            "dataset_config": os.path.abspath(dataset_config_path),
            "checkpoint_path": (
                os.path.abspath(checkpoint_path) if checkpoint_path else None
            ),
            "embedding_source": embedding_source,
            "embedding_dim": int(embeddings_np.shape[1]),
            "class_threshold": float(class_threshold),
            "dominance_threshold": float(dominance_threshold),
            "portion_per_sample": float(portion_per_sample),
            "stride": int(effective_stride),
            "seed": int(seed),
            "n_classes": int(n_classes),
        },
        "tiles": {
            "n_total": n_tiles,
            "n_sure": int(sure_mask.sum()),
            "mean_dominant_portion": float(gt_max_portion_np.mean()),
            "class_counts_sure": class_counts,
        },
        "class_separation": class_separation,
        "within_class_compactness": compactness,
        "agnostic_fraction": agnostic,
        "failure_modes": failure_modes,
    }

    os.makedirs(output_dir, exist_ok=True)
    model_name = _derive_model_name(model_config_path)
    suffix = f"{model_name}_{embedding_source}"
    output_path = os.path.join(
        output_dir,
        f"{output_basename}_{suffix}.json",
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", output_path)

    pixel_plot_path = os.path.join(
        output_dir, f"{output_basename}_{suffix}_confusion_pixel.png"
    )
    tile_plot_path = os.path.join(
        output_dir, f"{output_basename}_{suffix}_confusion_tile.png"
    )
    _plot_confusion_matrix(
        pixel_confusion,
        class_names,
        title="Pixel-level confusion (row-normalized)",
        output_path=pixel_plot_path,
    )
    _plot_confusion_matrix(
        tile_confusion,
        class_names,
        title="Tile-level confusion (row-normalized)",
        output_path=tile_plot_path,
    )
    logger.info("Saved pixel-level confusion plot to %s", pixel_plot_path)
    logger.info("Saved tile-level confusion plot to %s", tile_plot_path)

    _log_summary(report, logger)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the evaluation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional override for the checkpoint defined in the model config.",
    )
    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.25,
        help=(
            "Minimum dominant-class portion required to include a tile in the "
            "clustering metrics (silhouette, Davies-Bouldin, within-class "
            "compactness). Tiles below this are dropped from those metrics."
        ),
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.5,
        help=(
            "Dominance cut-off for the agnostic-tile-fraction metric. Tiles "
            "with max GT-class portion below this threshold are counted as "
            "agnostic. Defaults to 0.5 per the reference paper."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
    )
    parser.add_argument(
        "--output-basename",
        default="tile_feature_space",
        help=(
            "Base filename; outputs are written as "
            "<basename>_<model-name>_<embedding-source>.json (plus matching "
            "_confusion_pixel.png / _confusion_tile.png). <model-name> is "
            "derived from --model-config: 'model-resnet50-magnification.yaml' "
            "-> 'resnet50-magnification'."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of test batches to process.",
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
    parser.add_argument(
        "--embedding-source",
        choices=SUPPORTED_EMBEDDING_SOURCES,
        default="encoder",
        help=(
            "Per-tile embedding source. 'encoder': pool the encoder's deepest "
            "feature map. 'decoder': spatial-mean of the tissue_segmentation "
            "softmax predictions (class-aligned by construction). 'pre-logits': "
            "spatial-mean of the decoder's last hidden feature map."
        ),
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help=(
            "Sample size passed to sklearn.metrics.silhouette_score to keep "
            "the O(n^2) pairwise computation tractable. Set <= 0 to use all "
            "samples."
        ),
    )
    parser.add_argument(
        "--confusion-top-k",
        type=int,
        default=20,
        help="Number of top off-diagonal confusion pairs to report per level.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """CLI entrypoint: parse arguments and run :func:`evaluate`."""
    args = _build_arg_parser().parse_args()
    evaluate(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        checkpoint_path=args.checkpoint_path,
        class_threshold=args.class_threshold,
        dominance_threshold=args.dominance_threshold,
        device_str=args.device,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        max_batches=args.max_batches,
        portion_per_sample=args.portion_per_sample,
        stride=args.stride,
        embedding_source=args.embedding_source,
        silhouette_sample_size=args.silhouette_sample_size,
        confusion_top_k=args.confusion_top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
