"""2D scatter of tile-encoder embeddings on the held-out test split.

Loads a tile model trained on ``tissue_segmentation`` from a checkpoint, runs
its encoder over the test tiles, and projects the resulting per-tile
embeddings with PCA and/or UMAP. The held-out slide split is taken from the
``TCGATileDataset`` test split, but per-slide tile selection follows the
``T * portion_per_sample`` rule used by ``TCGASlideDataset``: candidate
centers within each slide's BCSS ROIs are enumerated on a regular grid, and
``floor(T * portion_per_sample)`` of them are sampled without replacement.
Each tile is assigned the tissue class that occupies the largest portion of
its ground-truth mask, but only when that portion meets
``--class-threshold``; otherwise the tile is labelled ``unsure``. A subset
of classes is highlighted in color while the rest (including ``unsure``)
are shown as "agnostic" gray points.
"""

from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA
from torch import Tensor

from VexDR.datasets.factory import get_dataset_from_config
from VexDR.datasets.tcga_tile_dataset import (  # noqa: F401  (private inner Dataset reused for test sampling)
    TCGATileDataset,
    _TileDataset,
)
from VexDR.datasets.utils import (
    SlideRecord,
    TileRecord,
    derive_bcss_slide_name,
    get_slide_mpp,
    resolve_tissue_label_metadata_path,
)
from VexDR.models.tile_level.tile_model import TileModel
from VexDR.utils.config import load_yaml_config

SUPPORTED_METHODS = ("pca", "umap")
SUPPORTED_EMBEDDING_SOURCES = ("encoder", "decoder", "pre-logits")
SUPPORTED_LABEL_SOURCES = ("gt-mask", "model-prediction")

_TITLE_LOWERCASE_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "as",
    }
)


def _format_class_name(name: str) -> str:
    """Convert a snake_case class name to title-case for display.

    The first word is always capitalized; subsequent words are kept
    lowercase when they appear in :data:`_TITLE_LOWERCASE_WORDS` (e.g.
    ``"necrosis_or_debris"`` becomes ``"Necrosis or Debris"``).
    """
    words = name.split("_")
    formatted: list[str] = []
    for i, word in enumerate(words):
        if not word:
            continue
        if i > 0 and word.lower() in _TITLE_LOWERCASE_WORDS:
            formatted.append(word.lower())
        else:
            formatted.append(word[:1].upper() + word[1:].lower())
    return " ".join(formatted)


def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the visualization run."""
    log_dir = os.path.join("logs", "visualization")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "tile_embeddings.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("tile_embeddings")
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


def _pool_decoder_pre_logits(model: TileModel, encoded: Any) -> Tensor:
    """Mean-pool the decoder's last hidden layer (input to its final classifier).

    Uses a forward pre-hook on the decoder's final ``head`` layer to capture
    its input tensor — that is the per-pixel feature map immediately before
    the final 1x1 convolution that produces logits. The captured map is
    spatial-mean-pooled to ``[B, C]``.

    Works for both ``DPTDecoder`` (where ``head`` is a ``Sequential``; the
    final layer is ``head[-1]``) and ``UNetDecoder`` (where ``head`` is a
    single ``Conv2d`` or ``Identity``). For the segmentation configs in
    this repo ``C`` is ``head_channels`` (DPT) or ``d0_channels`` (U-Net) —
    much smaller and more class-specialized than the encoder embedding,
    while preserving more nuance than the 22-D softmax probabilities.

    Parameters
    ----------
    model:
        Tile model that exposes a ``tissue_segmentation`` decoder in
        ``model.decoders``. The decoder must expose its final classification
        layer (or a ``Sequential`` ending in one) under the attribute
        ``head``.
    encoded:
        Output of ``model.encoder(image)``; forwarded directly to the
        decoder during the hooked forward pass.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, C]`` — one mean-pooled pre-logit vector per
        tile.
    """
    if "tissue_segmentation" not in model.decoders:
        raise ValueError(
            "pre-logits embedding source requires a 'tissue_segmentation' decoder."
        )
    decoder = model.decoders["tissue_segmentation"]
    head = getattr(decoder, "head", None)
    if head is None:
        raise AttributeError(
            f"Decoder {type(decoder).__name__} has no 'head' attribute; "
            "pre-logits extraction expects a final classification layer named 'head'."
        )
    final_layer = head[-1] if isinstance(head, torch.nn.Sequential) else head

    captured: dict[str, Tensor] = {}

    def _capture_input(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        del module
        if not args or not isinstance(args[0], Tensor):
            raise RuntimeError(
                "Expected the decoder's final layer to receive a tensor input."
            )
        captured["pre_logits"] = args[0]

    handle = final_layer.register_forward_pre_hook(_capture_input)
    try:
        decoder(encoded)
    finally:
        handle.remove()

    pre_logits = captured.get("pre_logits")
    if pre_logits is None:
        raise RuntimeError("Failed to capture decoder pre-logits via forward hook.")
    if pre_logits.ndim != 4:
        raise ValueError(
            f"Expected (B, C, H, W) pre-logits; got shape {tuple(pre_logits.shape)}."
        )
    return pre_logits.flatten(2).mean(dim=2)


def _pool_decoder_predictions(model: TileModel, encoded: Any) -> Tensor:
    """Mean-pool the tissue-segmentation decoder's softmax predictions per tile.

    Runs the ``tissue_segmentation`` decoder on the encoded features, applies
    softmax over the class axis, and averages over the spatial axes. The
    resulting ``[B, n_classes]`` vector is the expected class distribution
    over each tile and is class-aligned by construction, which usually
    produces much cleaner 2-D PCA/UMAP separation than encoder features.

    Parameters
    ----------
    model:
        Tile model that exposes a ``tissue_segmentation`` decoder in
        ``model.decoders``.
    encoded:
        Output of ``model.encoder(image)``; format depends on the encoder
        and is forwarded directly to the decoder.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, n_classes]`` — one mean-softmax probability
        vector per tile.
    """
    if "tissue_segmentation" not in model.decoders:
        raise ValueError(
            "Decoder embedding source requires a 'tissue_segmentation' decoder."
        )
    decoder = model.decoders["tissue_segmentation"]
    logits = decoder(encoded)
    if not isinstance(logits, Tensor) or logits.ndim != 4:
        raise ValueError(
            "tissue_segmentation decoder must return a (B, C, H, W) tensor; "
            f"got {type(logits).__name__} with shape "
            f"{tuple(logits.shape) if isinstance(logits, Tensor) else 'n/a'}."
        )
    probs = torch.softmax(logits, dim=1)
    return probs.flatten(2).mean(dim=2)


def _pool_encoder_features(model: TileModel, encoded: Any) -> Tensor:
    """Reduce encoder outputs to one ``[B, D]`` embedding vector per tile.

    For ``ViTEncoder`` the model's existing CLS / mean-patch pooling is reused
    via ``_get_last_features_vit``. For ``ResNetEncoder`` and ``UNetEncoder``
    the deepest feature map is global-average-pooled to a vector.
    """
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


def _assign_tile_classes(
    target: Tensor, threshold: float, *, unsure_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each tile the dominant tissue class (or ``unsure``).

    The dominant class is the argmax over per-class pixel portions in the
    one-hot ground-truth mask. If the dominant portion is below ``threshold``
    the tile is reassigned to ``unsure_index``.
    """
    n_pixels = target.shape[2] * target.shape[3]
    portions = target.float().sum(dim=(2, 3)) / n_pixels  # (B, n_classes)
    max_portion, dominant = portions.max(dim=1)
    classes = dominant.clone()
    classes[max_portion < threshold] = unsure_index
    return classes.cpu().numpy(), max_portion.cpu().numpy()


def _assign_classes_from_probs(
    probs: Tensor, threshold: float, *, unsure_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each tile the highest-probability class from per-tile predictions.

    Mirrors :func:`_assign_tile_classes` but consumes already-pooled
    ``[B, n_classes]`` probability vectors (e.g. from
    :func:`_pool_decoder_predictions`) instead of one-hot GT masks. Tiles
    whose top probability falls below ``threshold`` are reassigned to
    ``unsure_index``.

    Parameters
    ----------
    probs:
        Per-tile class probabilities of shape ``(B, n_classes)`` summing to
        1 along ``dim=1``.
    threshold:
        Minimum top probability required to keep the predicted class label.
    unsure_index:
        Class index used when no class meets ``threshold``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(classes, max_prob)`` of shape ``(B,)`` each. ``classes`` carries
        the predicted class index with ``unsure_index`` substituted; ``max_prob``
        is the raw top probability before substitution.
    """
    max_prob, dominant = probs.max(dim=1)
    classes = dominant.clone()
    classes[max_prob < threshold] = unsure_index
    return classes.cpu().numpy(), max_prob.cpu().numpy()


def _to_uint8_image(image: Tensor) -> np.ndarray:
    """Convert a CHW float tile in ``[0, 1]`` to an HWC uint8 array for plotting."""
    arr = image.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _project(
    embeddings: np.ndarray,
    method: str,
    seed: int,
    logger: logging.Logger,
    *,
    standardize: bool = True,
) -> np.ndarray:
    """Project tile embeddings to 2D with PCA or UMAP.

    When ``standardize`` is ``True`` the embeddings are first whitened with
    :class:`sklearn.preprocessing.StandardScaler` so that one
    high-variance noise direction (e.g. stain intensity) doesn't dominate
    the leading components.
    """
    if standardize:
        from sklearn.preprocessing import (
            StandardScaler,
        )  # pylint: disable=import-outside-toplevel

        embeddings = StandardScaler().fit_transform(embeddings)
        logger.info("Applied per-feature standardization before %s.", method.upper())
    if method == "pca":
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(embeddings)
        logger.info(
            "PCA explained variance ratio: %.3f, %.3f",
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )
        return coords
    if method == "umap":
        try:
            import umap  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "UMAP requested but 'umap-learn' is not installed. "
                "Install it with `pip install umap-learn`."
            ) from exc
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)  # type: ignore
    raise ValueError(
        f"Unsupported method '{method}'. Expected one of {SUPPORTED_METHODS}."
    )


def _enumerate_bcss_grid_records(
    *,
    slide_record: SlideRecord,
    slide_rois: pd.DataFrame,
    base_mpp: float,
    output_size: int,
    stride: int,
    logger: logging.Logger,
) -> list[TileRecord]:
    """Enumerate non-overlapping BCSS-ROI tile records on a regular grid.

    For each ROI of the slide, walks a grid of tile centers spaced by
    ``stride`` (pixels at ``base_mpp``) and emits one ``TileRecord`` per
    fully-contained position. Only positions whose tile fits inside the
    ROI are kept; tile-extent / level resolution mirrors the logic in
    ``VexDR.datasets.utils._make_tile_record_for_mpp``.

    Parameters
    ----------
    slide_record:
        Slide to enumerate centers from.
    slide_rois:
        BCSS ROI rows for this slide (columns ``slide_name``, ``xmin``,
        ``ymin``, ``xmax``, ``ymax`` in level-0 coordinates).
    base_mpp:
        Microns-per-pixel used to read tiles.
    output_size:
        Output tile size in pixels at ``base_mpp``.
    stride:
        Spacing between consecutive grid centers in pixels at ``base_mpp``.
    logger:
        Logger forwarded to slide-MPP resolution.

    Returns
    -------
    list[TileRecord]
        One record per grid position; may be empty if every ROI is smaller
        than the tile extent.
    """
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
    """Sample ``floor(T * portion_per_sample)`` BCSS tile records per test slide.

    For each held-out test slide, all candidate centers within the slide's BCSS
    ROIs are enumerated on a grid (``T``), and ``max(1, floor(T * portion_per_sample))``
    of them are sampled without replacement.

    Parameters
    ----------
    datamodule:
        A prepared :class:`TCGATileDataset` (``prepare_data`` already called)
        whose ``tasks`` include ``"tissue_segmentation"``.
    portion_per_sample:
        Fraction of candidates per slide to keep; must be in ``(0, 1]``.
    stride:
        Grid spacing in pixels at ``datamodule.base_mpp``. Use
        ``datamodule.tile_size`` for non-overlapping tiles.
    seed:
        Random seed used to sample the per-slide subsets.
    logger:
        Logger for warnings (e.g. slides with no candidate centers).

    Returns
    -------
    list[TileRecord]
        Concatenated test-set records ready to feed the inner ``_TileDataset``.
    """
    if not 0.0 < portion_per_sample <= 1.0:
        raise ValueError(
            f"portion_per_sample must be in (0, 1]. Got: {portion_per_sample}"
        )
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer. Got: {stride}")

    splits = datamodule._get_slide_splits()
    test_slides = splits["test"]
    _, _, roi_df = datamodule._load_bcss_metadata()
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


def _select_highlight_indices(
    *,
    highlight_classes: list[str] | None,
    exclude_classes: list[str],
    top_classes: int,
    classes_with_unsure: np.ndarray,
    full_class_names: list[str],
    unsure_index: int,
) -> list[int]:
    """Pick which class indices to color in the scatter.

    If ``highlight_classes`` is provided, those names are translated to
    indices directly. Otherwise the most-populated classes in
    ``classes_with_unsure`` are picked (up to ``top_classes``), skipping
    ``unsure_index`` and any names listed in ``exclude_classes``.
    """
    if highlight_classes:
        indices: list[int] = []
        for name in highlight_classes:
            if name not in full_class_names:
                raise ValueError(
                    f"Highlight class '{name}' is not in the BCSS class list. "
                    f"Available: {full_class_names}"
                )
            indices.append(full_class_names.index(name))
        return indices

    excluded_indices = {unsure_index}
    for name in exclude_classes:
        if name in full_class_names:
            excluded_indices.add(full_class_names.index(name))

    unique, counts = np.unique(classes_with_unsure, return_counts=True)
    order = np.argsort(-counts)
    chosen: list[int] = []
    for idx in order:
        cls = int(unique[idx])
        if cls in excluded_indices:
            continue
        chosen.append(cls)
        if len(chosen) >= top_classes:
            break
    return chosen


def _build_color_map(
    highlight_indices: list[int],
) -> dict[int, tuple[float, float, float, float]]:
    """Assign a stable RGBA color to each highlighted class.

    Colors are drawn from ``tab10`` followed by ``tab20`` to support up to
    ~30 highlighted classes before wrapping.
    """
    base = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)  # type: ignore
    return {idx: (*base[i % len(base)], 1.0) for i, idx in enumerate(highlight_indices)}


def _plot_method(
    *,
    method: str,
    coords: np.ndarray,
    classes_with_unsure: np.ndarray,
    images_uint8: np.ndarray,
    highlight_indices: list[int],
    full_class_names: list[str],
    colors: dict[int, tuple[float, float, float, float]],
    num_examples: int,
    output_path: str,
    rng: np.random.Generator,
) -> None:
    """Render and save a scatter + example-tile figure for one projection method.

    The left panel is a 2D scatter of ``coords`` colored by class (highlighted
    classes get distinct colors; everything else is gray "Agnostic"). The
    right panel shows ``num_examples`` example tiles per highlighted class
    plus one row of agnostic tiles, each row framed in the row's color.
    """
    agnostic_mask = ~np.isin(classes_with_unsure, highlight_indices)
    n_rows = len(highlight_indices) + 1
    n_cols = max(num_examples, 1)
    cell_size = 1.0
    fig = plt.figure(
        figsize=((n_rows + n_cols) * cell_size, max(4.0, n_rows * cell_size))
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[n_rows, n_cols], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        coords[agnostic_mask, 0],
        coords[agnostic_mask, 1],
        c="lightgray",
        alpha=0.5,
        s=8,
        linewidths=0,
        label="Agnostic",
    )
    for cls_idx in highlight_indices:
        mask = classes_with_unsure == cls_idx
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[cls_idx]],
            alpha=0.85,
            s=12,
            linewidths=0,
            label=_format_class_name(full_class_names[cls_idx]),
        )
    axis_labels = {"pca": ("PC1", "PC2"), "umap": ("UMAP1", "UMAP2")}
    xlabel, ylabel = axis_labels[method]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(1)
    ax.legend(loc="best", frameon=True, fontsize=8)

    sub_gs = gs[0, 1].subgridspec(n_rows, n_cols, hspace=0.06, wspace=0.04)
    rows: list[tuple[Any, Any]] = [
        (cls_idx, colors[cls_idx]) for cls_idx in highlight_indices
    ]
    rows.append(("agnostic", "gray"))

    for row_idx, (cls_id, color) in enumerate(rows):
        if cls_id == "agnostic":
            pool = np.where(agnostic_mask)[0]
        else:
            pool = np.where(classes_with_unsure == cls_id)[0]
        if pool.size == 0:
            continue
        sample_count = min(n_cols, pool.size)
        sample = rng.choice(pool, size=sample_count, replace=False)
        for col_idx in range(n_cols):
            tile_ax = fig.add_subplot(sub_gs[row_idx, col_idx])
            if col_idx < sample_count:
                tile_ax.imshow(images_uint8[sample[col_idx]])
            tile_ax.set_xticks([])
            tile_ax.set_yticks([])
            for spine in tile_ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3.0)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize(
    *,
    model_config_path: str,
    dataset_config_path: str,
    checkpoint_path: str | None,
    class_threshold: float,
    device_str: str,
    output_dir: str,
    output_basename: str,
    methods: list[str],
    max_batches: int | None,
    num_examples: int,
    top_classes: int,
    highlight_classes: list[str] | None,
    exclude_classes: list[str],
    portion_per_sample: float,
    stride: int | None,
    embedding_source: str,
    label_source: str,
    standardize: bool,
    seed: int,
) -> None:
    """Run the full embedding-extraction and visualization pipeline.

    Loads the tile model and the test split, extracts per-tile embeddings,
    assigns each tile a tissue class via the dominant-portion rule, and
    writes one figure per requested projection method to
    ``{output_dir}/{output_basename}_{method}.png``.

    Parameters
    ----------
    model_config_path:
        Path to the model YAML config (e.g. ``configs/model-resnet50-full.yaml``).
    dataset_config_path:
        Path to the dataset YAML config; must include ``tissue_segmentation``
        as a task to provide GT masks.
    checkpoint_path:
        Optional override for the checkpoint specified in the model config.
    class_threshold:
        Minimum portion of pixels a class must occupy in a tile's GT mask
        to be assigned as that tile's class; otherwise the tile is labelled
        ``unsure``.
    device_str:
        ``auto``, ``cpu``, ``cuda``, or ``cuda:N``.
    output_dir:
        Directory where output figures are written.
    output_basename:
        Filename stem; the projection method is appended as ``_<method>.png``.
    methods:
        Subset of ``SUPPORTED_METHODS`` to run.
    max_batches:
        Optional cap on the number of test batches to process.
    num_examples:
        Number of example tiles per highlighted-class row in the figure.
    top_classes:
        Number of most-populated classes to auto-highlight (ignored when
        ``highlight_classes`` is set).
    highlight_classes:
        Optional explicit list of class names to highlight.
    exclude_classes:
        Class names to skip during auto-highlighting (``unsure`` is always
        skipped).
    portion_per_sample:
        Fraction of BCSS-grid candidate centers per test slide to keep
    stride:
        Grid spacing in pixels at ``base_mpp``. ``None`` means use the
        dataset's ``tile_size`` (non-overlapping tiles).
    embedding_source:
        ``"encoder"`` to use globally-pooled encoder features,
        ``"decoder"`` to use the spatial-mean of the tissue-segmentation
        decoder's softmax predictions (class-aligned by construction), or
        ``"pre-logits"`` to use the spatial-mean of the decoder's last
        hidden feature map (the input to its final 1x1 conv).
    label_source:
        ``"gt-mask"`` to label tiles by their dominant BCSS GT class, or
        ``"model-prediction"`` to label tiles by the model's own predicted
        class (spatial-mean softmax → argmax, with the same
        ``class_threshold``). The latter mirrors the reference paper's
        patch-labeling rule.
    standardize:
        If ``True`` apply per-feature standardization before PCA / UMAP so
        a high-variance noise direction (e.g. stain intensity) does not
        dominate the leading components.
    seed:
        Random seed used by the projector and example-tile sampling.
    """
    logger = _setup_logger()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        device = torch.device(device_str)  # type: ignore
    logger.info("Using device: %s", device)

    invalid = [m for m in methods if m not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported methods {invalid}. Expected subset of {SUPPORTED_METHODS}."
        )
    if embedding_source not in SUPPORTED_EMBEDDING_SOURCES:
        raise ValueError(
            f"Unsupported embedding_source '{embedding_source}'. "
            f"Expected one of {SUPPORTED_EMBEDDING_SOURCES}."
        )
    if label_source not in SUPPORTED_LABEL_SOURCES:
        raise ValueError(
            f"Unsupported label_source '{label_source}'. "
            f"Expected one of {SUPPORTED_LABEL_SOURCES}."
        )
    logger.info(
        "Embedding source: %s. Label source: %s. Standardize: %s.",
        embedding_source,
        label_source,
        standardize,
    )

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
        len(datamodule.test_dataset),
        portion_per_sample,
        effective_stride,
    )  # type: ignore

    class_names = _load_class_names(datamodule.root_dir, logger)  # type: ignore
    n_classes = len(class_names)
    unsure_index = n_classes
    full_class_names = class_names + ["unsure"]
    logger.info("Loaded %d tissue classes", n_classes)

    embeddings: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    portions_max: list[np.ndarray] = []
    images: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            task_batch = batch["tissue_segmentation"]
            image_batch = task_batch["image"].to(device, non_blocking=True)
            target_batch = task_batch["target"].to(device, non_blocking=True)

            encoded = model.encoder(image_batch)

            decoder_probs: Tensor | None = None
            if embedding_source == "decoder":
                feats = _pool_decoder_predictions(model, encoded)
                decoder_probs = feats
            elif embedding_source == "pre-logits":
                feats = _pool_decoder_pre_logits(model, encoded)
            else:
                feats = _pool_encoder_features(model, encoded)

            if label_source == "model-prediction":
                if decoder_probs is None:
                    decoder_probs = _pool_decoder_predictions(model, encoded)
                tile_classes, tile_max_portion = _assign_classes_from_probs(
                    decoder_probs, class_threshold, unsure_index=unsure_index
                )
            else:
                tile_classes, tile_max_portion = _assign_tile_classes(
                    target_batch, class_threshold, unsure_index=unsure_index
                )

            embeddings.append(feats.cpu().numpy())
            classes.append(tile_classes)
            portions_max.append(tile_max_portion)
            for i in range(task_batch["image"].shape[0]):
                images.append(_to_uint8_image(task_batch["image"][i]))

            if batch_idx % 10 == 0:
                logger.info("Processed batch %d", batch_idx)

    embeddings_np = np.concatenate(embeddings, axis=0)
    classes_np = np.concatenate(classes, axis=0)
    portions_np = np.concatenate(portions_max, axis=0)
    images_np = np.stack(images, axis=0)
    logger.info(
        "Collected %d tiles. Mean dominant portion: %.3f. Unsure rate: %.2f%%",
        embeddings_np.shape[0],
        float(portions_np.mean()),
        100.0 * float(np.mean(classes_np == unsure_index)),
    )

    highlight_indices = _select_highlight_indices(
        highlight_classes=highlight_classes,
        exclude_classes=exclude_classes,
        top_classes=top_classes,
        classes_with_unsure=classes_np,
        full_class_names=full_class_names,
        unsure_index=unsure_index,
    )
    logger.info(
        "Highlighted classes: %s",
        [full_class_names[i] for i in highlight_indices],
    )

    colors = _build_color_map(highlight_indices)
    base_rng_seed = np.random.SeedSequence(seed)

    os.makedirs(output_dir, exist_ok=True)
    for method in methods:
        coords = _project(embeddings_np, method, seed, logger, standardize=standardize)
        output_path = os.path.join(
            output_dir,
            f"{output_basename}_{method}_{embedding_source}_{label_source}_top-{top_classes}.png",
        )
        _plot_method(
            method=method,
            coords=coords,
            classes_with_unsure=classes_np,
            images_uint8=images_np,
            highlight_indices=highlight_indices,
            full_class_names=full_class_names,
            colors=colors,
            num_examples=num_examples,
            output_path=output_path,
            rng=np.random.default_rng(base_rng_seed),
        )
        logger.info("Saved %s visualization to %s", method.upper(), output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the visualization script."""
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
        help="Minimum portion required for a tile to be assigned a tissue class.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization",
    )
    parser.add_argument(
        "--output-basename",
        default="tile_embeddings",
        help="Base filename; outputs are written as <basename>_<method>_top-<top_classes>.png",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(SUPPORTED_METHODS),
        help="Which projection methods to run.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of test batches to process.",
    )
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument(
        "--top-classes",
        type=int,
        default=5,
        help="Number of most-populated classes to highlight when --highlight-classes is not set.",
    )
    parser.add_argument(
        "--highlight-classes",
        nargs="*",
        default=None,
        help="Explicit class names to highlight; overrides --top-classes.",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=["outside_roi"],
        help="Class names that should never be auto-highlighted.",
    )
    parser.add_argument(
        "--portion-per-sample",
        type=float,
        default=1.0,
        help=(
            "Fraction of grid centers per held-out test slide to keep "
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
            "Per-tile embedding source. 'encoder': pool the encoder's "
            "deepest feature map (high-dim, generic). 'decoder': mean-pool "
            "the tissue_segmentation softmax predictions over the tile, "
            "yielding a (n_classes,) probability vector that is "
            "class-aligned by construction. 'pre-logits': mean-pool the "
            "decoder's last hidden feature map (the input to its final 1x1 "
            "conv) — class-specialized but with more capacity than the "
            "softmax probabilities."
        ),
    )
    parser.add_argument(
        "--label-source",
        choices=SUPPORTED_LABEL_SOURCES,
        default="gt-mask",
        help=(
            "How each tile's class label is derived. 'gt-mask' (default): "
            "argmax of pixel portions in the BCSS one-hot ground-truth mask. "
            "'model-prediction': argmax of the spatial-mean softmax produced "
            "by the tissue_segmentation decoder, mirroring the patch-labeling "
            "rule in the reference paper. The same '--class-threshold' applies "
            "to both (interpreted as min portion / min mean probability "
            "respectively)."
        ),
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply per-feature standardization (mean=0, std=1) before "
            "projecting. On by default to prevent a single high-variance "
            "noise direction (e.g. stain intensity) from dominating PC1. "
            "Use --no-standardize to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """CLI entrypoint: parse arguments and run :func:`visualize`."""
    args = _build_arg_parser().parse_args()
    visualize(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        checkpoint_path=args.checkpoint_path,
        class_threshold=args.class_threshold,
        device_str=args.device,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        methods=args.methods,
        max_batches=args.max_batches,
        num_examples=args.num_examples,
        top_classes=args.top_classes,
        highlight_classes=args.highlight_classes,
        exclude_classes=args.exclude_classes,
        portion_per_sample=args.portion_per_sample,
        stride=args.stride,
        embedding_source=args.embedding_source,
        label_source=args.label_source,
        standardize=args.standardize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
