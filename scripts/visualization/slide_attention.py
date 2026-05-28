"""Visualize CLAM attention maps on a whole-slide image.

Produces a four-panel figure mirroring the layout of CLAM-style attention
visualizations:

1. **Slide thumbnail + tumor outline.** The original slide rendered at its
   lowest-resolution pyramid level with the included tumor polygon traced
   in blue. Carries a physical scale bar.
2. **Slide thumbnail + attention heatmap.** The same thumbnail with the
   per-tile attention values painted over every sampled tile center,
   plus a solid black rectangle marking the requested ROI.
3. **ROI crop + attention heatmap.** A zoomed-in crop of the level-0 ROI
   with the same attention overlay restricted to the ROI region. Carries
   a physical scale bar.
4. **Extreme-attention tiles.** Top: a few high-attention tiles framed in
   red. Bottom: a few low-attention tiles framed in light blue.

The script is split into two layers so you can also design your own
figure:

- :func:`compute_slide_attention` runs the model and returns a
  :class:`SlideAttentionResult` with every artifact (tile centers,
  attention, tile records, polygons, slide metadata).
- :func:`render_thumbnail_with_attention`, :func:`render_roi_with_attention`,
  :func:`render_extreme_tiles`, :func:`draw_polygons`, and
  :func:`draw_scale_bar` are reusable building blocks that consume the
  result.
- :func:`build_default_figure` composes the canonical four-panel layout.

If pre-encoded features exist at ``<features_dir>/<slide_id>.pt`` (as
written by ``scripts/model_training/precompute_tile_features.py``), they
are loaded directly and the tile encoder is skipped. Otherwise the
encoder defined by ``--tile-model-config`` is built on the fly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from openslide import OpenSlide
from PIL import Image
from torch import Tensor, nn

from augur.datasets.cancer_subtyping import load_subtyping_labels
from augur.datasets.utils import (
    SlideRecord,
    TileRecord,
    _make_tile_record_for_mpp,
    _resolve_path_from_atlas,
    enumerate_slide_tile_centers,
    get_slide_mpp,
    read_tile_from_record,
    resolve_slide_main_label_path,
)
from augur.models.slide_level.dual_clam import DualCLAM
from augur.models.tile_level.tile_model import TileModel
from augur.utils.config import load_aggregator_config, load_yaml_config

ATTENTION_HEAD_PREDICTED = "predicted"
ATTENTION_HEAD_MEAN = "mean"

POLYGON_SCALE_LEVEL0 = 16.0  # atlas polygons are stored at 1/16 of level 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SlideAttentionResult:
    """All artifacts produced by one attention-visualization pass.

    Use :func:`compute_slide_attention` to construct one of these; then
    either call :func:`build_default_figure` or read individual fields to
    design a custom figure.
    """

    # Slide metadata
    slide_id: str
    submitter_id: str
    slide_path: str
    slide_dims_l0: tuple[int, int]  # (width, height) at level 0
    slide_mpp: float  # microns per pixel at level 0

    # Tile bag
    centers_l0: np.ndarray  # (K, 2) int64 — level-0 tile centers (x, y)
    tile_extent_l0: float  # tile side length in level-0 pixels
    tile_records: list[TileRecord]
    base_mpp: float
    tile_size: int
    image_size: int

    # Model outputs
    features: np.ndarray  # (K, D) float32 tile features
    attention_all: np.ndarray  # (num_heads, K) raw attention weights
    attention: np.ndarray  # (K,) raw attention for the selected branch
    attention_display: np.ndarray  # (K,) in [0, 1], percentile-clipped
    branch_idx: int  # -1 means 'mean'; otherwise an index into attention_all
    subtyping_logits: np.ndarray  # (num_classes,)
    subtyping_class_names: tuple[str, ...] = field(
        default_factory=tuple
    )  # class index -> human-readable label
    subtyping_groundtruth_class: int = -1  # -1 if the submitter has no GT label

    # Tumor annotations
    incl_polygons_l0: list[np.ndarray] = field(default_factory=list)
    excl_polygons_l0: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging + config plumbing
# ---------------------------------------------------------------------------


def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the visualization run."""
    log_dir = os.path.join("logs", "visualization")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "slide_attention.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("slide_attention")
    logger.handlers = [handler, logging.StreamHandler()]
    logger.setLevel(logging.INFO)
    return logger


def _resolve_relative_config_path(value: Any, *, config_dir: str) -> Any:
    """Resolve a config-relative path against ``config_dir``."""
    if not isinstance(value, str) or os.path.isabs(value):
        return value
    candidate = os.path.join(config_dir, value)
    return os.path.abspath(candidate) if os.path.exists(candidate) else value


def _resolve_tile_model_paths(
    tile_model_config: dict[str, Any], *, config_path: str
) -> dict[str, Any]:
    """Resolve nested ``encoder_config`` / ``decoders_config`` paths."""
    resolved = dict(tile_model_config)
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
            name: _resolve_relative_config_path(spec, config_dir=config_dir)
            for name, spec in decoders_config.items()
        }
    resolved["params"] = resolved_params
    return resolved


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Normalize Lightning / plain-state-dict checkpoints into a parameter map."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must deserialize into a dict-like object.")
    if isinstance(checkpoint.get("state_dict"), dict):
        return dict(checkpoint["state_dict"])
    if isinstance(checkpoint.get("model_state_dict"), dict):
        return dict(checkpoint["model_state_dict"])
    return dict(checkpoint)


def _load_tile_encoder(
    model_config_path: str,
    *,
    device: torch.device,  # type: ignore[name-defined]
    logger: logging.Logger,
) -> tuple[nn.Module, str]:
    """Load the tile encoder from a tile-model YAML; mirrors precompute_tile_features."""
    tile_model_cfg = _resolve_tile_model_paths(
        load_yaml_config(model_config_path),
        config_path=model_config_path,
    )
    params = tile_model_cfg.get("params", tile_model_cfg)
    if not isinstance(params, dict):
        raise TypeError("Tile-model config 'params' must be a dict.")
    tile_model = TileModel.from_config(params)

    checkpoint_path = tile_model_cfg.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        raise ValueError("Tile-model config must include a 'checkpoint_path' string.")
    resolved = os.path.abspath(checkpoint_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Tile-model checkpoint not found: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = tile_model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded tile-model weights from %s (missing=%d, unexpected=%d).",
        resolved,
        len(missing),
        len(unexpected),
    )

    encoder = tile_model.encoder
    encoder_name = encoder.__class__.__name__
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.to(device=device)
    return encoder, encoder_name


def _load_aggregator(
    aggregator_cfg: dict[str, Any],
    *,
    device: torch.device,  # type: ignore[name-defined]
    logger: logging.Logger,
) -> DualCLAM:
    """Build the DualCLAM aggregator and load its checkpoint."""
    params = aggregator_cfg.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Aggregator config 'params' must be a dict.")
    model = DualCLAM.from_config(params)

    checkpoint_path = aggregator_cfg.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        raise ValueError("Aggregator config must include a 'checkpoint_path' string.")
    resolved = os.path.abspath(checkpoint_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Aggregator checkpoint not found: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded aggregator weights from %s (missing=%d, unexpected=%d).",
        resolved,
        len(missing),
        len(unexpected),
    )
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Feature extraction (cache-aware)
# ---------------------------------------------------------------------------


def _load_cached_features(
    features_path: str,
    *,
    expected_tile_size: int,
    expected_image_size: int,
    expected_base_mpp: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(features, tile_centers_l0)`` from a precomputed ``.pt`` file.

    The file is written by ``precompute_tile_features.py`` and contains
    ``features (K, D)``, ``tile_centers (K, 2)``, plus the tile geometry
    that was used. Returns ``(features_K_D_float32, centers_K_2_int64)``.
    """
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    for key in ("features", "tile_centers"):
        if key not in payload:
            raise KeyError(f"Cached feature file {features_path} missing key '{key}'.")
    features = payload["features"]
    centers = payload["tile_centers"]
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError(
            f"Cached features must be a (K, D) tensor. Got: {type(features).__name__}."
        )
    if not isinstance(centers, Tensor) or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(
            "Cached tile_centers must be an (K, 2) tensor. "
            f"Got shape: {tuple(centers.shape) if isinstance(centers, Tensor) else 'n/a'}"
        )
    if features.shape[0] != centers.shape[0]:
        raise ValueError(
            f"Cached features K={features.shape[0]} does not match "
            f"tile_centers K={centers.shape[0]}."
        )

    cached_tile_size = int(payload.get("tile_size", expected_tile_size))
    cached_image_size = int(payload.get("image_size", expected_image_size))
    cached_base_mpp = float(payload.get("base_mpp", expected_base_mpp))
    if cached_tile_size != expected_tile_size or cached_base_mpp != expected_base_mpp:
        logger.warning(
            "Cached tile geometry (tile_size=%d, base_mpp=%.4f) differs from the "
            "dataset config (tile_size=%d, base_mpp=%.4f). Using cached geometry; "
            "ensure this matches the aggregator's training-time tiling.",
            cached_tile_size,
            cached_base_mpp,
            expected_tile_size,
            expected_base_mpp,
        )
    if cached_image_size != expected_image_size:
        logger.info(
            "Cached image_size=%d differs from config image_size=%d. "
            "The image_size only affects later tile re-reads for the example panel.",
            cached_image_size,
            expected_image_size,
        )

    logger.info(
        "Loaded cached features from %s (K=%d, D=%d).",
        features_path,
        int(features.shape[0]),
        int(features.shape[1]),
    )
    return (
        features.float().numpy(),
        centers.long().numpy(),
    )


def _extract_encoder_features(
    encoder: nn.Module,
    encoder_name: str,
    outputs: Tensor | Sequence[Tensor],
) -> Tensor:
    """Reduce raw encoder outputs to ``(N, D)`` per-tile features."""
    if encoder_name == "ViTEncoder":
        if isinstance(outputs, Tensor):
            tokens = outputs
        elif isinstance(outputs, Sequence) and len(outputs) > 0:
            tokens = outputs[-1]
        else:
            raise TypeError(f"Unexpected ViT outputs type: {type(outputs)!r}")
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise ValueError("Expected ViT token tensor with shape (N, seq, C).")
        num_prefix = getattr(encoder, "num_prefix_tokens", None)
        if num_prefix is None:
            raise RuntimeError("ViTEncoder must expose num_prefix_tokens.")
        if num_prefix >= 1:
            return tokens[:, 0, :]
        return tokens[:, num_prefix:, :].mean(dim=1)
    if encoder_name in {"ResNetEncoder", "UNetEncoder"}:
        if not (isinstance(outputs, Sequence) and len(outputs) == 5):
            raise ValueError(
                f"Expected 5 feature maps from {encoder_name}; got {type(outputs)!r}."
            )
        c4 = outputs[-1]
        return c4.flatten(start_dim=2).mean(dim=-1)
    if isinstance(outputs, Tensor) and outputs.ndim == 2:
        return outputs
    raise ValueError(f"No feature-extraction strategy for encoder {encoder_name}.")


@torch.no_grad()
def _encode_tiles_live(
    *,
    tile_records: Sequence[TileRecord],
    slide: OpenSlide,
    encoder: nn.Module,
    encoder_name: str,
    image_size: int,
    chunk_size: int,
    device: torch.device,  # type: ignore[name-defined]
    logger: logging.Logger,
) -> np.ndarray:
    """Read every tile and encode it; returns ``(K, D)`` CPU float32 array."""
    raw_tiles: list[np.ndarray] = []
    for record in tile_records:
        tile = read_tile_from_record(slide, record)
        if tile.shape[0] != image_size or tile.shape[1] != image_size:
            tile = cv2.resize(  # pylint: disable=no-member
                tile,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            )
        raw_tiles.append(tile)
    if not raw_tiles:
        raise RuntimeError("No tiles were read; cannot compute attention.")

    stack = np.stack(raw_tiles, axis=0).astype(np.float32) / 255.0  # (K, H, W, 3)
    stack = np.ascontiguousarray(np.transpose(stack, (0, 3, 1, 2)))  # (K, 3, H, W)
    tensor = torch.from_numpy(stack)

    features: list[Tensor] = []
    total = tensor.shape[0]
    for start in range(0, total, chunk_size):
        chunk = tensor[start : start + chunk_size].to(device, non_blocking=True)
        out = encoder(chunk)
        feats = _extract_encoder_features(encoder, encoder_name, out)
        features.append(feats.detach().cpu().float())
        if start == 0 or (start // chunk_size) % 25 == 0:
            logger.info(
                "Encoded tile chunk %d-%d / %d.",
                start,
                min(start + chunk_size, total),
                total,
            )
    return torch.cat(features, dim=0).numpy()


# ---------------------------------------------------------------------------
# Tumor polygon loading
# ---------------------------------------------------------------------------


def _flatten_polygons(item: Any) -> list[np.ndarray]:
    """Collect every ``(N, 2)`` integer polygon nested inside a possibly-list value."""
    if isinstance(item, np.ndarray):
        if item.ndim == 2 and item.shape[1] == 2:
            return [item.astype(np.int64)]
        return []
    if isinstance(item, (list, tuple)):
        polygons: list[np.ndarray] = []
        for sub in item:
            polygons.extend(_flatten_polygons(sub))
        return polygons
    return []


def _load_tumor_polygons(
    root_dir: str,
    slide_path: str,
    *,
    polygon_scale: float,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(incl_polys_l0, excl_polys_l0)`` for the slide."""
    atlas_path = os.path.join(root_dir, "atlases", "tumor_label_atlas.txt")
    annotations_path = _resolve_path_from_atlas(atlas_path, "annotations", logger)

    with open(annotations_path, "rb") as fp:
        data = pickle.load(fp)
    if not isinstance(data, dict) or "incl_vec" not in data:
        raise ValueError(
            f"Tumor annotations at {annotations_path} are missing 'incl_vec'."
        )

    basename = os.path.basename(slide_path)
    stem, _ = os.path.splitext(basename)
    key_tiff = f"{stem}.tiff"

    incl_raw = data.get("incl_vec", {}).get(key_tiff, [])
    excl_raw = data.get("excl_vec", {}).get(key_tiff, [])
    if not incl_raw and not excl_raw:
        logger.warning(
            "No tumor annotations found for %s in %s.",
            key_tiff,
            annotations_path,
        )

    def _to_l0(polys: Any) -> list[np.ndarray]:
        return [
            (poly.astype(np.float64) * float(polygon_scale)).astype(np.int64)
            for poly in _flatten_polygons(polys)
        ]

    return _to_l0(incl_raw), _to_l0(excl_raw)


# ---------------------------------------------------------------------------
# Color scaling
# ---------------------------------------------------------------------------


def normalize_attention_for_display(
    values: np.ndarray,
    *,
    clip_percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Percentile-clip and min-max scale a 1-D attention vector to ``[0, 1]``.

    Min-max alone collapses 95% of the distribution onto the bottom of
    the colormap (because softmax over ~20k tiles produces a long tail
    of large outliers). Clipping to the ``[lo, hi]`` percentile of the
    raw attention before normalizing spreads color over the body of the
    distribution and reveals contrast between low/mid attention tiles.
    """
    if values.size == 0:
        return values.astype(np.float32, copy=True)
    lo_p, hi_p = clip_percentile
    lo = float(np.percentile(values, lo_p))
    hi = float(np.percentile(values, hi_p))
    if hi - lo < 1e-12:
        lo = float(values.min())
        hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _select_attention_head(
    *,
    attention_weights: Tensor,
    subtyping_logits: Tensor,
    head: str,
    num_main_branches: int,
    unknown_class_index: int | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose one branch of attention. Returns ``(all_heads, vector, branch)``."""
    if attention_weights.ndim != 3:
        raise ValueError(
            "Expected attention_weights of shape (B, num_heads, K). "
            f"Got: {tuple(attention_weights.shape)}"
        )
    all_heads = attention_weights[0].detach().cpu().float().numpy()  # (num_heads, K)
    num_heads = all_heads.shape[0]

    if head == ATTENTION_HEAD_MEAN:
        vector = all_heads[:num_main_branches].mean(axis=0)
        logger.info(
            "Visualizing mean attention across %d main-task branches.",
            num_main_branches,
        )
        return all_heads, vector, -1

    if head == ATTENTION_HEAD_PREDICTED:
        logits = subtyping_logits[0].detach().cpu().float().numpy()
        if (
            unknown_class_index is not None
            and 0 <= unknown_class_index < logits.shape[0]
        ):
            logits = logits.copy()
            logits[unknown_class_index] = -np.inf
        branch = int(np.argmax(logits))
        logger.info(
            "Predicted subtyping branch=%d (logit=%.3f).", branch, float(logits[branch])
        )
        return all_heads, all_heads[branch], branch

    try:
        branch = int(head)
    except ValueError as exc:
        raise ValueError(
            f"--attention-head must be one of '{ATTENTION_HEAD_PREDICTED}', "
            f"'{ATTENTION_HEAD_MEAN}', or an integer in [0, {num_heads - 1}]. "
            f"Got: {head!r}"
        ) from exc
    if not 0 <= branch < num_heads:
        raise ValueError(
            f"--attention-head index {branch} out of range [0, {num_heads - 1}]."
        )
    return all_heads, all_heads[branch], branch


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def _derive_slide_record(slide_path: str) -> SlideRecord:
    """Infer ``SlideRecord`` (slide_id, submitter_id) from the manifest path layout.

    Mirrors ``load_slide_records``: the manifest writes slides at
    ``<root>/ordered_data/<submitter_id>/<subfolder>/<file_id>/<filename>``,
    so the slide_id is the file_id directory name and the submitter_id is
    two directories above it.
    """
    slide_dir = os.path.dirname(slide_path)  # .../submitter/subfolder/<file_id>
    slide_id = os.path.basename(slide_dir)
    submitter_id = os.path.basename(
        os.path.dirname(os.path.dirname(slide_dir))
    )  # .../submitter
    return SlideRecord(
        slide_id=slide_id,
        submitter_id=submitter_id,
        slide_path=slide_path,
    )


def compute_slide_attention(
    *,
    data_config_path: str,
    aggregator_cfg: dict[str, Any],
    slide_path: str,
    features_dir: str | None = None,
    tile_model_config_path: str | None = None,
    attention_head: str = ATTENTION_HEAD_PREDICTED,
    clip_percentile: tuple[float, float] = (2.0, 98.0),
    device_str: str = "auto",
    chunk_size: int = 64,
    logger: logging.Logger | None = None,
) -> SlideAttentionResult:
    """Run the aggregator on a single slide and return all visualization artifacts.

    Loads the slide-dataset YAML for tiling parameters (``tile_size``,
    ``image_size``, ``base_mpp``, ``stride``, ``min_tissue_fraction``,
    ``thumbnail_max_size``, ``white_threshold``, ``root_dir``), enumerates
    every tissue tile center (``portion_per_sample = 1.0`` is the only
    sensible choice for visualization), and:

    1. If ``<features_dir>/<slide_id>.pt`` exists, loads cached features
       and their level-0 ``tile_centers`` directly — no tile encoder
       needed.
    2. Otherwise builds the tile encoder from ``tile_model_config_path``
       and encodes every tile on the fly.

    The aggregator forward pass then yields ``(num_heads, K)`` attention.
    ``attention_head`` picks the branch to expose at ``result.attention``:

    - ``'predicted'``: argmax over subtyping logits (unknown class
      masked).
    - ``'mean'``: mean over the main-task branches.
    - An integer: a specific branch index.

    Parameters
    ----------
    data_config_path:
        Slide-dataset YAML.
    aggregator_cfg:
        Merged aggregator config dict (must include ``params`` and
        ``checkpoint_path``). Build it via
        :func:`augur.utils.config.load_aggregator_config`.
    slide_path:
        Path to the ``.svs`` slide.
    features_dir:
        Directory holding ``<slide_id>.pt`` feature caches. If ``None``,
        the cache is bypassed and ``tile_model_config_path`` is required.
    tile_model_config_path:
        Tile-model YAML providing the encoder when no feature cache is
        available.
    attention_head:
        Branch selector, see above.
    clip_percentile:
        ``(lo, hi)`` percentiles used to clip ``result.attention`` before
        scaling to ``[0, 1]`` for colormapping.
    device_str:
        ``auto``, ``cpu``, ``cuda``, or ``cuda:N``.
    chunk_size:
        Encoder batch size when features are not cached.
    """
    if logger is None:
        logger = _setup_logger()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[arg-type]
    else:
        device = torch.device(device_str)  # type: ignore[arg-type]
    logger.info("Using device: %s", device)

    if not os.path.isfile(slide_path):
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    data_cfg = load_yaml_config(data_config_path)
    data_params = data_cfg.get("params", {})
    if not isinstance(data_params, dict):
        raise TypeError("Slide-dataset config 'params' must be a dict.")
    root_dir = data_params.get("root_dir")
    if not isinstance(root_dir, str):
        raise ValueError("Slide-dataset config must include a string 'root_dir'.")
    tile_size = int(data_params.get("tile_size", 512))
    image_size = int(data_params.get("image_size", 224))
    base_mpp = float(data_params.get("base_mpp", 0.25))
    stride = int(data_params.get("stride", tile_size))
    min_tissue_fraction = float(data_params.get("min_tissue_fraction", 0.25))
    enum_thumbnail_max_size = int(data_params.get("thumbnail_max_size", 1024))
    white_threshold = float(data_params.get("white_threshold", 0.85))
    logger.info(
        "Slide-dataset params: tile_size=%d image_size=%d base_mpp=%.3f stride=%d "
        "min_tissue_fraction=%.2f thumbnail_max_size=%d white_threshold=%.2f portion_per_sample=1.0",
        tile_size,
        image_size,
        base_mpp,
        stride,
        min_tissue_fraction,
        enum_thumbnail_max_size,
        white_threshold,
    )

    aggregator = _load_aggregator(aggregator_cfg, device=device, logger=logger)
    num_main_branches = int(aggregator.backbone.num_main_branches)
    unknown_class_index = aggregator.task_kwargs.get(aggregator.main_task, {}).get(
        "unknown_class_index", 0
    )

    slide_record = _derive_slide_record(slide_path)

    # Load the subtyping class names + per-submitter ground-truth so the
    # predicted index becomes interpretable downstream and the ground-truth
    # class for this slide is available too. Mirrors
    # TCGASlideDataset._ensure_main_labels_loaded.
    subtyping_class_names: tuple[str, ...] = ()
    subtyping_groundtruth_class: int = -1
    if aggregator.main_task == "subtyping":
        try:
            labels_path = resolve_slide_main_label_path(
                root_dir, "subtyping", None, logger
            )
            submitter_labels, subtyping_class_names = load_subtyping_labels(
                labels_path, logger=logger
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Could not load subtyping class names from atlases: %s. "
                "Metadata will only carry the numeric class index.",
                exc,
            )
        else:
            gt = submitter_labels.get(slide_record.submitter_id)
            if gt is None:
                logger.warning(
                    "No subtyping ground-truth label for submitter %s; "
                    "metadata will report -1.",
                    slide_record.submitter_id,
                )
            else:
                subtyping_groundtruth_class = int(gt)
    cached_path: str | None = None
    if features_dir:
        candidate = os.path.join(features_dir, f"{slide_record.slide_id}.pt")
        if os.path.isfile(candidate):
            cached_path = candidate
        else:
            logger.warning(
                "Feature cache %s does not exist; falling back to live encoding.",
                candidate,
            )

    slide = OpenSlide(slide_path)
    try:
        width_l0, height_l0 = slide.dimensions
        slide_mpp = get_slide_mpp(slide, logger)
        logger.info(
            "Slide %s: dims=(%d, %d) mpp=%.4f levels=%d",
            os.path.basename(slide_path),
            width_l0,
            height_l0,
            slide_mpp,
            slide.level_count,
        )
        tile_extent_l0 = float(tile_size * (base_mpp / slide_mpp))

        if cached_path:
            features_np, centers_np = _load_cached_features(
                cached_path,
                expected_tile_size=tile_size,
                expected_image_size=image_size,
                expected_base_mpp=base_mpp,
                logger=logger,
            )
            centers_list = [(int(x), int(y)) for x, y in centers_np]
        else:
            if not tile_model_config_path:
                raise ValueError(
                    "No feature cache available and tile_model_config_path is "
                    "not set; cannot encode tiles."
                )
            centers_list = enumerate_slide_tile_centers(
                slide_record,
                output_size=tile_size,
                context_mpp=base_mpp,
                min_tissue_fraction=min_tissue_fraction,
                thumbnail_max_size=enum_thumbnail_max_size,
                white_threshold=white_threshold,
                stride=stride,
                slide=slide,
                logger=logger,
            )
            if not centers_list:
                raise RuntimeError(
                    f"No tissue tile centers were found for {slide_path}."
                )
            logger.info("Enumerated %d candidate centers.", len(centers_list))

            encoder, encoder_name = _load_tile_encoder(
                tile_model_config_path, device=device, logger=logger
            )
            tile_records_for_encode = [
                _make_tile_record_for_mpp(
                    slide_record,
                    slide,
                    center_x=cx,
                    center_y=cy,
                    output_size=tile_size,
                    target_mpp=base_mpp,
                    logger=logger,
                )
                for (cx, cy) in centers_list
            ]
            features_np = _encode_tiles_live(
                tile_records=tile_records_for_encode,
                slide=slide,
                encoder=encoder,
                encoder_name=encoder_name,
                image_size=image_size,
                chunk_size=chunk_size,
                device=device,
                logger=logger,
            )
            centers_np = np.asarray(centers_list, dtype=np.int64)

        # Build TileRecord list (so callers can re-read individual tiles).
        tile_records = [
            _make_tile_record_for_mpp(
                slide_record,
                slide,
                center_x=int(cx),
                center_y=int(cy),
                output_size=tile_size,
                target_mpp=base_mpp,
                logger=logger,
            )
            for (cx, cy) in centers_list
        ]

        # Run the aggregator on the feature bag.
        features_gpu = torch.from_numpy(features_np).to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = aggregator(features_gpu)
        attention_weights = outputs["_attention_weights"]  # (1, num_heads, K)
        subtyping_logits = outputs[aggregator.main_task]  # (1, num_classes)

        all_heads_np, attention_vec, branch_idx = _select_attention_head(
            attention_weights=attention_weights,
            subtyping_logits=subtyping_logits,
            head=attention_head,
            num_main_branches=num_main_branches,
            unknown_class_index=unknown_class_index,
            logger=logger,
        )
        attention_display = normalize_attention_for_display(
            attention_vec, clip_percentile=clip_percentile
        )

        incl_polygons_l0, excl_polygons_l0 = _load_tumor_polygons(
            root_dir=root_dir,
            slide_path=slide_path,
            polygon_scale=POLYGON_SCALE_LEVEL0,
            logger=logger,
        )

        return SlideAttentionResult(
            slide_id=slide_record.slide_id,
            submitter_id=slide_record.submitter_id,
            slide_path=slide_path,
            slide_dims_l0=(width_l0, height_l0),
            slide_mpp=slide_mpp,
            centers_l0=centers_np,
            tile_extent_l0=tile_extent_l0,
            tile_records=tile_records,
            base_mpp=base_mpp,
            tile_size=tile_size,
            image_size=image_size,
            features=features_np,
            attention_all=all_heads_np,
            attention=attention_vec.astype(np.float32),
            attention_display=attention_display,
            branch_idx=branch_idx,
            subtyping_logits=subtyping_logits[0].detach().cpu().float().numpy(),
            subtyping_class_names=subtyping_class_names,
            subtyping_groundtruth_class=subtyping_groundtruth_class,
            incl_polygons_l0=incl_polygons_l0,
            excl_polygons_l0=excl_polygons_l0,
        )
    finally:
        slide.close()


# ---------------------------------------------------------------------------
# Renderers (reusable building blocks)
# ---------------------------------------------------------------------------


def get_slide_thumbnail(slide: OpenSlide, max_size: int) -> np.ndarray:
    """Return an RGB uint8 thumbnail with the longest side at most ``max_size``."""
    width_l0, height_l0 = slide.dimensions
    scale = max_size / max(width_l0, height_l0)
    target = (
        max(int(round(width_l0 * scale)), 1),
        max(int(round(height_l0 * scale)), 1),
    )
    thumb = slide.get_thumbnail(target).convert("RGB")
    return np.asarray(thumb, dtype=np.uint8)


def read_roi_crop(
    slide: OpenSlide,
    *,
    x_l0: int,
    y_l0: int,
    w_l0: int,
    h_l0: int,
    max_dim: int,
) -> tuple[np.ndarray, float]:
    """Read an ROI from the slide. Returns ``(roi_rgb, scale_l0_to_roi)``."""
    target_downsample = max(w_l0 / max_dim, h_l0 / max_dim, 1.0)
    level = slide.get_best_level_for_downsample(target_downsample)
    actual_downsample = float(slide.level_downsamples[level])
    read_w = max(int(round(w_l0 / actual_downsample)), 1)
    read_h = max(int(round(h_l0 / actual_downsample)), 1)
    region = slide.read_region((x_l0, y_l0), level, (read_w, read_h)).convert("RGB")
    return np.asarray(region, dtype=np.uint8), 1.0 / actual_downsample


def paint_attention_overlay(
    *,
    base_rgb: np.ndarray,
    tile_centers_l0: np.ndarray,
    tile_extent_l0: float,
    attention_display: np.ndarray,
    origin_l0: tuple[int, int] = (0, 0),
    scale_l0_to_canvas: tuple[float, float] | float,
    alpha: float = 0.30,
    cmap: str = "coolwarm",
) -> np.ndarray:
    """Alpha-blend a per-tile attention heatmap onto ``base_rgb``.

    ``attention_display`` must already be normalized to ``[0, 1]`` (use
    :func:`normalize_attention_for_display`). Each tile contributes a
    colored rectangle covering its level-0 footprint, mapped to canvas
    pixels via ``scale_l0_to_canvas``.
    """
    if isinstance(scale_l0_to_canvas, (tuple, list)):
        sx, sy = scale_l0_to_canvas
    else:
        sx = sy = float(scale_l0_to_canvas)
    canvas_h, canvas_w = base_rgb.shape[:2]

    half_w = max(1, int(round(tile_extent_l0 * sx / 2.0)))
    half_h = max(1, int(round(tile_extent_l0 * sy / 2.0)))

    color_map = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    ox, oy = origin_l0
    for (cx_l0, cy_l0), value in zip(tile_centers_l0, attention_display):
        cx = int(round((int(cx_l0) - ox) * sx))
        cy = int(round((int(cy_l0) - oy) * sy))
        x0 = max(cx - half_w, 0)
        y0 = max(cy - half_h, 0)
        x1 = min(cx + half_w, canvas_w)
        y1 = min(cy + half_h, canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue
        color_map[y0:y1, x0:x1] = float(value)

    valid = color_map >= 0.0
    if not valid.any():
        return base_rgb.copy()
    cmap_fn = plt.get_cmap(cmap)
    colored = cmap_fn(np.clip(color_map, 0.0, 1.0))[:, :, :3]
    colored_uint8 = (colored * 255.0).astype(np.uint8)
    out = base_rgb.copy()
    out[valid] = (
        (1.0 - alpha) * base_rgb[valid].astype(np.float32)
        + alpha * colored_uint8[valid].astype(np.float32)
    ).astype(np.uint8)
    return out


def render_thumbnail_with_attention(
    *,
    slide: OpenSlide,
    result: SlideAttentionResult,
    thumbnail_max_size: int = 1024,
    alpha: float = 0.30,
    cmap: str = "coolwarm",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(thumb_rgb, thumb_with_attention, scale_x, scale_y)``.

    ``scale_x``/``scale_y`` are thumbnail pixels per level-0 pixel.
    """
    thumb_rgb = get_slide_thumbnail(slide, thumbnail_max_size)
    width_l0, height_l0 = result.slide_dims_l0
    scale_x = thumb_rgb.shape[1] / width_l0
    scale_y = thumb_rgb.shape[0] / height_l0
    overlay = paint_attention_overlay(
        base_rgb=thumb_rgb,
        tile_centers_l0=result.centers_l0,
        tile_extent_l0=result.tile_extent_l0,
        attention_display=result.attention_display,
        origin_l0=(0, 0),
        scale_l0_to_canvas=(scale_x, scale_y),
        alpha=alpha,
        cmap=cmap,
    )
    return thumb_rgb, overlay, scale_x, scale_y


def render_roi_with_attention(
    *,
    slide: OpenSlide,
    result: SlideAttentionResult,
    roi_xywh_l0: tuple[int, int, int, int],
    roi_max_dim: int = 1024,
    alpha: float = 0.30,
    cmap: str = "coolwarm",
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int, int]]:
    """Return ``(roi_rgb, roi_with_attention, scale_l0_to_roi, roi_xywh_clamped)``."""
    width_l0, height_l0 = result.slide_dims_l0
    x_l0, y_l0, w_l0, h_l0 = roi_xywh_l0
    x_l0_c = max(0, min(int(x_l0), max(width_l0 - 1, 0)))
    y_l0_c = max(0, min(int(y_l0), max(height_l0 - 1, 0)))
    w_l0_c = max(1, min(int(w_l0), width_l0 - x_l0_c))
    h_l0_c = max(1, min(int(h_l0), height_l0 - y_l0_c))
    roi_rgb, roi_scale = read_roi_crop(
        slide,
        x_l0=x_l0_c,
        y_l0=y_l0_c,
        w_l0=w_l0_c,
        h_l0=h_l0_c,
        max_dim=roi_max_dim,
    )
    overlay = paint_attention_overlay(
        base_rgb=roi_rgb,
        tile_centers_l0=result.centers_l0,
        tile_extent_l0=result.tile_extent_l0,
        attention_display=result.attention_display,
        origin_l0=(x_l0_c, y_l0_c),
        scale_l0_to_canvas=roi_scale,
        alpha=alpha,
        cmap=cmap,
    )
    return roi_rgb, overlay, roi_scale, (x_l0_c, y_l0_c, w_l0_c, h_l0_c)


def render_extreme_tiles(
    *,
    slide: OpenSlide,
    result: SlideAttentionResult,
    num_high: int = 3,
    num_low: int = 4,
    image_size: int = 256,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Return ``(high_tiles, low_tiles, high_indices, low_indices)``.

    ``high_indices`` / ``low_indices`` are indices into
    ``result.tile_records`` and ``result.centers_l0`` so the caller can
    re-locate the example tiles on the slide.
    """
    order = np.argsort(result.attention)
    low_idx = [int(i) for i in order[: max(num_low, 0)]]
    high_idx = (
        [int(i) for i in order[-max(num_high, 0) :][::-1]] if num_high > 0 else []
    )

    def _read(idx: int) -> np.ndarray:
        tile = read_tile_from_record(slide, result.tile_records[idx])
        if tile.shape[0] != image_size or tile.shape[1] != image_size:
            tile = cv2.resize(  # pylint: disable=no-member
                tile,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            )
        return tile

    return [_read(i) for i in high_idx], [_read(i) for i in low_idx], high_idx, low_idx


def draw_polygons(
    ax: plt.Axes,  # type: ignore
    polygons: Sequence[np.ndarray],
    *,
    scale: float,
    edgecolor: str,
    linewidth: float = 1.6,
    offset_l0: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Trace level-0 polygons onto ``ax`` after ``(p - offset_l0) * scale``.

    Use ``offset_l0=(0, 0)`` when drawing on a slide thumbnail; use
    ``offset_l0=(roi_x_l0, roi_y_l0)`` to draw the same level-0 polygons
    on a cropped ROI image.
    """
    ox, oy = offset_l0
    shift = np.array([ox, oy], dtype=np.float64)
    for poly in polygons:
        if poly.shape[0] < 3:
            continue
        scaled = (poly.astype(np.float64) - shift) * scale
        patch = patches.Polygon(
            scaled,
            closed=True,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(patch)


def _pick_scale_bar_length_mm(
    mm_per_canvas_px: float, *, target_canvas_px: float
) -> tuple[float, str]:
    """Pick a 'nice' scale-bar length close to ``target_canvas_px`` pixels.

    Returns ``(length_mm, label)`` where ``label`` is formatted in mm or
    µm depending on magnitude.
    """
    raw_mm = mm_per_canvas_px * float(target_canvas_px)
    if raw_mm <= 0.0:
        return 0.0, "0"
    nice = [
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ]
    best = min(nice, key=lambda value: abs(np.log(value) - np.log(raw_mm)))
    if best < 1.0:
        label = f"{int(round(best * 1000))} µm"
    else:
        label = f"{best:g} mm"
    return best, label


def draw_scale_bar(
    ax: plt.Axes,  # type: ignore
    *,
    canvas_w: int,
    canvas_h: int,
    mm_per_canvas_px: float,
    target_fraction: float = 0.18,
    color: str = "black",
    bg_color: str | None = "white",
    fontsize: float = 8.0,
    linewidth: float = 3.0,
    pad_fraction: float = 0.04,
) -> None:
    """Draw a horizontal scale bar in the lower-left corner of ``ax``.

    ``mm_per_canvas_px`` is the physical mm per canvas pixel — typically
    ``slide_mpp / 1000 / scale_canvas_per_l0``. ``linewidth`` and
    ``fontsize`` are in matplotlib display units; scale them up on
    larger canvases (e.g. by ``canvas_w / reference_canvas_w``) so the
    ruler looks equally thick across panels of different pixel sizes.
    """
    target_px = target_fraction * canvas_w
    bar_mm, label = _pick_scale_bar_length_mm(
        mm_per_canvas_px, target_canvas_px=target_px
    )
    if bar_mm <= 0.0:
        return
    bar_px = bar_mm / mm_per_canvas_px
    pad_x = canvas_w * pad_fraction
    pad_y = canvas_h * pad_fraction
    x0 = pad_x
    x1 = pad_x + bar_px
    y0 = canvas_h - pad_y

    if bg_color is not None:
        ax.add_patch(
            patches.Rectangle(
                (x0 - canvas_w * 0.01, y0 - canvas_h * 0.045),
                (x1 - x0) + canvas_w * 0.02,
                canvas_h * 0.06,
                facecolor=bg_color,
                edgecolor="none",
                alpha=0.75,
            )
        )
    ax.plot([x0, x1], [y0, y0], color=color, linewidth=linewidth, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2.0,
        y0 - canvas_h * 0.015,
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


# ---------------------------------------------------------------------------
# Default 4-panel figure
# ---------------------------------------------------------------------------


def build_default_figure(
    *,
    result: SlideAttentionResult,
    roi_xywh_l0: tuple[int, int, int, int],
    output_path: str,
    thumbnail_max_size: int = 1024,
    roi_max_dim: int = 1024,
    alpha: float = 0.30,
    cmap: str = "coolwarm",
    num_high: int = 3,
    num_low: int = 4,
    polygon_linewidth: float = 1.6,
    roi_box_linewidth: float = 1.5,
    title: str | None = None,
) -> None:
    """Render the canonical four-panel figure and save it as a PNG."""
    slide = OpenSlide(result.slide_path)
    try:
        thumb_rgb, thumb_with_attention, scale_x, _scale_y = (
            render_thumbnail_with_attention(
                slide=slide,
                result=result,
                thumbnail_max_size=thumbnail_max_size,
                alpha=alpha,
                cmap=cmap,
            )
        )
        (
            roi_rgb,  # pylint: disable=unused-variable # type: ignore
            roi_with_attention,
            roi_scale,
            roi_xywh_clamped,
        ) = render_roi_with_attention(
            slide=slide,
            result=result,
            roi_xywh_l0=roi_xywh_l0,
            roi_max_dim=roi_max_dim,
            alpha=alpha,
            cmap=cmap,
        )
        high_tiles, low_tiles, _, _ = render_extreme_tiles(
            slide=slide,
            result=result,
            num_high=num_high,
            num_low=num_low,
        )
    finally:
        slide.close()

    n_cols_right = max(max(num_high, num_low, 1), 3)
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(
        2,
        3 + n_cols_right,
        width_ratios=[2.0, 2.0, 2.0] + [0.7] * n_cols_right,
        height_ratios=[1, 1],
        wspace=0.04,
        hspace=0.06,
    )

    # --- Panel 1: slide thumbnail + tumor outline + scale bar
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(thumb_rgb)
    draw_polygons(
        ax1,
        result.incl_polygons_l0,
        scale=scale_x,
        edgecolor="#00cc00",
        linewidth=polygon_linewidth,
    )
    draw_polygons(
        ax1,
        result.excl_polygons_l0,
        scale=scale_x,
        edgecolor="#ff5555",
        linewidth=polygon_linewidth,
    )
    draw_scale_bar(
        ax1,
        canvas_w=thumb_rgb.shape[1],
        canvas_h=thumb_rgb.shape[0],
        mm_per_canvas_px=result.slide_mpp / 1000.0 / scale_x,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # --- Panel 2: thumbnail + attention + solid ROI box
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(thumb_with_attention)
    x_l0, y_l0, w_l0, h_l0 = roi_xywh_clamped
    rect = patches.Rectangle(
        (x_l0 * scale_x, y_l0 * scale_x),
        w_l0 * scale_x,
        h_l0 * scale_x,
        linewidth=roi_box_linewidth,
        edgecolor="black",
        facecolor="none",
    )
    ax2.add_patch(rect)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # --- Panel 3: ROI + attention + scale bar
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.imshow(roi_with_attention)
    draw_scale_bar(
        ax3,
        canvas_w=roi_with_attention.shape[1],
        canvas_h=roi_with_attention.shape[0],
        mm_per_canvas_px=result.slide_mpp / 1000.0 / roi_scale,
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # --- Panel 4: extreme tiles
    def _draw_tile_row(row_idx: int, tiles: list[np.ndarray], color: str) -> None:
        for col_idx in range(n_cols_right):
            ax = fig.add_subplot(gs[row_idx, 3 + col_idx])
            if col_idx < len(tiles):
                ax.imshow(tiles[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3.0)

    _draw_tile_row(0, high_tiles, color="#d62728")  # red — high attention
    _draw_tile_row(1, low_tiles, color="#7fb6e6")  # light blue — low attention

    if title:
        fig.suptitle(title, fontsize=11)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-asset export
# ---------------------------------------------------------------------------


def _save_rgb_png(array: np.ndarray, path: str) -> None:
    """Save an HxWx3 uint8 RGB array as a PNG at ``path``."""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_decorated_panel(
    rgb: np.ndarray,
    path: str,
    *,
    decorate: Any,
    dpi: int = 200,
) -> None:
    """Render ``rgb`` on a single matplotlib axis, call ``decorate(ax)``, save.

    The figure is sized so the PNG keeps roughly the source image's pixel
    density at ``dpi``. ``decorate`` may draw polygons, rectangles, scale
    bars, etc. on the axis.
    """
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # type: ignore
    ax.imshow(rgb)
    decorate(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def export_attention_assets(
    *,
    result: SlideAttentionResult,
    roi_xywh_l0: tuple[int, int, int, int],
    output_dir: str,
    thumbnail_max_size: int = 1024,
    roi_max_dim: int = 1024,
    alpha: float = 0.30,
    cmap: str = "coolwarm",
    num_high: int = 3,
    num_low: int = 4,
    slide_polygon_linewidth: float = 1.0,
    roi_polygon_linewidth: float = 5.0,
    roi_box_linewidth: float = 1.5,
    extreme_tile_size: int = 256,
) -> dict[str, str]:
    """Save every visualization component to ``output_dir`` as a separate file.

    Three views are rendered (``slide``, ``slide_roi``, ``roi``), each in
    eight versions covering the cartesian product of three on/off toggles:
    attention overlay, tumor outline, scale ruler. The view ``slide_roi``
    additionally draws the ROI rectangle on every variant — that is what
    distinguishes it from ``slide``.

    The output layout is::

        output_dir/
          slide/                    # whole slide (no ROI box)
            plain.png
            attention.png
            outline.png
            ruler.png
            attention_outline.png
            attention_ruler.png
            outline_ruler.png
            attention_outline_ruler.png
          slide_roi/                # whole slide + ROI box (always on)
            <same 8 filenames>
          roi/                      # ROI crop
            <same 8 filenames>
          tiles/
            high_attention_00.png ...
            low_attention_00.png ...
          attention.npy              # raw attention vector (K floats)
          attention_display.npy      # percentile-clipped, [0, 1]
          centers_l0.npy             # (K, 2) tile centers at level 0
          metadata.json

    Returns a mapping ``{asset_name: absolute_path}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    saved: dict[str, str] = {}

    slide = OpenSlide(result.slide_path)
    try:
        thumb_rgb, thumb_with_attention, scale_x, _scale_y = (
            render_thumbnail_with_attention(
                slide=slide,
                result=result,
                thumbnail_max_size=thumbnail_max_size,
                alpha=alpha,
                cmap=cmap,
            )
        )
        roi_rgb, roi_with_attention, roi_scale, roi_xywh_clamped = (
            render_roi_with_attention(
                slide=slide,
                result=result,
                roi_xywh_l0=roi_xywh_l0,
                roi_max_dim=roi_max_dim,
                alpha=alpha,
                cmap=cmap,
            )
        )
        high_tiles, low_tiles, high_idx, low_idx = render_extreme_tiles(
            slide=slide,
            result=result,
            num_high=num_high,
            num_low=num_low,
            image_size=extreme_tile_size,
        )
    finally:
        slide.close()

    x_l0_c, y_l0_c, w_l0_c, h_l0_c = roi_xywh_clamped

    # Each view supplies its plain + attention base RGBs, its level-0
    # offset, its scale (level-0 -> canvas), and whether the ROI box is
    # always drawn on it.
    view_specs = [
        {
            "name": "slide",
            "plain": thumb_rgb,
            "attn": thumb_with_attention,
            "offset_l0": (0.0, 0.0),
            "scale": scale_x,
            "always_roi_box": False,
            "outline_linewidth": slide_polygon_linewidth,
        },
        {
            "name": "slide_roi",
            "plain": thumb_rgb,
            "attn": thumb_with_attention,
            "offset_l0": (0.0, 0.0),
            "scale": scale_x,
            "always_roi_box": True,
            "outline_linewidth": slide_polygon_linewidth,
        },
        {
            "name": "roi",
            "plain": roi_rgb,
            "attn": roi_with_attention,
            "offset_l0": (float(x_l0_c), float(y_l0_c)),
            "scale": roi_scale,
            "always_roi_box": False,
            "outline_linewidth": roi_polygon_linewidth,
        },
    ]

    reference_canvas_w = float(thumb_rgb.shape[1])

    def _make_decorator(
        *,
        scale: float,
        offset_l0: tuple[float, float],
        canvas_w: int,
        canvas_h: int,
        outline: bool,
        ruler: bool,
        roi_box: bool,
        outline_linewidth: float,
    ) -> Any:
        visual_scale = canvas_w / reference_canvas_w

        def decorate(ax: plt.Axes) -> None:  # type: ignore
            if outline:
                draw_polygons(
                    ax,
                    result.incl_polygons_l0,
                    scale=scale,
                    offset_l0=offset_l0,
                    edgecolor="#00cc00",
                    linewidth=outline_linewidth,
                )
                draw_polygons(
                    ax,
                    result.excl_polygons_l0,
                    scale=scale,
                    offset_l0=offset_l0,
                    edgecolor="#ff5555",
                    linewidth=outline_linewidth,
                )
            if roi_box:
                ox, oy = offset_l0
                ax.add_patch(
                    patches.Rectangle(
                        ((x_l0_c - ox) * scale, (y_l0_c - oy) * scale),
                        w_l0_c * scale,
                        h_l0_c * scale,
                        linewidth=roi_box_linewidth * visual_scale,
                        edgecolor="black",
                        facecolor="none",
                    )
                )
            if ruler:
                draw_scale_bar(
                    ax,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    mm_per_canvas_px=result.slide_mpp / 1000.0 / scale,
                    fontsize=8.0 * visual_scale,
                    linewidth=3.0 * visual_scale,
                )

        return decorate

    for spec in view_specs:
        view_dir = os.path.join(output_dir, spec["name"])
        os.makedirs(view_dir, exist_ok=True)
        for attn in (False, True):
            base = spec["attn"] if attn else spec["plain"]
            canvas_h, canvas_w = base.shape[:2]
            for outline in (False, True):
                for ruler in (False, True):
                    parts: list[str] = []
                    if attn:
                        parts.append("attention")
                    if outline:
                        parts.append("outline")
                    if ruler:
                        parts.append("ruler")
                    fname = ("_".join(parts) if parts else "plain") + ".png"
                    path = os.path.join(view_dir, fname)

                    needs_decoration = outline or ruler or spec["always_roi_box"]
                    if needs_decoration:
                        _save_decorated_panel(
                            base,
                            path,
                            decorate=_make_decorator(
                                scale=spec["scale"],
                                offset_l0=spec["offset_l0"],
                                canvas_w=canvas_w,
                                canvas_h=canvas_h,
                                outline=outline,
                                ruler=ruler,
                                roi_box=spec["always_roi_box"],
                                outline_linewidth=spec["outline_linewidth"],
                            ),
                        )
                    else:
                        _save_rgb_png(base, path)
                    saved[f"{spec['name']}/{fname[:-4]}"] = os.path.abspath(path)

    # Extreme-attention tiles.
    for i, tile in enumerate(high_tiles):
        path = os.path.join(tiles_dir, f"high_attention_{i:02d}.png")
        _save_rgb_png(tile, path)
        saved[f"high_attention_{i:02d}"] = os.path.abspath(path)
    for i, tile in enumerate(low_tiles):
        path = os.path.join(tiles_dir, f"low_attention_{i:02d}.png")
        _save_rgb_png(tile, path)
        saved[f"low_attention_{i:02d}"] = os.path.abspath(path)

    # Numpy arrays.
    np.save(os.path.join(output_dir, "attention.npy"), result.attention)
    np.save(os.path.join(output_dir, "attention_display.npy"), result.attention_display)
    np.save(os.path.join(output_dir, "centers_l0.npy"), result.centers_l0)
    saved["attention"] = os.path.abspath(os.path.join(output_dir, "attention.npy"))
    saved["attention_display"] = os.path.abspath(
        os.path.join(output_dir, "attention_display.npy")
    )
    saved["centers_l0"] = os.path.abspath(os.path.join(output_dir, "centers_l0.npy"))

    # Metadata.
    metadata = {
        "slide_id": result.slide_id,
        "submitter_id": result.submitter_id,
        "slide_path": result.slide_path,
        "slide_dims_l0": list(result.slide_dims_l0),
        "slide_mpp": float(result.slide_mpp),
        "base_mpp": float(result.base_mpp),
        "tile_size": int(result.tile_size),
        "image_size": int(result.image_size),
        "K": int(result.centers_l0.shape[0]),
        "tile_extent_l0": float(result.tile_extent_l0),
        "branch_idx": int(result.branch_idx),
        "num_heads": int(result.attention_all.shape[0]),
        "num_main_branches": int(result.subtyping_logits.shape[0]),
        "subtyping_class_names": list(result.subtyping_class_names),
        "subtyping_logits": [float(v) for v in result.subtyping_logits.tolist()],
        "subtyping_logits_by_class": {
            name: float(logit)
            for name, logit in zip(
                result.subtyping_class_names, result.subtyping_logits.tolist()
            )
        },
        "subtyping_predicted_class": int(np.argmax(result.subtyping_logits)),
        "subtyping_predicted_class_name": (
            result.subtyping_class_names[int(np.argmax(result.subtyping_logits))]
            if result.subtyping_class_names
            and int(np.argmax(result.subtyping_logits))
            < len(result.subtyping_class_names)
            else None
        ),
        "subtyping_groundtruth_class": int(result.subtyping_groundtruth_class),
        "subtyping_groundtruth_class_name": (
            result.subtyping_class_names[result.subtyping_groundtruth_class]
            if result.subtyping_class_names
            and 0
            <= result.subtyping_groundtruth_class
            < len(result.subtyping_class_names)
            else None
        ),
        "roi_xywh_l0_input": list(roi_xywh_l0),
        "roi_xywh_l0_clamped": list(roi_xywh_clamped),
        "thumbnail_size": [int(thumb_rgb.shape[1]), int(thumb_rgb.shape[0])],
        "thumbnail_scale_l0_to_canvas": float(scale_x),
        "roi_size": [int(roi_rgb.shape[1]), int(roi_rgb.shape[0])],
        "roi_scale_l0_to_canvas": float(roi_scale),
        "high_attention_indices": [int(i) for i in high_idx],
        "low_attention_indices": [int(i) for i in low_idx],
        "alpha": float(alpha),
        "cmap": str(cmap),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    saved["metadata"] = os.path.abspath(meta_path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def visualize(
    *,
    data_config_path: str,
    aggregator_cfg: dict[str, Any],
    tile_model_config_path: str,
    slide_path: str,
    features_dir: str | None,
    roi_xywh: tuple[int, int, int, int],
    output_dir: str,
    figure_path: str | None,
    device_str: str,
    chunk_size: int,
    attention_head: str,
    num_high: int,
    num_low: int,
    thumbnail_max_size: int,
    roi_max_dim: int,
    cmap: str,
    alpha: float,
    clip_percentile: tuple[float, float],
    slide_polygon_linewidth: float,
    roi_polygon_linewidth: float,
    roi_box_linewidth: float,
) -> SlideAttentionResult:
    """CLI entrypoint helper: compute attention, export assets, optionally figure."""
    logger = _setup_logger()
    result = compute_slide_attention(
        data_config_path=data_config_path,
        aggregator_cfg=aggregator_cfg,
        slide_path=slide_path,
        features_dir=features_dir,
        tile_model_config_path=tile_model_config_path,
        attention_head=attention_head,
        clip_percentile=clip_percentile,
        device_str=device_str,
        chunk_size=chunk_size,
        logger=logger,
    )
    saved = export_attention_assets(
        result=result,
        roi_xywh_l0=roi_xywh,
        output_dir=output_dir,
        thumbnail_max_size=thumbnail_max_size,
        roi_max_dim=roi_max_dim,
        alpha=alpha,
        cmap=cmap,
        num_high=num_high,
        num_low=num_low,
        slide_polygon_linewidth=slide_polygon_linewidth,
        roi_polygon_linewidth=roi_polygon_linewidth,
        roi_box_linewidth=roi_box_linewidth,
    )
    logger.info("Exported %d assets to %s", len(saved), os.path.abspath(output_dir))
    for name, path in saved.items():
        logger.info("  %s -> %s", name, path)

    if figure_path:
        title = (
            f"{result.slide_id}  •  "
            f"branch={result.branch_idx if result.branch_idx >= 0 else 'mean'}  •  "
            f"K={result.centers_l0.shape[0]}  •  "
            f"ROI=({roi_xywh[0]},{roi_xywh[1]},{roi_xywh[2]},{roi_xywh[3]})"
        )
        build_default_figure(
            result=result,
            roi_xywh_l0=roi_xywh,
            output_path=figure_path,
            thumbnail_max_size=thumbnail_max_size,
            roi_max_dim=roi_max_dim,
            alpha=alpha,
            cmap=cmap,
            num_high=num_high,
            num_low=num_low,
            polygon_linewidth=slide_polygon_linewidth,
            roi_box_linewidth=roi_box_linewidth,
            title=title,
        )
        logger.info("Saved combined figure to %s", figure_path)
    return result


def _parse_roi(values: list[str]) -> tuple[int, int, int, int]:
    """Parse an ROI argument into ``(x, y, w, h)`` integer level-0 coords."""
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            f"--roi expects exactly 4 integers (x y w h). Got: {values}"
        )
    parsed = tuple(int(value) for value in values)
    if any(v < 0 for v in parsed[:2]) or any(v <= 0 for v in parsed[2:]):
        raise argparse.ArgumentTypeError(
            f"--roi must satisfy x>=0, y>=0, w>0, h>0. Got: {parsed}"
        )
    return parsed  # type: ignore[return-value]


def _parse_percentile(value: str) -> tuple[float, float]:
    """Parse a 'lo,hi' percentile pair (e.g. '2,98')."""
    parts = [chunk.strip() for chunk in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--clip-percentile must be 'lo,hi'. Got: {value!r}"
        )
    lo, hi = float(parts[0]), float(parts[1])
    if not 0.0 <= lo < hi <= 100.0:
        raise argparse.ArgumentTypeError(
            f"--clip-percentile must satisfy 0<=lo<hi<=100. Got: ({lo}, {hi})"
        )
    return lo, hi


def _default_features_dir(root_dir: str, tile_model_config_path: str) -> str:
    """Guess ``<root_dir>/features/<encoder_arch>-<tile_pretext>`` from config name."""
    stem = os.path.splitext(os.path.basename(tile_model_config_path))[0]
    if stem.startswith("model-"):
        stem = stem[len("model-") :]
    return os.path.join(root_dir, "features", stem)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the visualization script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        default="configs/slide_dataset-TCGA-BRCA-test.yaml",
        help="Slide-dataset YAML config (used for tile-extraction parameters and root_dir).",
    )
    parser.add_argument(
        "--aggregator-config-dir",
        default="configs/aggregator",
        help=(
            "Directory holding the partial aggregator YAMLs (base-/subtask-/"
            "variant-/add-on-/encoder-/pretext-/optimizer-/lr-scheduler-)."
        ),
    )
    parser.add_argument(
        "--base",
        default="clam",
        choices=["clam", "mil"],
        help="Bag-level aggregator architecture.",
    )
    parser.add_argument(
        "--subtask",
        default="sbs",
        choices=["sbs", "dbs", "id", "cnv"],
        help=(
            "Optional aggregator-level auxiliary subtask (COSMIC signature "
            "regression). Pass '' to omit. Only valid with --base clam."
        ),
    )
    parser.add_argument(
        "--variant",
        default="mb",
        choices=["sb", "mb", "mean", "max", "attention"],
        help="Variant within the base.",
    )
    parser.add_argument(
        "--add-on",
        default="gated",
        choices=["", "gated"],
        help="Optional attention add-on; pass '' to omit.",
    )
    parser.add_argument(
        "--encoder",
        default="resnet50",
        help="Encoder architecture token (e.g. 'resnet50', 'prov-gigapath').",
    )
    parser.add_argument(
        "--pretext",
        default="full",
        choices=["full", "hematoxylin", "jigmag", "magnification", "none"],
        help="Encoder pretext task; selects pretext-{name}.yaml.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw"],
        help="Optimizer recipe partial.",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=["cosine"],
        help="LR-scheduler recipe partial.",
    )
    parser.add_argument(
        "--tile-model-config",
        default="configs/model-resnet50-full.yaml",
        help=(
            "Tile-model YAML config providing the frozen encoder. Only used "
            "when no cached features exist for the slide."
        ),
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help=(
            "Directory containing precomputed <slide_id>.pt feature files "
            "from precompute_tile_features.py. Defaults to "
            "<root_dir>/features/<tile-model-stem>."
        ),
    )
    parser.add_argument(
        "--slide-path",
        default=(
            "data/TCGA-BRCA-test/ordered_data/TCGA-AR-A2LR/images/"
            "3d22d2bd-4b46-4579-8638-11cc269c738c/"
            "TCGA-AR-A2LR-01Z-00-DX1.C686A7D6-0361-49EE-B6CA-672B3086243C.svs"
        ),
        help="Absolute or repo-relative path to the .svs slide to visualize.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        default=["110000", "40000", "12947", "12947"],
        metavar=("X", "Y", "W", "H"),
        help="ROI to zoom into, in level-0 pixel coords: x y w h.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization/slide_attention",
        help="Directory to export per-asset PNGs and numpy arrays into.",
    )
    parser.add_argument(
        "--figure-path",
        default=None,
        help="Optional path for the combined 4-panel figure (off by default).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Tiles per encoder forward (only used when features aren't cached).",
    )
    parser.add_argument(
        "--attention-head",
        default=ATTENTION_HEAD_PREDICTED,
        help=(
            "Attention branch to render. 'predicted' (default) uses the branch "
            "for the predicted subtyping class. 'mean' averages over main-task "
            "branches. An integer selects a specific branch index."
        ),
    )
    parser.add_argument("--num-high", type=int, default=3)
    parser.add_argument("--num-low", type=int, default=4)
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=1024,
        help="Longest side of the slide thumbnail used in panels 1 and 2.",
    )
    parser.add_argument(
        "--roi-max-dim",
        type=int,
        default=1024,
        help="Longest side of the ROI crop used in panel 3.",
    )
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.30,
        help="Opacity of the attention overlay (0=invisible, 1=fully opaque).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=_parse_percentile,
        default=(2.0, 98.0),
        help=(
            "lo,hi percentiles for clipping the attention vector before "
            "scaling to [0, 1]. Lower lo / higher hi -> more vibrant midtones."
        ),
    )
    parser.add_argument(
        "--slide-polygon-linewidth",
        type=float,
        default=1.2,
        help="Tumor polygon outline thickness on the slide / slide_roi views.",
    )
    parser.add_argument(
        "--roi-polygon-linewidth",
        type=float,
        default=6.0,
        help="Tumor polygon outline thickness on the roi view (independent of slide).",
    )
    parser.add_argument(
        "--roi-box-linewidth",
        type=float,
        default=1.5,
        help="Line width for the ROI rectangle drawn on slide_roi.",
    )
    return parser


def main() -> None:
    """CLI entrypoint: parse arguments and run :func:`visualize`."""
    args = _build_arg_parser().parse_args()
    roi_xywh = _parse_roi(list(args.roi))

    # Infer features_dir from the data config + tile-model stem if not given.
    features_dir = args.features_dir
    if features_dir is None:
        data_cfg = load_yaml_config(args.data_config)
        data_params = data_cfg.get("params", {})
        root_dir = (
            data_params.get("root_dir") if isinstance(data_params, dict) else None
        )
        if isinstance(root_dir, str):
            features_dir = _default_features_dir(root_dir, args.tile_model_config)

    aggregator_cfg = load_aggregator_config(
        args.aggregator_config_dir,
        base=args.base,
        variant=args.variant,
        add_on=(args.add_on or None),
        subtask=(args.subtask or None),
        encoder=args.encoder,
        pretext=args.pretext,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
    )

    visualize(
        data_config_path=args.data_config,
        aggregator_cfg=aggregator_cfg,
        tile_model_config_path=args.tile_model_config,
        slide_path=args.slide_path,
        features_dir=features_dir,
        roi_xywh=roi_xywh,
        output_dir=args.output_dir,
        figure_path=args.figure_path,
        device_str=args.device,
        chunk_size=args.chunk_size,
        attention_head=args.attention_head,
        num_high=args.num_high,
        num_low=args.num_low,
        thumbnail_max_size=args.thumbnail_max_size,
        roi_max_dim=args.roi_max_dim,
        cmap=args.cmap,
        alpha=args.alpha,
        clip_percentile=args.clip_percentile,
        slide_polygon_linewidth=args.slide_polygon_linewidth,
        roi_polygon_linewidth=args.roi_polygon_linewidth,
        roi_box_linewidth=args.roi_box_linewidth,
    )


if __name__ == "__main__":
    main()
