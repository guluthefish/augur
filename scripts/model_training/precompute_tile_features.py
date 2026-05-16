"""One-shot precompute of frozen-encoder tile features for every slide.

Walks every labelled slide in the configured TCGA cohort, enumerates *all*
tissue tile centers (not the per-epoch sub-sample), encodes every tile through
the frozen tile encoder defined by ``--model-config``, and writes a per-slide
``<slide_id>.pt`` containing the ``(K, D)`` feature tensor plus tile-center
coordinates.

Once this cache exists, the slide-level aggregator can train on cached
features instead of re-encoding tiles every epoch. The aggregator backbones
(``DualCLAM`` and ``EmbeddingMIL``) already accept pre-encoded bags of shape
``(B, K, D)``; see ``_encode_bag`` in ``augur/models/slide_level/dual_clam.py``.

Typical invocation::

    python scripts/model_training/precompute_tile_features.py \\
        --config-dir configs \\
        --model-config model-resnet50-full.yaml \\
        --data-config slide_dataset-TCGA-BRCA.yaml \\
        --output-dir data/TCGA-BRCA/features/resnet50-full \\
        --device cuda --chunk-size 64

The output layout is::

    <output-dir>/
      <slide_id>.pt        # one file per slide
      _manifest.tsv        # slide_id, submitter_id, K, status, elapsed
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from openslide import OpenSlide
from torch import Tensor, nn

from augur.datasets.factory import get_dataset_from_config
from augur.datasets.tcga_slide_dataset import TCGASlideDataset
from augur.datasets.utils import (
    SlideRecord,
    _make_tile_record_for_mpp,
    read_tile_from_record,
)
from augur.models.tile_level.tile_model import TileModel
from augur.utils.config import load_yaml_config
from augur.utils.logger import setup_logger

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------


def _setup_logger_for_run(log_dir: str) -> logging.Logger:
    """Create the file+stdout logger used by this script."""
    return setup_logger(log_dir, name="precompute_tile_features")


# ----------------------------------------------------------------------------
# Encoder loading (mirrors DualCLAM.from_config tile-model loading)
# ----------------------------------------------------------------------------


def _resolve_relative_config_path(value: Any, *, config_dir: str) -> Any:
    """Resolve a config-relative path against ``config_dir``."""
    if not isinstance(value, str):
        return value
    if os.path.isabs(value):
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


def _load_tile_encoder(
    model_config_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
) -> tuple[nn.Module, str, int]:
    """Load the encoder from a tile-model YAML and return ``(encoder, name, D)``.

    The encoder is moved to ``device`` *and* cast to ``dtype``. Casting the
    weights matters because :func:`_encode_tile_bag` casts each input chunk
    to ``dtype`` before the forward pass — leaving the weights at fp32 while
    feeding fp16 input triggers a ``HalfTensor`` vs ``FloatTensor`` mismatch
    inside the first conv.

    The model YAML follows the same shape as the one referenced by aggregator
    configs via ``tile_model_config`` (e.g. ``model-resnet50-full.yaml``):

    .. code-block:: yaml

        params:
          encoder_config: encoder-resnet50.yaml
          decoders_config: {...}
        checkpoint_path: checkpoints/resnet50-full.pth

    The decoders are instantiated by ``TileModel.from_config`` but ignored
    here — only the encoder is retained.
    """
    tile_model_cfg = _resolve_tile_model_paths(
        load_yaml_config(model_config_path),
        config_path=model_config_path,
    )
    tile_model_params = tile_model_cfg.get("params", tile_model_cfg)
    tile_model = TileModel.from_config(tile_model_params)

    checkpoint_path = tile_model_cfg.get("checkpoint_path", None)
    if not isinstance(checkpoint_path, str):
        raise ValueError(
            f"tile-model config must include a 'checkpoint_path' string. "
            f"Got: {checkpoint_path!r}"
        )
    resolved_ckpt = os.path.abspath(checkpoint_path)
    if not os.path.isfile(resolved_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    checkpoint = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = tile_model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded tile-model checkpoint %s (missing=%d, unexpected=%d).",
        resolved_ckpt,
        len(missing),
        len(unexpected),
    )

    encoder = tile_model.encoder
    encoder_name = encoder.__class__.__name__
    enc_dim = _infer_encoder_dim(encoder, encoder_name)

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.to(device=device, dtype=dtype)

    logger.info(
        "Encoder ready: name=%s enc_dim=%d device=%s dtype=%s",
        encoder_name,
        enc_dim,
        device,
        dtype,
    )
    return encoder, encoder_name, enc_dim


def _infer_encoder_dim(encoder: nn.Module, encoder_name: str) -> int:
    """Read the encoder's per-tile feature dimension.

    Each encoder type stores this differently; the dispatch here mirrors
    :func:`_get_last_features` so adding a new encoder means updating both.
    """
    match encoder_name:
        case "ResNetEncoder" | "UNetEncoder":
            # Both expose ``feature_channels = (c0, c1, c2, c3, c4)``; we pool
            # c4 in _get_last_features, so its channel count IS the feature dim.
            feature_channels = getattr(encoder, "feature_channels", None)
            if not isinstance(feature_channels, Sequence) or len(feature_channels) == 0:
                raise RuntimeError(
                    f"{encoder_name} must expose a non-empty 'feature_channels' "
                    "sequence (c0...c4)."
                )
            return int(feature_channels[-1])
        case "ViTEncoder":
            # Set in ViTEncoder.__init__ from timm's vit_model.embed_dim.
            embed_dim = getattr(encoder, "embed_dim", None)
            if not isinstance(embed_dim, int) or embed_dim <= 0:
                raise RuntimeError(
                    "ViTEncoder must expose a positive integer 'embed_dim'."
                )
            return embed_dim
        case _:
            raise RuntimeError(
                f"No enc_dim inference strategy for encoder '{encoder_name}'. "
                "Add a branch here and in _get_last_features."
            )


# ----------------------------------------------------------------------------
# Feature extraction (mirrors TileModel/DualCLAM._get_last_features)
# ----------------------------------------------------------------------------


def _get_last_features(
    encoder: nn.Module,
    encoder_name: str,
    outputs: Tensor | Sequence[Tensor],
) -> Tensor:
    """Reduce raw encoder outputs to ``(N, D)`` per-tile features."""
    match encoder_name:
        case "ViTEncoder":
            if isinstance(outputs, Tensor):
                tokens = outputs
            elif (
                isinstance(outputs, Sequence)
                and len(outputs) > 0
                and isinstance(outputs[-1], Tensor)
            ):
                tokens = outputs[-1]
            else:
                raise TypeError(f"Unexpected ViT outputs type: {type(outputs)!r}")
            if tokens.ndim != 3:
                raise ValueError(
                    f"Expected ViT token output (N, seq, C). Got {tuple(tokens.shape)}"
                )
            num_prefix = getattr(encoder, "num_prefix_tokens", None)
            if num_prefix is None:
                raise RuntimeError(
                    "ViTEncoder must expose a num_prefix_tokens attribute."
                )
            if num_prefix >= 1:
                return tokens[:, 0, :]
            return tokens[:, num_prefix:, :].mean(dim=1)
        case "ResNetEncoder" | "UNetEncoder":
            if not (isinstance(outputs, Sequence) and len(outputs) == 5):
                raise ValueError(
                    f"Expected 5 feature maps from {encoder_name}; got {type(outputs)!r}."
                )
            c4 = outputs[-1]
            return c4.flatten(start_dim=2).mean(dim=-1)
        case _:
            if isinstance(outputs, Tensor) and outputs.ndim == 2:
                return outputs
            raise ValueError(
                f"No default feature-extraction strategy for {encoder_name}."
            )


@torch.no_grad()
def _encode_tile_bag(
    *,
    tiles: np.ndarray,
    encoder: nn.Module,
    encoder_name: str,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode a ``(K, H, W, 3)`` numpy tile bag into a ``(K, D)`` CPU tensor.

    Tiles are chunked through the encoder to bound GPU memory; per-chunk
    features are moved back to CPU float32 before concatenation so the slide's
    full bag never sits on GPU at once.
    """
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(f"Expected tiles of shape (K, H, W, 3). Got: {tiles.shape}")

    tiles_float = (
        tiles if tiles.dtype == np.float32 else tiles.astype(np.float32) / 255.0
    )
    # (K, H, W, 3) -> (K, 3, H, W)
    tiles_chw = np.ascontiguousarray(tiles_float.transpose(0, 3, 1, 2))
    flat = torch.from_numpy(tiles_chw)

    num_tiles = flat.shape[0]
    feature_chunks: list[Tensor] = []
    step = max(1, int(chunk_size))
    for start in range(0, num_tiles, step):
        chunk = flat[start : start + step].to(
            device=device, dtype=dtype, non_blocking=True
        )
        outputs = encoder(chunk)
        features = _get_last_features(encoder, encoder_name, outputs)
        feature_chunks.append(features.detach().to("cpu", dtype=torch.float32))
    return torch.cat(feature_chunks, dim=0)


# ----------------------------------------------------------------------------
# Per-slide processing
# ----------------------------------------------------------------------------


def _read_tile_images(
    *,
    slide: OpenSlide,
    slide_record: SlideRecord,
    centers: Sequence[tuple[int, int]],
    tile_size: int,
    image_size: int,
    base_mpp: float,
    logger: logging.Logger,
) -> np.ndarray:
    """Read every center as a tile and stack into ``(K, image_size, image_size, 3)``."""
    tile_images: list[np.ndarray] = []
    for center_x, center_y in centers:
        record = _make_tile_record_for_mpp(
            slide_record,
            slide,
            center_x=center_x,
            center_y=center_y,
            output_size=tile_size,
            target_mpp=base_mpp,
            logger=logger,
        )
        image = read_tile_from_record(slide, record)
        if image.shape[0] != image_size or image.shape[1] != image_size:
            image = cv2.resize(  # pylint: disable=no-member
                image,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            )
        tile_images.append(image)
    return np.stack(tile_images, axis=0)


def _output_path_for(slide_id: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"{slide_id}.pt")


def _process_slide(
    *,
    slide_record: SlideRecord,
    centers: Sequence[tuple[int, int]],
    encoder: nn.Module,
    encoder_name: str,
    enc_dim: int,
    output_dir: str,
    base_mpp: float,
    tile_size: int,
    image_size: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Encode one slide's full tile bag and persist features to ``<slide_id>.pt``."""
    output_path = _output_path_for(slide_record.slide_id, output_dir)
    start = time.monotonic()

    if not centers:
        logger.warning(
            "Slide %s has no tissue centers; skipping.", slide_record.slide_id
        )
        return {
            "slide_id": slide_record.slide_id,
            "submitter_id": slide_record.submitter_id,
            "K": 0,
            "status": "no_centers",
            "elapsed": 0.0,
        }

    slide = OpenSlide(slide_record.slide_path)
    try:
        tiles = _read_tile_images(
            slide=slide,
            slide_record=slide_record,
            centers=centers,
            tile_size=tile_size,
            image_size=image_size,
            base_mpp=base_mpp,
            logger=logger,
        )
    finally:
        slide.close()

    features = _encode_tile_bag(
        tiles=tiles,
        encoder=encoder,
        encoder_name=encoder_name,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
    )
    if features.shape[0] != tiles.shape[0]:
        raise RuntimeError(
            f"Encoder produced {features.shape[0]} features for {tiles.shape[0]} tiles."
        )
    if features.shape[1] != enc_dim:
        logger.warning(
            "Encoder produced D=%d but config declares enc_dim=%d for %s.",
            features.shape[1],
            enc_dim,
            encoder_name,
        )

    tile_centers = torch.tensor(list(centers), dtype=torch.long)
    payload = {
        "slide_id": slide_record.slide_id,
        "submitter_id": slide_record.submitter_id,
        "slide_path": slide_record.slide_path,
        "features": features,  # (K, D) float32 on CPU
        "tile_centers": tile_centers,  # (K, 2) long; level-0 (x, y)
        "encoder_name": encoder_name,
        "enc_dim": int(features.shape[1]),
        "base_mpp": float(base_mpp),
        "tile_size": int(tile_size),
        "image_size": int(image_size),
    }
    # Atomic write: tmp then rename so an interrupted run never leaves a
    # half-written file that --skip-existing would later trust.
    tmp_path = output_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)

    elapsed = time.monotonic() - start
    logger.info(
        "[done] slide_id=%s submitter=%s K=%d D=%d elapsed=%.2fs path=%s",
        slide_record.slide_id,
        slide_record.submitter_id,
        features.shape[0],
        features.shape[1],
        elapsed,
        output_path,
    )
    return {
        "slide_id": slide_record.slide_id,
        "submitter_id": slide_record.submitter_id,
        "K": int(features.shape[0]),
        "status": "ok",
        "elapsed": elapsed,
    }


# ----------------------------------------------------------------------------
# Manifest helpers
# ----------------------------------------------------------------------------


def _append_manifest_row(manifest_path: str, row: dict[str, Any]) -> None:
    """Append one TSV row to the run manifest (header written on first call)."""
    header_needed = not os.path.exists(manifest_path)
    keys = ["slide_id", "submitter_id", "K", "status", "elapsed"]
    with open(manifest_path, "a", encoding="utf-8") as handle:
        if header_needed:
            handle.write("\t".join(keys) + "\n")
        handle.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")


# ----------------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------------


def _instantiate_datamodule(
    data_config_path: str, logger: logging.Logger
) -> TCGASlideDataset:
    """Build the TCGASlideDataset datamodule from its YAML config."""
    config = load_yaml_config(data_config_path)
    datamodule = get_dataset_from_config(config)
    if not isinstance(datamodule, TCGASlideDataset):
        raise TypeError(
            "Precompute requires a TCGASlideDataset datamodule. "
            f"Got: {type(datamodule).__name__}"
        )
    logger.info("Loaded datamodule from %s", os.path.abspath(data_config_path))
    return datamodule


def _collect_all_slide_records(
    datamodule: TCGASlideDataset,
) -> tuple[list[SlideRecord], dict[str, list[tuple[int, int]]]]:
    """Trigger dataset setup and return every labelled slide + its tile centers.

    ``setup(stage="predict")`` enumerates every labelled slide (its
    ``predict`` split is the union of all labelled records) and precomputes
    their tile centers.
    """
    datamodule.setup(stage="predict")

    splits = datamodule._get_slide_splits()  # pylint: disable=protected-access
    centers_by_slide_id = (
        datamodule._centers_by_slide_id  # pylint: disable=protected-access
    )
    if centers_by_slide_id is None:
        raise RuntimeError("Datamodule.setup() did not populate tile centers.")

    seen: set[str] = set()
    records: list[SlideRecord] = []
    for record in splits["predict"]:
        if record.slide_id in seen:
            continue
        seen.add(record.slide_id)
        records.append(record)
    return records, centers_by_slide_id


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _resolve_config_path(config_dir: str, name: str) -> str:
    """Resolve a config name against ``--config-dir`` or accept an absolute path."""
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    candidate = os.path.join(config_dir, name)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Config not found: {candidate}")
    return os.path.abspath(candidate)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    match name:
        case "float32" | "fp32":
            return torch.float32
        case "float16" | "fp16" | "half":
            return torch.float16
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported --precision: {name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory holding the YAML configs.",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Tile-model YAML name (e.g. model-resnet50-full.yaml).",
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Slide-dataset YAML name (e.g. slide_dataset-TCGA-BRCA.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-slide .pt feature files will be written.",
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto | cuda | cuda:0 | cpu."
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Encoder forward dtype. fp16/bf16 cuts memory; safe for frozen encoders.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Tiles per encoder forward pass.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip slides whose <slide_id>.pt already exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-encode slides even if their <slide_id>.pt already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N slides (for smoke tests).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/precompute_tile_features",
        help="Directory for the run log file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = _setup_logger_for_run(args.log_dir)
    manifest_path = os.path.join(args.output_dir, "_manifest.tsv")

    model_config_path = _resolve_config_path(args.config_dir, args.model_config)
    data_config_path = _resolve_config_path(args.config_dir, args.data_config)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.precision)

    logger.info("model_config=%s", model_config_path)
    logger.info("data_config=%s", data_config_path)
    logger.info("output_dir=%s", os.path.abspath(args.output_dir))
    logger.info(
        "device=%s precision=%s chunk_size=%d",
        device,
        args.precision,
        args.chunk_size,
    )

    encoder, encoder_name, enc_dim = _load_tile_encoder(
        model_config_path, device=device, dtype=dtype, logger=logger
    )

    datamodule = _instantiate_datamodule(data_config_path, logger=logger)
    records, centers_by_slide_id = _collect_all_slide_records(datamodule)

    # Replay the slide-dataset's tile-extraction params so the cache matches
    # what the runtime dataset would have read.
    base_mpp = datamodule.base_mpp
    tile_size = datamodule.tile_size
    image_size = datamodule.image_size

    if args.limit is not None:
        records = records[: args.limit]
    total = len(records)
    logger.info(
        "Starting feature extraction over %d slides (limit=%s, skip_existing=%s).",
        total,
        args.limit,
        args.skip_existing,
    )

    ok = skipped = failed = no_centers = 0
    run_start = time.monotonic()
    for index, record in enumerate(records, start=1):
        out_path = _output_path_for(record.slide_id, args.output_dir)
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            logger.info(
                "[skip %d/%d] slide_id=%s already cached at %s",
                index,
                total,
                record.slide_id,
                out_path,
            )
            continue

        centers = centers_by_slide_id.get(record.slide_id, [])
        logger.info(
            "[start %d/%d] slide_id=%s submitter=%s K_candidates=%d path=%s",
            index,
            total,
            record.slide_id,
            record.submitter_id,
            len(centers),
            record.slide_path,
        )
        try:
            row = _process_slide(
                slide_record=record,
                centers=centers,
                encoder=encoder,
                encoder_name=encoder_name,
                enc_dim=enc_dim,
                output_dir=args.output_dir,
                base_mpp=base_mpp,
                tile_size=tile_size,
                image_size=image_size,
                chunk_size=args.chunk_size,
                device=device,
                dtype=dtype,
                logger=logger,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            logger.error(
                "[fail %d/%d] slide_id=%s error=%s\n%s",
                index,
                total,
                record.slide_id,
                exc,
                traceback.format_exc(),
            )
            row = {
                "slide_id": record.slide_id,
                "submitter_id": record.submitter_id,
                "K": 0,
                "status": "error",
                "elapsed": 0.0,
            }
        else:
            if row["status"] == "ok":
                ok += 1
            elif row["status"] == "no_centers":
                no_centers += 1

        _append_manifest_row(manifest_path, row)

    elapsed_total = time.monotonic() - run_start
    logger.info(
        "Done in %.1fs. ok=%d skipped=%d no_centers=%d failed=%d manifest=%s",
        elapsed_total,
        ok,
        skipped,
        no_centers,
        failed,
        manifest_path,
    )


if __name__ == "__main__":
    main()
