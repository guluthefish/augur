"""Render DualCLAM-MB attention heatmaps for one task on a single slide.

This is the evaluation-side, task-focused counterpart to
``scripts/visualization/slide_attention.py``. It runs a *multi-branch*
DualCLAM (``DualCLAM_MB``) on one whole-slide image and writes a fixed set
of five figures into an output folder:

1. ``01_slide_roi_box.png`` - the whole slide rendered from its thumbnail
   with a solid black rectangle around the chosen ROI (no attention).
2. ``02_slide_roi_box_attention.png`` - the same view with the per-tile
   attention overlay painted on (red = high attention, blue = low).
3. ``03_roi.png`` - a zoom-in crop of the chosen ROI, no attention.
4. ``04_roi_attention.png`` - the ROI crop with the attention overlay.
5. ``05_top{k}_attention_tiles.png`` - the ``top-k`` highest-attention tiles
   (framed red) over the ``top-k`` lowest-attention tiles (framed blue).
6. ``06_roi_tissue_label.png`` *(optional, ``--tissue-label``)* - the ROI crop
   with the BCSS ground-truth tissue segmentation overlaid (one color per
   tissue class) plus a class legend. Only written when the chosen ROI overlaps
   a BCSS-annotated region for the slide.

The defining feature of this script is the ``--task`` selector, which maps a
task to one of the model's multi-branch attention heads
(:attr:`DualCLAM_MB.branch_layout`):

- ``subtyping`` (the main task): the attention map is taken from the branch
  of the *predicted* subtype class (``argmax`` over the subtyping logits,
  with the optional unknown class masked).
- ``<x>_regression`` (a configured COSMIC-signature subtask, e.g.
  ``cnv_regression``): the attention map is taken from that subtask's own
  branch.

The model is rebuilt from the same partial-merge config axes used at train
time (so the exported ``checkpoint_path`` weights load into a byte-compatible
architecture), exactly like ``slide_results.py``. This script is **only**
valid for the multi-branch variant; it errors out if the composed config is
not ``attn_kwargs.multi_branch == True``.

If pre-encoded features exist at ``<features_dir>/<slide_id>.pt`` (as written
by ``scripts/model_training/precompute_tile_features.py``) they are loaded
directly and the tile encoder is skipped; otherwise the encoder defined by
``--tile-model-config`` is built on the fly. All of the heavy lifting
(feature loading, the aggregator forward pass, attention selection,
percentile clipping, overlay painting, ROI cropping, tile re-reads) is reused
from ``scripts/visualization/slide_attention.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: cluster / eval runs have no display.

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from openslide import OpenSlide  # noqa: E402

# Put the repo root on the path so we can reuse the visualization script's
# rendering building blocks verbatim. Both scripts are launched as plain files
# (``python scripts/.../foo.py``) from the repo root, so the repo root is not
# otherwise importable; ``scripts`` is an implicit namespace package. This
# mirrors ``scripts/evaluation/slide_results.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augur.datasets.tissue_segmentation import (  # noqa: E402
    build_tissue_roi_groups,
    load_tissue_metadata,
)
from augur.datasets.utils import derive_tissue_slide_name  # noqa: E402
from augur.utils.config import (  # noqa: E402
    load_aggregator_config,
    load_dataset_config,
)
from augur.utils.logger import setup_logger  # noqa: E402
from scripts.visualization.slide_attention import (  # noqa: E402
    ATTENTION_HEAD_PREDICTED,
    SlideAttentionResult,
    _default_features_dir,
    _parse_percentile,
    _parse_roi,
    compute_slide_attention,
    render_extreme_tiles,
    render_roi_with_attention,
    render_thumbnail_with_attention,
)

# Short CLI subtask tokens -> the full task name used inside the model and the
# merged config (``params.subtasks`` entries, ``output_dims`` keys, branch
# layout). Mirrors ``augur.utils.config._AGGREGATOR_SUBTASK_TASK_NAMES`` so the
# user can pass either ``--task cnv`` or ``--task cnv_regression``.
_SUBTASK_TOKEN_TO_TASK: dict[str, str] = {
    "sbs": "sbs_regression",
    "dbs": "dbs_regression",
    "id": "id_regression",
    "cnv": "cnv_regression",
}

# coolwarm runs blue (0.0) -> red (1.0), i.e. low -> high attention, which is
# exactly the requested encoding (red = high, blue = low).
_DEFAULT_CMAP = "coolwarm"


# ---------------------------------------------------------------------------
# Task -> attention branch resolution (the multi-branch selector)
# ---------------------------------------------------------------------------


def _resolve_task(task: str, *, main_task: str, model_subtasks: list[str]) -> str:
    """Normalize a ``--task`` token to a task name present on the model.

    Accepts the main task name (``subtyping``), a full subtask name
    (``cnv_regression``), or a short subtask token (``cnv``). Raises
    ``ValueError`` if the resolved task is not one the model was built with.
    """
    candidate = _SUBTASK_TOKEN_TO_TASK.get(task, task)
    valid = [main_task, *model_subtasks]
    if candidate not in valid:
        raise ValueError(
            f"--task={task!r} (resolved to {candidate!r}) is not a task on this "
            f"model. The model declares main_task={main_task!r} and "
            f"subtasks={model_subtasks!r}. Pass one of: {valid}. For a "
            "regression subtask make sure you also selected it with --subtask "
            "(e.g. `--subtask cnv --task cnv_regression`)."
        )
    return candidate


def _resolve_attention_head(
    resolved_task: str,
    *,
    main_task: str,
    model_subtasks: list[str],
    num_main_branches: int,
) -> tuple[str, str]:
    """Map a resolved task to the ``attention_head`` selector + a description.

    Returns ``(attention_head, description)`` where ``attention_head`` is the
    string consumed by :func:`compute_slide_attention`:

    - main task -> ``"predicted"``: the branch of the predicted subtype class
      (``DualCLAM_MB.branch_layout[main_task] == (0, num_main_branches)``, so
      class ``c`` lives on branch ``c``).
    - subtask -> the integer branch index ``num_main_branches + i`` where ``i``
      is the subtask's position in the model's subtask list
      (``branch_layout[subtask] == (num_main_branches + i, ...)``).
    """
    if resolved_task == main_task:
        return ATTENTION_HEAD_PREDICTED, (
            f"main task {main_task!r}: attention from the predicted-class branch"
        )
    branch_idx = num_main_branches + model_subtasks.index(resolved_task)
    return str(branch_idx), (
        f"subtask {resolved_task!r}: attention from branch {branch_idx}"
    )


# ---------------------------------------------------------------------------
# BCSS tissue-segmentation label overlay (ground truth)
# ---------------------------------------------------------------------------

# Class 0 is ``outside_roi`` (no annotation) and is always rendered transparent.
_TISSUE_OUTSIDE_ROI = 0


def _tissue_palette() -> list[tuple[float, float, float]]:
    """A deterministic list of distinct RGB colors for tissue classes >= 1."""
    return list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)


def _load_roi_tissue_label(
    *,
    root_dir: str,
    slide_name: str,
    base_mpp: float,
    roi_xywh_l0: tuple[int, int, int, int],
    out_hw: tuple[int, int],
    slide_rois: Any,
    logger: Any,
) -> np.ndarray:
    """Composite the BCSS ground-truth tissue mask onto the chosen ROI box.

    Generalizes ``augur.datasets.utils.load_tissue_mask_label_for_free_tile``
    from a square tile to an arbitrary rectangular level-0 ROI: every annotated
    BCSS ROI that overlaps ``roi_xywh_l0`` is read, intersected, scaled, and
    written into an output canvas of shape ``out_hw`` (matching the rendered ROI
    crop). Pixels with no annotation stay at class ``0`` (``outside_roi``).
    """
    out_h, out_w = out_hw
    out_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    x0, y0, w_l0, h_l0 = roi_xywh_l0
    if slide_rois is None or slide_rois.empty or w_l0 <= 0 or h_l0 <= 0:
        return out_mask

    box_right = x0 + w_l0
    box_bottom = y0 + h_l0
    out_per_l0_x = out_w / w_l0
    out_per_l0_y = out_h / h_l0
    masks_dir = os.path.join(root_dir, "labels", "tissues", "masks")

    for roi in slide_rois.itertuples(index=False):
        rxmin, rymin = int(roi.xmin), int(roi.ymin)
        rxmax, rymax = int(roi.xmax), int(roi.ymax)
        ix_lo = max(float(x0), float(rxmin))
        iy_lo = max(float(y0), float(rymin))
        ix_hi = min(float(box_right), float(rxmax))
        iy_hi = min(float(box_bottom), float(rymax))
        if ix_hi <= ix_lo or iy_hi <= iy_lo:
            continue  # this annotated ROI does not overlap the chosen box

        mask_path = os.path.join(
            masks_dir,
            f"{slide_name}_xmin{rxmin}_ymin{rymin}_MPP-{float(base_mpp):.4f}.png",
        )
        if not os.path.exists(mask_path):
            if logger is not None:
                logger.warning("Tissue mask missing: %s", mask_path)
            continue
        mask_np = cv2.imread(  # pylint: disable=no-member
            mask_path, cv2.IMREAD_UNCHANGED  # pylint: disable=no-member
        )
        if mask_np is None:
            if logger is not None:
                logger.warning("Failed to read tissue mask: %s", mask_path)
            continue
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = mask_np.astype(np.uint8)  # class codes are small (<= 21)

        roi_w_l0 = max(rxmax - rxmin, 1)
        roi_h_l0 = max(rymax - rymin, 1)
        mask_sx = mask_np.shape[1] / roi_w_l0
        mask_sy = mask_np.shape[0] / roi_h_l0

        src_left = max(0, min(int(round((ix_lo - rxmin) * mask_sx)), mask_np.shape[1]))
        src_right = max(
            src_left, min(int(round((ix_hi - rxmin) * mask_sx)), mask_np.shape[1])
        )
        src_top = max(0, min(int(round((iy_lo - rymin) * mask_sy)), mask_np.shape[0]))
        src_bottom = max(
            src_top, min(int(round((iy_hi - rymin) * mask_sy)), mask_np.shape[0])
        )
        if src_right <= src_left or src_bottom <= src_top:
            continue

        dst_left = max(0, min(int(round((ix_lo - x0) * out_per_l0_x)), out_w))
        dst_right = max(dst_left, min(int(round((ix_hi - x0) * out_per_l0_x)), out_w))
        dst_top = max(0, min(int(round((iy_lo - y0) * out_per_l0_y)), out_h))
        dst_bottom = max(dst_top, min(int(round((iy_hi - y0) * out_per_l0_y)), out_h))
        if dst_right <= dst_left or dst_bottom <= dst_top:
            continue

        crop = mask_np[src_top:src_bottom, src_left:src_right]
        resized = cv2.resize(  # pylint: disable=no-member
            crop,
            (dst_right - dst_left, dst_bottom - dst_top),
            interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
        )
        out_mask[dst_top:dst_bottom, dst_left:dst_right] = resized

    return out_mask


def _tissue_overlay_rgba(
    label_mask: np.ndarray,
    *,
    alpha: float,
    palette: list[tuple[float, float, float]],
) -> tuple[np.ndarray, list[int]]:
    """Build an ``(H, W, 4)`` RGBA overlay; class 0 is transparent.

    Returns ``(rgba, present_classes)`` where ``present_classes`` is the sorted
    list of non-background class codes that appear in ``label_mask``.
    """
    height, width = label_mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    present = [int(c) for c in np.unique(label_mask) if int(c) != _TISSUE_OUTSIDE_ROI]
    for code in present:
        color = palette[(code - 1) % len(palette)]
        selector = label_mask == code
        rgba[selector, 0] = color[0]
        rgba[selector, 1] = color[1]
        rgba[selector, 2] = color[2]
        rgba[selector, 3] = alpha
    return rgba, present


# ---------------------------------------------------------------------------
# Figure rendering (the five requested outputs)
# ---------------------------------------------------------------------------


def _new_image_axes(rgb: np.ndarray, *, base_width_in: float = 9.0) -> tuple[Any, Any]:
    """Create a single bare axis sized to the aspect ratio of ``rgb``."""
    height, width = rgb.shape[:2]
    aspect = height / max(width, 1)
    fig, ax = plt.subplots(figsize=(base_width_in, base_width_in * aspect))
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _add_attention_colorbar(fig: Any, ax: Any) -> None:
    """Attach a slim 'attention low -> high' colorbar to ``ax``."""
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=_DEFAULT_CMAP)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("attention")


def _draw_roi_box(
    ax: Any,
    roi_xywh_l0: tuple[int, int, int, int],
    *,
    scale: float,
    linewidth: float,
) -> None:
    """Draw a solid black rectangle for the level-0 ROI scaled onto ``ax``."""
    x_l0, y_l0, w_l0, h_l0 = roi_xywh_l0
    ax.add_patch(
        Rectangle(
            (x_l0 * scale, y_l0 * scale),
            w_l0 * scale,
            h_l0 * scale,
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
        )
    )


def _save(fig: Any, path: str, *, title: str | None) -> None:
    """Title, tighten, save, and close a figure."""
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_heatmap_figures(
    *,
    result: SlideAttentionResult,
    roi_xywh_l0: tuple[int, int, int, int],
    output_dir: str,
    task_label: str,
    top_k: int,
    thumbnail_max_size: int,
    roi_max_dim: int,
    alpha: float,
    cmap: str,
    roi_box_linewidth: float,
    extreme_tile_size: int,
    logger: Any,
    root_dir: str | None = None,
    draw_tissue: bool = False,
    tissue_alpha: float = 0.45,
    tissue_slide_rois: Any = None,
    tissue_code_to_name: dict[int, str] | None = None,
) -> dict[str, str]:
    """Render and save the requested figures + a ``metadata.json``.

    Always writes the five core figures. When ``draw_tissue`` is set and the
    chosen ROI overlaps a BCSS-annotated region, additionally writes
    ``06_roi_tissue_label.png`` (the ROI crop with the ground-truth tissue
    segmentation overlaid + a class legend).

    Returns a mapping ``{asset_name: absolute_path}``.
    """
    tissue_code_to_name = tissue_code_to_name or {}
    os.makedirs(output_dir, exist_ok=True)
    saved: dict[str, str] = {}

    slide = OpenSlide(result.slide_path)
    try:
        thumb_rgb, thumb_attn, scale_x, _scale_y = render_thumbnail_with_attention(
            slide=slide,
            result=result,
            thumbnail_max_size=thumbnail_max_size,
            alpha=alpha,
            cmap=cmap,
        )
        roi_rgb, roi_attn, _roi_scale, roi_clamped = render_roi_with_attention(
            slide=slide,
            result=result,
            roi_xywh_l0=roi_xywh_l0,
            roi_max_dim=roi_max_dim,
            alpha=alpha,
            cmap=cmap,
        )
        high_tiles, low_tiles, high_idx, low_idx = render_extreme_tiles(
            slide=slide,
            result=result,
            num_high=top_k,
            num_low=top_k,
            image_size=extreme_tile_size,
        )
    finally:
        slide.close()

    pred_idx = int(np.argmax(result.subtyping_logits))
    pred_name = (
        result.subtyping_class_names[pred_idx]
        if result.subtyping_class_names and pred_idx < len(result.subtyping_class_names)
        else str(pred_idx)
    )
    header = f"{result.slide_id}  •  {task_label}  •  predicted subtype: {pred_name}"

    # --- Figure 1: slide overview + black ROI box (no attention).
    fig, ax = _new_image_axes(thumb_rgb)
    _draw_roi_box(ax, roi_clamped, scale=scale_x, linewidth=roi_box_linewidth)
    path = os.path.join(output_dir, "01_slide_roi_box.png")
    _save(fig, path, title=f"{header}\nwhole slide + ROI box")
    saved["01_slide_roi_box"] = os.path.abspath(path)

    # --- Figure 2: slide overview + black ROI box + attention overlay.
    fig, ax = _new_image_axes(thumb_attn)
    _draw_roi_box(ax, roi_clamped, scale=scale_x, linewidth=roi_box_linewidth)
    _add_attention_colorbar(fig, ax)
    path = os.path.join(output_dir, "02_slide_roi_box_attention.png")
    _save(fig, path, title=f"{header}\nwhole slide + ROI box + attention")
    saved["02_slide_roi_box_attention"] = os.path.abspath(path)

    # --- Figure 3: ROI zoom-in, no attention.
    fig, ax = _new_image_axes(roi_rgb)
    path = os.path.join(output_dir, "03_roi.png")
    _save(fig, path, title=f"{header}\nROI zoom-in")
    saved["03_roi"] = os.path.abspath(path)

    # --- Figure 4: ROI zoom-in + attention.
    fig, ax = _new_image_axes(roi_attn)
    _add_attention_colorbar(fig, ax)
    path = os.path.join(output_dir, "04_roi_attention.png")
    _save(fig, path, title=f"{header}\nROI zoom-in + attention")
    saved["04_roi_attention"] = os.path.abspath(path)

    # --- Figure 5: top-k highest (red) over top-k lowest (blue) attention.
    high_vals = [float(result.attention[i]) for i in high_idx]
    low_vals = [float(result.attention[i]) for i in low_idx]
    n_cols = max(top_k, 1)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.4 * n_cols, 5.4),
        squeeze=False,
    )
    for col in range(n_cols):
        # Top row: highest attention (red frame).
        ax_hi = axes[0][col]
        if col < len(high_tiles):
            ax_hi.imshow(high_tiles[col])
            ax_hi.set_title(f"#{col + 1}  a={high_vals[col]:.3g}", fontsize=8)
        ax_hi.set_xticks([])
        ax_hi.set_yticks([])
        for spine in ax_hi.spines.values():
            spine.set_color("#d62728")
            spine.set_linewidth(3.0)
        # Bottom row: lowest attention (blue frame).
        ax_lo = axes[1][col]
        if col < len(low_tiles):
            ax_lo.imshow(low_tiles[col])
            ax_lo.set_title(f"#{col + 1}  a={low_vals[col]:.3g}", fontsize=8)
        ax_lo.set_xticks([])
        ax_lo.set_yticks([])
        for spine in ax_lo.spines.values():
            spine.set_color("#7fb6e6")
            spine.set_linewidth(3.0)
    axes[0][0].set_ylabel("highest", fontsize=10, color="#d62728")
    axes[1][0].set_ylabel("lowest", fontsize=10, color="#3a7bd5")
    path = os.path.join(output_dir, f"05_top{top_k}_attention_tiles.png")
    _save(fig, path, title=f"{header}\ntop-{top_k} highest / lowest attention tiles")
    saved[f"05_top{top_k}_attention_tiles"] = os.path.abspath(path)

    # --- Figure 6 (optional): ROI zoom-in + BCSS tissue-label ground truth.
    tissue_present: list[tuple[int, str]] = []
    if (
        draw_tissue
        and root_dir
        and tissue_slide_rois is not None
        and not tissue_slide_rois.empty
    ):
        slide_name = derive_tissue_slide_name(result.slide_path, result.submitter_id)
        label_mask = _load_roi_tissue_label(
            root_dir=root_dir,
            slide_name=slide_name,
            base_mpp=result.base_mpp,
            roi_xywh_l0=roi_clamped,
            out_hw=roi_rgb.shape[:2],
            slide_rois=tissue_slide_rois,
            logger=logger,
        )
        palette = _tissue_palette()
        rgba, present = _tissue_overlay_rgba(
            label_mask, alpha=tissue_alpha, palette=palette
        )
        if present:
            tissue_present = [(c, tissue_code_to_name.get(c, str(c))) for c in present]
            fig, ax = _new_image_axes(roi_rgb)
            ax.imshow(rgba)
            legend_handles = [
                Patch(
                    facecolor=palette[(c - 1) % len(palette)],
                    edgecolor="none",
                    label=f"{c}: {tissue_code_to_name.get(c, str(c))}",
                )
                for c in present
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=7,
                framealpha=0.85,
                title="BCSS tissue",
                title_fontsize=8,
            )
            path = os.path.join(output_dir, "06_roi_tissue_label.png")
            _save(fig, path, title=f"{header}\nROI + tissue label (BCSS ground truth)")
            saved["06_roi_tissue_label"] = os.path.abspath(path)
        else:
            logger.warning(
                "Tissue overlay is empty for the chosen ROI (no BCSS annotation "
                "overlaps it); skipping 06_roi_tissue_label.png. Pick an ROI over "
                "an annotated region to see the tissue label."
            )

    # --- Sidecar metadata.
    metadata = {
        "slide_id": result.slide_id,
        "submitter_id": result.submitter_id,
        "slide_path": result.slide_path,
        "task_label": task_label,
        "branch_idx": int(result.branch_idx),
        "num_heads": int(result.attention_all.shape[0]),
        "num_main_branches": int(result.subtyping_logits.shape[0]),
        "K": int(result.centers_l0.shape[0]),
        "subtyping_predicted_class": pred_idx,
        "subtyping_predicted_class_name": pred_name,
        "subtyping_groundtruth_class": int(result.subtyping_groundtruth_class),
        "subtyping_class_names": list(result.subtyping_class_names),
        "roi_xywh_l0_input": list(roi_xywh_l0),
        "roi_xywh_l0_clamped": list(roi_clamped),
        "top_k": int(top_k),
        "high_attention_indices": [int(i) for i in high_idx],
        "low_attention_indices": [int(i) for i in low_idx],
        "high_attention_values": high_vals,
        "low_attention_values": low_vals,
        "alpha": float(alpha),
        "cmap": str(cmap),
        "tissue_label_drawn": bool(tissue_present),
        "tissue_label_classes": [
            {"code": code, "name": name} for code, name in tissue_present
        ],
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    saved["metadata"] = os.path.abspath(meta_path)

    for name, asset_path in saved.items():
        logger.info("  %s -> %s", name, asset_path)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the DualCLAM-MB attention-heatmap script."""
    parser = argparse.ArgumentParser(description=__doc__)

    # --- Model (DualCLAM-MB) selection: the partial-merge axes. ``base`` is
    # fixed to ``clam`` and the variant is fixed to ``mb`` because this script
    # is multi-branch-only.
    parser.add_argument(
        "--aggregator-config-dir",
        default="configs/aggregator",
        help="Directory holding the partial aggregator YAMLs.",
    )
    parser.add_argument(
        "--subtask",
        default=["cnv"],
        nargs="*",
        choices=["sbs", "dbs", "id", "cnv", "full"],
        metavar="TOKEN",
        help=(
            "Aggregator-level COSMIC-signature subtask head(s) the model was "
            "trained with. Must match the trained checkpoint. Pass `full` for "
            "all four, or `--subtask` with no tokens to omit (subtyping only)."
        ),
    )
    parser.add_argument(
        "--add-on",
        default="",
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

    # --- The task -> attention branch selector.
    parser.add_argument(
        "--task",
        default="subtyping",
        help=(
            "Which task's attention branch to visualize. 'subtyping' (the main "
            "task) uses the branch of the predicted subtype class. A regression "
            "subtask uses that subtask's branch; accepts either the full name "
            "('cnv_regression') or the short token ('cnv'). The chosen subtask "
            "must also be selected via --subtask."
        ),
    )

    # --- Dataset (slide-flavor tile-extraction params) selection.
    parser.add_argument(
        "--dataset-config-dir",
        default="configs/dataset",
        help="Directory holding the partial dataset YAMLs (base-/flavor-).",
    )
    parser.add_argument(
        "--dataset",
        default="tcga-brca-test",
        help="Dataset base token (selects base-{token}.yaml for tiling params).",
    )

    # --- Feature cache / encoder fallback.
    parser.add_argument(
        "--tile-model-config",
        default="configs/model-resnet50-full.yaml",
        help=(
            "Tile-model YAML providing the frozen encoder. Only used when no "
            "cached features exist for the slide."
        ),
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help=(
            "Directory containing precomputed <slide_id>.pt feature files. "
            "Defaults to <root_dir>/features/<tile-model-stem>."
        ),
    )

    # --- Slide + ROI.
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

    # --- Output + rendering.
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/attention_heat_map",
        help="Directory to write the five figures + metadata.json into.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Tiles per extreme row.")
    parser.add_argument(
        "--tissue-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlay the BCSS ground-truth tissue segmentation on the ROI crop "
            "(writes 06_roi_tissue_label.png). Use --no-tissue-label to disable. "
            "Only rendered when the chosen ROI overlaps a BCSS-annotated region."
        ),
    )
    parser.add_argument(
        "--tissue-alpha",
        type=float,
        default=0.45,
        help="Opacity of the tissue-label overlay on the ROI crop.",
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=2048,
        help="Longest side of the whole-slide overview rendering.",
    )
    parser.add_argument(
        "--roi-max-dim",
        type=int,
        default=1024,
        help="Longest side of the ROI crop.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Opacity of the attention overlay (0=invisible, 1=opaque).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=_parse_percentile,
        default=(2.0, 98.0),
        help="lo,hi percentiles for clipping attention before scaling to [0,1].",
    )
    parser.add_argument(
        "--roi-box-linewidth",
        type=float,
        default=2.5,
        help="Line width of the black ROI rectangle on the slide overview.",
    )
    parser.add_argument(
        "--extreme-tile-size",
        type=int,
        default=256,
        help="Pixel size each extreme-attention example tile is rendered at.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Tiles per encoder forward (only used when features aren't cached).",
    )
    return parser


def main() -> None:
    """CLI entrypoint: compose the MB config, run, and export the figures."""
    args = _build_arg_parser().parse_args()
    roi_xywh = _parse_roi(list(args.roi))
    logger = setup_logger(os.path.join("logs", "evaluation"), "attention_heat_map")

    # Compose the aggregator config. base/variant are pinned to the
    # multi-branch DualCLAM (this script is MB-only).
    aggregator_cfg = load_aggregator_config(
        args.aggregator_config_dir,
        base="clam",
        variant="mb",
        add_on=(args.add_on or None),
        subtasks=args.subtask,
        encoder=args.encoder,
        pretext=args.pretext,
    )
    model_params = aggregator_cfg.get("params", {})
    if not isinstance(model_params, dict):
        raise TypeError("Aggregator config 'params' must be a dict.")

    # Hard guard: refuse anything that is not multi-branch.
    attn_kwargs = model_params.get("attn_kwargs", {})
    if not (isinstance(attn_kwargs, dict) and attn_kwargs.get("multi_branch") is True):
        raise ValueError(
            "attention_heat_map.py only supports DualCLAM-MB "
            "(attn_kwargs.multi_branch == True). The composed config has "
            f"attn_kwargs={attn_kwargs!r}."
        )

    main_task = model_params.get("main_task")
    if not isinstance(main_task, str) or not main_task:
        raise ValueError("Aggregator config must declare 'params.main_task'.")
    model_subtasks = list(model_params.get("subtasks") or [])
    output_dims = model_params.get("output_dims", {})
    if not isinstance(output_dims, dict) or main_task not in output_dims:
        raise ValueError(
            "Aggregator config must declare 'params.output_dims' including the "
            f"main task '{main_task}'."
        )
    num_main_branches = int(output_dims[main_task])

    # Resolve the requested task to a model task, then to an attention head.
    resolved_task = _resolve_task(
        args.task, main_task=main_task, model_subtasks=model_subtasks
    )
    attention_head, task_desc = _resolve_attention_head(
        resolved_task,
        main_task=main_task,
        model_subtasks=model_subtasks,
        num_main_branches=num_main_branches,
    )
    logger.info("Task selector: %s", task_desc)

    # Compose the slide-flavor dataset config (tile-extraction params), following
    # the main task / subtasks the model declares.
    dataset_cfg = load_dataset_config(
        args.dataset_config_dir,
        base=args.dataset,
        main_task=main_task,
        subtasks=model_subtasks,
    )

    root_dir = dataset_cfg.get("params", {}).get("root_dir")
    if not isinstance(root_dir, str):
        root_dir = None

    features_dir = args.features_dir
    if features_dir is None and root_dir is not None:
        features_dir = _default_features_dir(root_dir, args.tile_model_config)

    result = compute_slide_attention(
        dataset_cfg=dataset_cfg,
        aggregator_cfg=aggregator_cfg,
        slide_path=args.slide_path,
        features_dir=features_dir,
        tile_model_config_path=args.tile_model_config,
        attention_head=attention_head,
        clip_percentile=args.clip_percentile,
        device_str=args.device,
        chunk_size=args.chunk_size,
        logger=logger,
    )

    # A human-readable label for the figure headers, e.g.
    # "subtyping (branch 3)" or "cnv_regression (branch 5)".
    task_label = f"{resolved_task} (branch {result.branch_idx})"

    # Load the BCSS tissue-segmentation ground truth for this slide (optional).
    tissue_slide_rois = None
    tissue_code_to_name: dict[int, str] = {}
    if args.tissue_label:
        if root_dir is None:
            logger.warning(
                "Dataset config has no 'root_dir'; cannot load tissue labels."
            )
        else:
            try:
                gt_codes, _slide_df, roi_df = load_tissue_metadata(root_dir, logger)
                tissue_code_to_name = {int(code): name for name, code in gt_codes.items()}
                slide_name = derive_tissue_slide_name(
                    result.slide_path, result.submitter_id
                )
                tissue_slide_rois = build_tissue_roi_groups(roi_df).get(slide_name)
                if tissue_slide_rois is None or tissue_slide_rois.empty:
                    logger.warning(
                        "No BCSS tissue ROIs for slide_name=%s; the tissue overlay "
                        "will be skipped.",
                        slide_name,
                    )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "Could not load tissue labels (%s); skipping tissue overlay.", exc
                )

    saved = export_heatmap_figures(
        result=result,
        roi_xywh_l0=roi_xywh,
        output_dir=args.output_dir,
        task_label=task_label,
        top_k=args.top_k,
        thumbnail_max_size=args.thumbnail_max_size,
        roi_max_dim=args.roi_max_dim,
        alpha=args.alpha,
        cmap=_DEFAULT_CMAP,
        roi_box_linewidth=args.roi_box_linewidth,
        extreme_tile_size=args.extreme_tile_size,
        logger=logger,
        root_dir=root_dir,
        draw_tissue=args.tissue_label,
        tissue_alpha=args.tissue_alpha,
        tissue_slide_rois=tissue_slide_rois,
        tissue_code_to_name=tissue_code_to_name,
    )
    logger.info(
        "Exported %d assets for task=%s to %s",
        len(saved),
        resolved_task,
        os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()
