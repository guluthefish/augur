"""Compare per-step training curves across tile-level pretext-task setups.

Loads metrics from PyTorch Lightning CSV logs, merges every logged
``version_*/metrics.csv`` belonging to one model config (averaging duplicate
``(epoch, step)`` rows from resumed runs), and overlays the
``train/tissue_segmentation_loss`` curve for each model on a single figure
so convergence speed can be compared across pretext-task configurations
(e.g. hematoxylin only, jigmag only, magnification only, or all three).

Each model config is loaded with :func:`augur.utils.config.load_yaml_config`
just like :mod:`scripts.visualization.tile_embeddings` so the
``extends``-based composition works identically; the run's lightning_logs
subdirectory is derived from the config's ``checkpoint_path`` stem.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from augur.utils.config import load_yaml_config

_DEFAULT_LOGS_DIR = "outputs/model_training/lightning_logs"
_DEFAULT_METRIC = "train/tissue_segmentation_loss"
_INDEX_COLS = ("epoch", "step")
_SUPERVISED_TASK = "tissue_segmentation"


def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the visualization run."""
    log_dir = os.path.join("logs", "visualization")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "tile_learning_curves.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("tile_learning_curves")
    logger.handlers = [handler, logging.StreamHandler()]
    logger.setLevel(logging.INFO)
    return logger


def _resolve_run_name(config: dict[str, Any], config_path: str) -> str:
    """Return the lightning_logs subdir name for a model config.

    Prefers the stem of ``checkpoint_path`` (e.g. ``resnet50-full``); falls
    back to the config filename stem with any leading ``model-`` stripped.
    """
    ckpt = config.get("checkpoint_path")
    if isinstance(ckpt, str) and ckpt:
        return Path(ckpt).stem
    stem = Path(config_path).stem
    return stem[len("model-") :] if stem.startswith("model-") else stem


def _resolve_pretext_tasks(config: dict[str, Any]) -> list[str]:
    """Return the configured pretext-task names (excluding the supervised target)."""
    params = config.get("params", {})
    decoders = params.get("decoders_config", {}) if isinstance(params, dict) else {}
    if not isinstance(decoders, dict):
        return []
    return [name for name in decoders.keys() if name != _SUPERVISED_TASK]


def load_run_metrics(run_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load and merge every ``version_*/metrics.csv`` under ``run_dir``.

    Concatenates each per-version CSV, drops rows missing either ``epoch``
    or ``step`` (e.g. learning-rate-only logs), then groups by
    ``(epoch, step)`` and averages all numeric columns so duplicates from
    resumed training runs collapse to one row each.
    """
    csv_paths = sorted(run_dir.glob("version_*/metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No version_*/metrics.csv files under {run_dir}")
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["__version__"] = csv_path.parent.name
        frames.append(df)
        logger.info("Loaded %d rows from %s", len(df), csv_path)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=list(_INDEX_COLS))
    combined[list(_INDEX_COLS)] = combined[list(_INDEX_COLS)].astype(int)
    numeric_cols = [c for c in combined.select_dtypes(include="number").columns]
    merged = combined.groupby(list(_INDEX_COLS), as_index=False)[numeric_cols].mean()
    return merged.sort_values("step").reset_index(drop=True)


def _curve_label(run_name: str, pretext_tasks: list[str]) -> str:
    """Format a legend label combining the run name and its pretext tasks."""
    if not pretext_tasks:
        return f"{run_name} (no pretext)"
    return f"{run_name} ({', '.join(sorted(pretext_tasks))})"


def plot_learning_curves(
    *,
    model_configs: list[str],
    logs_dir: str,
    output_path: str,
    metric: str,
    smooth_window: int,
    stage: str,
    logger: logging.Logger,
) -> None:
    """Render the comparison figure for a list of model configs.

    Parameters
    ----------
    model_configs:
        Paths to model YAML configs, one per pretext-task setup.
    logs_dir:
        Root containing ``<run_name>/version_*/metrics.csv`` subtrees.
    output_path:
        Destination path for the rendered PNG.
    metric:
        Column in the merged metrics dataframe to plot on the y-axis.
        Defaults to ``train/tissue_segmentation_loss``.
    smooth_window:
        Rolling-mean window in points along the metric series. ``1`` means
        no smoothing; useful when a metric is logged densely (e.g.
        ``train/loss_step``).
    logger:
        Logger used for progress and warnings.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = list(plt.get_cmap("tab10").colors)  # type: ignore

    for color_idx, config_path in enumerate(model_configs):
        config = load_yaml_config(config_path)
        run_name = _resolve_run_name(config, config_path)
        pretext_tasks = _resolve_pretext_tasks(config)
        run_dir = Path(logs_dir) / run_name
        merged = load_run_metrics(run_dir, logger)
        if metric not in merged.columns:
            logger.warning("Run %s has no '%s' column; skipping.", run_name, metric)
            continue
        series = merged[["step", metric]].dropna(subset=[metric])
        if series.empty:
            logger.warning("Run %s has no non-null '%s' values.", run_name, metric)
            continue
        steps = series["step"].to_numpy()
        loss = series[metric].to_numpy()
        if smooth_window > 1:
            loss = (
                pd.Series(loss)
                .rolling(window=smooth_window, min_periods=1)
                .mean()
                .to_numpy()
            )
        ax.plot(
            steps,
            loss,
            color=palette[color_idx % len(palette)],
            linewidth=1.6,
            label=_curve_label(run_name, pretext_tasks),
        )
        logger.info(
            "Plotted %d points for %s (final %s = %.4f)",
            len(steps),
            run_name,
            metric,
            float(loss[-1]),
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(metric)
    # ax.set_title(f"Tissue-segmentation {stage} loss by pretext-task setup")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=9)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning-curve figure to %s", output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the learning-curve script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-configs",
        nargs="+",
        required=True,
        help="One or more model YAML configs to compare on a single plot.",
    )
    parser.add_argument(
        "--logs-dir",
        default=_DEFAULT_LOGS_DIR,
        help="Root directory containing per-run lightning_logs subfolders.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/visualization/tile_learning_curves.png",
    )
    parser.add_argument(
        "--metric",
        default=_DEFAULT_METRIC,
        help=(
            "Metric column to plot (e.g. train/tissue_segmentation_loss, "
            "val/tissue_segmentation_loss, train/loss_step)."
        ),
    )
    parser.add_argument(
        "--stage",
        default="training",
        help=(
            "One of 'training' or 'validation' to describe the plotted metric in "
            "the figure title.  (This is just for labeling; the metric can be "
            "any column in the merged metrics dataframe.)"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Rolling-mean window over the metric series. 1 disables "
            "smoothing; raise it when plotting per-step metrics like "
            "train/loss_step that are noisy at single-step resolution."
        ),
    )
    return parser


def main() -> None:
    """CLI entrypoint: parse arguments and run :func:`plot_learning_curves`."""
    args = _build_arg_parser().parse_args()
    logger = _setup_logger()
    plot_learning_curves(
        model_configs=args.model_configs,
        logs_dir=args.logs_dir,
        output_path=args.output_path,
        metric=args.metric,
        smooth_window=args.smooth_window,
        logger=logger,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
