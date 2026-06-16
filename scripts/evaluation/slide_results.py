"""Dump slide-level predictions for a trained aggregator (subtyping + subtasks).

This is the evaluation counterpart to
``scripts/model_training/train_slide_aggregator.py``: it composes the *same*
aggregator/dataset configs from the partial-merge axes, rebuilds the model with
the identical helper used at train time (so the exported ``checkpoint_path``
weights load into a byte-compatible architecture), and runs inference over one
split's dataloader.

For each run it writes, under ``--output-dir/<model>/`` (``<model>`` is the
aggregator's checkpoint stem, e.g. ``clam-full-mb-gated-resnet50-full``):

- ``<main_task>.txt`` (e.g. ``subtyping.txt``) — a TSV with header
  ``slide_id  true_label  pred_label  <class_0>  <class_1> ...`` where each
  ``class_i`` column holds the softmax probability for that class and
  ``true_label`` / ``pred_label`` are the human-readable class names.
- ``<subtask>.txt`` (one per aggregator subtask, e.g. ``sbs_regression.txt``)
  — a TSV with header ``slide_id  <signature_0>  <signature_1> ...`` where each
  signature column holds the predicted regression value (no true label, since
  the subtasks are regression). ``<signature_i>`` is the COSMIC signature name
  (``SBS1``, ``DBS2``, ``ID7``, ``CN1`` ...) read from the labels table.

Model + data loading mirror the training script: see
:func:`scripts.model_training.train_slide_aggregator._create_model` and the
``augur.utils.config`` loaders for the merge semantics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, Dataset

# Put the repo root on the path so we can reuse the training script's
# model-construction helper verbatim. The two scripts are launched as plain
# files (``python scripts/.../foo.py``) from the repo root, so the repo root is
# not otherwise importable; ``scripts`` is an implicit namespace package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augur.datasets.dataset_abc import DatasetABC  # noqa: E402
from augur.datasets.factory import get_dataset_from_config  # noqa: E402
from augur.models.model_abc import ModelABC  # noqa: E402
from augur.utils.config import (  # noqa: E402
    load_aggregator_config,
    load_dataset_config,
)
from augur.utils.logger import setup_logger  # noqa: E402
from scripts.model_training.train_slide_aggregator import _create_model  # noqa: E402

_SPLIT_TO_STAGE = {"train": "fit", "val": "fit", "test": "test"}


def _setup_logger_for_evaluation(log_dir: str) -> logging.Logger:
    """Set up a process logger for evaluation."""
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="slide_results", rank_zero_only=True)


def _resolve_device(device: str | None) -> torch.device:
    """Resolve a ``--device`` string into a concrete ``torch.device``."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _class_column_name(class_names: list[str], index: int) -> str:
    """Name of the ``index``-th class column (falls back to ``class_{i}``)."""
    if 0 <= index < len(class_names):
        return class_names[index]
    return f"class_{index}"


def _label_name(class_names: list[str], index: int) -> str:
    """Human-readable label for a class index (falls back to the raw index)."""
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def _signature_column_name(signature_names: list[str], subtask: str, index: int) -> str:
    """Name of the ``index``-th signature column (falls back to ``{subtask}_{i}``)."""
    if 0 <= index < len(signature_names):
        return signature_names[index]
    return f"{subtask}_{index}"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser, mirroring the training script's config axes.

    The aggregator/dataset configs are composed exactly as in
    ``train_slide_aggregator.py`` so a given set of axes selects the model that
    was trained with the same axes. The extra ``--split``, ``--output-dir``,
    ``--device``, ``--batch-size`` and ``--num-workers`` flags control the
    inference pass and where predictions are written.
    """
    parser = argparse.ArgumentParser(
        description="Dump slide-level predictions for a trained aggregator."
    )

    # --- config-composition axes (must match training to find the model) -----
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--aggregator-config-subdir", default="aggregator")
    parser.add_argument("--dataset-config-subdir", default="dataset")
    parser.add_argument("--dataset", default="tcga-brca-test")
    parser.add_argument("--precomputed", action="store_true")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--base", default="clam", choices=["clam", "mil"])
    parser.add_argument(
        "--subtask",
        default=[],
        nargs="*",
        choices=["sbs", "dbs", "id", "cnv", "full"],
        metavar="TOKEN",
    )
    parser.add_argument(
        "--variant",
        default="mb",
        choices=["sb", "mb", "mean", "max", "attention"],
    )
    parser.add_argument("--add-on", default=None, choices=["gated"])
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument(
        "--pretext",
        default="full",
        choices=["full", "hematoxylin", "jigmag", "magnification", "none"],
    )
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--fold-idx", type=int, default=None)

    # --- inference / output controls -----------------------------------------
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to run predictions on.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Predictions are written under <output-dir>/<model>/.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device ('auto', 'cpu', 'cuda', 'cuda:0', ...).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Inference batch size. Defaults to 1 so each slide is encoded "
            "alone; values > 1 pad bags to the batch max, which lets padded "
            "tiles leak into attention-based aggregators."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)

    return parser


def _resolve_run_name_and_fold(
    model_config: dict[str, Any],
    *,
    n_folds: int | None,
    fold_idx: int | None,
) -> str:
    """Derive ``<model>`` and apply the per-fold checkpoint suffix in place.

    Mirrors ``train_slide_aggregator.main`` so the predictions land beside the
    matching training run: the run name is the checkpoint stem, and when a fold
    is requested the stem (and the ``checkpoint_path`` the weights load from)
    gain a ``-fold{idx}`` suffix.
    """
    checkpoint_path = model_config.get("checkpoint_path", "")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            "Merged aggregator config must declare a non-empty "
            "'checkpoint_path' string."
        )

    if (n_folds is None) != (fold_idx is None):
        raise ValueError(
            "--n-folds and --fold-idx must be provided together (or both omitted)."
        )

    full_model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if n_folds is not None:
        fold_suffix = f"-fold{fold_idx}"
        full_model_name = f"{full_model_name}{fold_suffix}"
        ckpt_dir = os.path.dirname(checkpoint_path)
        ckpt_stem, ckpt_ext = os.path.splitext(os.path.basename(checkpoint_path))
        model_config["checkpoint_path"] = os.path.join(
            ckpt_dir, f"{ckpt_stem}{fold_suffix}{ckpt_ext}"
        )
    return full_model_name


def _select_split_dataloader(
    datamodule: DatasetABC,
    split: str,
) -> tuple[DataLoader[Any], Dataset[Any]]:
    """Return a shuffle-free dataloader and the backing dataset for ``split``.

    ``setup`` is called for the stage that builds the requested split, then the
    split dataset is wrapped in a non-shuffling, drop-nothing dataloader (the
    ``train`` loader shuffles by default, which we suppress here so slide order
    is irrelevant but reproducible).
    """
    datamodule.setup(stage=_SPLIT_TO_STAGE[split])
    dataset = getattr(datamodule, f"{split}_dataset", None)
    if dataset is None:
        raise RuntimeError(
            f"The '{split}' split was not initialized by setup(); cannot run "
            "predictions on it."
        )

    batch_size = {
        "train": datamodule.batch_size,
        "val": datamodule.val_batch_size,
        "test": datamodule.test_batch_size,
    }[split]

    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": datamodule.num_workers,
        "pin_memory": datamodule.pin_memory,
        "drop_last": False,
    }
    if datamodule.collate_fn is not None:
        dataloader_kwargs["collate_fn"] = datamodule.collate_fn
    if datamodule.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = datamodule.persistent_workers
        if datamodule.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = datamodule.prefetch_factor

    return DataLoader(**dataloader_kwargs), dataset


@torch.no_grad()
def _run_inference(
    model: ModelABC,
    dataloader: DataLoader[Any],
    *,
    main_task: str,
    subtasks: list[str],
    class_names: list[str],
    signature_names: dict[str, list[str]],
    device: torch.device,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Run the model over the dataloader and collect per-slide prediction rows.

    Returns ``(main_rows, subtask_rows)`` where ``main_rows`` carries the
    subtyping probabilities and predicted/true labels and ``subtask_rows`` maps
    each subtask name to its list of per-slide regression rows.
    """
    model.eval()
    main_rows: list[dict[str, Any]] = []
    subtask_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in subtasks}
    expected_num_classes = len(class_names)
    mismatch_warned = False

    for batch in dataloader:
        if not isinstance(batch, dict):
            raise TypeError(
                "Expected a dict batch from the slide dataloader; "
                f"got {type(batch)!r}."
            )
        images = batch["image"].to(device)
        outputs = model({"image": images})

        main_logits = outputs[main_task].detach().float().cpu()
        if main_logits.ndim != 2:
            raise ValueError(
                f"Expected main-task logits of shape (B, C); got "
                f"{tuple(main_logits.shape)}."
            )
        probabilities = torch.softmax(main_logits, dim=-1)
        predicted_indices = probabilities.argmax(dim=-1)

        slide_ids = batch["metadata"]["slide_id"]
        targets = batch["target"].detach().cpu().view(-1)

        num_classes = probabilities.shape[1]
        if not mismatch_warned and num_classes != expected_num_classes:
            # The head width is fixed by output_dims[main_task] in the config,
            # while the class names are derived at runtime from the labels TSV
            # (Unknown rows skipped, conflicts dropped, survivors renumbered).
            # When they disagree, columns past the named range fall back to
            # "class_i" and out-of-range labels render as bare indices — which
            # violates the spec, and usually means a wrong checkpoint/labels
            # pairing. Warn loudly but still emit the predictions.
            logger.warning(
                "Model emits %d probability column(s) but the dataset declares "
                "%d class name(s); columns/labels beyond %d are named 'class_i' "
                "/ rendered as bare indices. Check output_dims['%s'] against the "
                "labels TSV.",
                num_classes,
                expected_num_classes,
                expected_num_classes,
                main_task,
            )
            mismatch_warned = True
        for i, slide_id in enumerate(slide_ids):
            true_index = int(targets[i])
            predicted_index = int(predicted_indices[i])
            row: dict[str, Any] = {
                "slide_id": slide_id,
                "true_label": _label_name(class_names, true_index),
                "pred_label": _label_name(class_names, predicted_index),
            }
            for class_index in range(num_classes):
                column = _class_column_name(class_names, class_index)
                row[column] = float(probabilities[i, class_index])
            main_rows.append(row)

        for subtask in subtasks:
            subtask_output = outputs[subtask].detach().float().cpu()
            if subtask_output.ndim != 2:
                raise ValueError(
                    f"Expected subtask '{subtask}' output of shape (B, D); got "
                    f"{tuple(subtask_output.shape)}."
                )
            names = signature_names.get(subtask, [])
            num_values = subtask_output.shape[1]
            for i, slide_id in enumerate(slide_ids):
                row = {"slide_id": slide_id}
                for value_index in range(num_values):
                    column = _signature_column_name(names, subtask, value_index)
                    row[column] = float(subtask_output[i, value_index])
                subtask_rows[subtask].append(row)

    logger.info("Collected predictions for %d slide(s).", len(main_rows))
    return main_rows, subtask_rows


def _write_table(rows: list[dict[str, Any]], path: str, logger: logging.Logger) -> None:
    """Write a list of row dicts to a tab-separated table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False)
    logger.info(
        "Wrote %d row(s) x %d column(s) to %s", len(frame), frame.shape[1], path
    )


def evaluate(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    run_name: str,
    *,
    aggregator_config_dir: str,
    split: str,
    output_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> None:
    """Load the trained aggregator, predict on ``split``, and write the tables."""
    resolved_aggregator_config_dir = os.path.abspath(aggregator_config_dir)
    output_run_dir = os.path.join(os.path.abspath(output_dir), run_name)
    os.makedirs(output_run_dir, exist_ok=True)

    logger = _setup_logger_for_evaluation(os.path.join(output_run_dir, "logs"))
    logger.info("Starting evaluation run '%s' on split '%s'", run_name, split)

    checkpoint_path = model_config.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not os.path.exists(
        os.path.abspath(checkpoint_path)
    ):
        raise FileNotFoundError(
            "Aggregator checkpoint not found at "
            f"{os.path.abspath(str(checkpoint_path))}. Train the model first "
            "(or pass the matching config axes) so the exported weights exist."
        )

    # _create_model rebuilds the architecture and loads checkpoint_path's
    # weights — identical to how training warm-starts an exported aggregator.
    model = _create_model(
        model_config,
        logger,
        base_dir=resolved_aggregator_config_dir,
        resume_checkpoint_path=None,
    )
    if any(isinstance(param, UninitializedParameter) for param in model.parameters()):
        # Lazy modules should have been materialized by the state-dict load
        # hook; a still-lazy parameter means a key was missing from the
        # checkpoint, so inference would run on uninitialized weights.
        raise RuntimeError(
            "Some model parameters remain uninitialized after loading the "
            "checkpoint; the checkpoint may not match the configured "
            "architecture."
        )
    model.to(device)

    main_task = getattr(model, "main_task", None)
    if not isinstance(main_task, str) or not main_task:
        raise ValueError("Loaded model does not declare a string 'main_task'.")
    subtasks = list(getattr(model, "subtasks", []) or [])
    logger.info("Model main_task=%s subtasks=%s", main_task, subtasks)

    datamodule = get_dataset_from_config(dataset_config)
    datamodule.batch_size = batch_size
    datamodule.val_batch_size = batch_size
    datamodule.test_batch_size = batch_size
    datamodule.num_workers = num_workers
    datamodule.persistent_workers = False
    datamodule.prefetch_factor = None

    dataloader, split_dataset = _select_split_dataloader(datamodule, split)

    class_names = list(getattr(split_dataset, "main_label_names", ()) or ())
    subtask_label_names = getattr(split_dataset, "subtask_label_names", {}) or {}
    signature_names = {
        subtask: list(subtask_label_names.get(subtask, ()) or ())
        for subtask in subtasks
    }
    logger.info(
        "Split '%s' has %d slide(s), %d main class(es).",
        split,
        len(split_dataset),  # type: ignore[arg-type]
        len(class_names),
    )

    main_rows, subtask_rows = _run_inference(
        model,
        dataloader,
        main_task=main_task,
        subtasks=subtasks,
        class_names=class_names,  # type: ignore
        signature_names=signature_names,  # type: ignore
        device=device,
        logger=logger,
    )

    if not main_rows:
        raise RuntimeError(
            f"No predictions were produced for split '{split}'. Is the split " "empty?"
        )

    _write_table(
        main_rows,
        os.path.join(output_run_dir, f"{main_task}.txt"),
        logger,
    )
    for subtask in subtasks:
        _write_table(
            subtask_rows[subtask],
            os.path.join(output_run_dir, f"{subtask}.txt"),
            logger,
        )

    logger.info(
        "Evaluation run '%s' completed; outputs in %s", run_name, output_run_dir
    )


def main() -> None:
    """CLI entrypoint for slide-aggregator prediction dumping."""
    args = _build_arg_parser().parse_args()

    aggregator_config_dir = os.path.join(args.config_dir, args.aggregator_config_subdir)
    dataset_config_dir = os.path.join(args.config_dir, args.dataset_config_subdir)

    model_config = load_aggregator_config(
        aggregator_config_dir,
        base=args.base,
        variant=args.variant,
        add_on=args.add_on,
        subtasks=args.subtask,
        encoder=args.encoder,
        pretext=args.pretext,
    )

    model_params = model_config.get("params", {})
    if not isinstance(model_params, dict):
        raise TypeError("Aggregator config 'params' must be a dict.")
    main_task = model_params.get("main_task")
    if not isinstance(main_task, str) or not main_task:
        raise ValueError("Aggregator config must declare 'params.main_task'.")
    model_subtasks = model_params.get("subtasks") or []

    dataset_config = load_dataset_config(
        dataset_config_dir,
        base=args.dataset,
        flavor="feature" if args.precomputed else "slide",
        main_task=main_task,
        subtasks=model_subtasks,
        encoder=args.encoder,
        pretext=args.pretext,
        features_dir=args.features_dir,
        n_folds=args.n_folds,
        fold_idx=args.fold_idx,
    )

    run_name = _resolve_run_name_and_fold(
        model_config,
        n_folds=args.n_folds,
        fold_idx=args.fold_idx,
    )

    evaluate(
        model_config=model_config,
        dataset_config=dataset_config,
        run_name=run_name,
        aggregator_config_dir=aggregator_config_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=_resolve_device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
