"""Train a slide-level aggregator (DualCLAM / EmbeddingMIL) with Lightning."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import torch
import torch.multiprocessing as mp
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment  # type: ignore
from torch.nn.parameter import UninitializedParameter

from augur.datasets.dataset_abc import DatasetABC
from augur.datasets.factory import get_dataset_from_config
from augur.models.model_abc import ModelABC
from augur.models.slide_level.factory import (
    get_module_from_config as get_slide_module_from_config,
)
from augur.utils.config import (
    load_aggregator_config,
    load_dataset_config,
    load_trainer_config,
    load_yaml_config,
)
from augur.utils.logger import setup_logger

mp.set_sharing_strategy("file_descriptor")


def _setup_logger_for_training(log_dir: str) -> logging.Logger:
    """Set up a process logger for training."""
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="train_slide_aggregator", rank_zero_only=True)


def _get_training_value(
    training_config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Read a trainer option from either ``trainer`` or the top level."""
    trainer_config = training_config.get("trainer", {})
    if isinstance(trainer_config, dict) and key in trainer_config:
        return trainer_config[key]
    return training_config.get(key, default)


def _resolve_relative_config_path(
    value: Any,
    *,
    config_dir: str,
) -> Any:
    """Resolve a relative config path against the model-config directory."""
    if not isinstance(value, str):
        return value
    if os.path.isabs(value):
        return value
    candidate = os.path.join(config_dir, value)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return value


def _resolve_tile_model_paths(
    tile_model_config: dict[str, Any],
    *,
    config_path: str,
) -> dict[str, Any]:
    """Resolve encoder_config / decoders_config paths inside a TileModel config."""
    resolved_config = dict(tile_model_config)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    params = resolved_config.get("params", {})

    if not isinstance(params, dict):
        return resolved_config

    resolved_params = dict(params)
    if "encoder_config" in resolved_params:
        resolved_params["encoder_config"] = _resolve_relative_config_path(
            resolved_params["encoder_config"],
            config_dir=config_dir,
        )

    decoders_config = resolved_params.get("decoders_config")
    if isinstance(decoders_config, dict):
        resolved_params["decoders_config"] = {
            task_name: _resolve_relative_config_path(
                decoder_spec,
                config_dir=config_dir,
            )
            for task_name, decoder_spec in decoders_config.items()
        }

    resolved_config["params"] = resolved_params
    return resolved_config


def _resolve_aggregator_component_paths(
    config: dict[str, Any],
    *,
    base_dir: str,
) -> dict[str, Any]:
    """Resolve the aggregator's ``tile_model_config`` path and inline-load it.

    The aggregator config may reference the tile-level model via a relative
    ``params.tile_model_config`` path. The aggregator's ``from_config``
    (``DualCLAM`` / ``EmbeddingMIL``) can load a string path via
    ``_resolve_component_config``, but nested ``encoder_config`` /
    ``decoders_config`` paths inside the tile-model YAML stay relative to
    that YAML's directory — not the CWD. This helper pre-loads the referenced
    YAML, resolves those nested paths to absolute paths, and replaces
    ``params.tile_model_config`` with the resolved dict so the aggregator
    factory receives a fully-resolved inline config regardless of CWD.
    """
    resolved_config = dict(config)
    config_dir = os.path.abspath(base_dir)
    params = resolved_config.get("params", {})

    if not isinstance(params, dict):
        return resolved_config

    resolved_params = dict(params)
    tile_model_spec = resolved_params.get("tile_model_config", None)

    if isinstance(tile_model_spec, str):
        tile_model_path = _resolve_relative_config_path(
            tile_model_spec, config_dir=config_dir
        )
        if not os.path.isabs(tile_model_path):
            tile_model_path = os.path.abspath(os.path.join(config_dir, tile_model_spec))
        tile_model_dict = load_yaml_config(tile_model_path)
        tile_model_dict = _resolve_tile_model_paths(
            tile_model_dict, config_path=tile_model_path
        )
        resolved_params["tile_model_config"] = tile_model_dict
    elif isinstance(tile_model_spec, dict):
        resolved_params["tile_model_config"] = tile_model_spec
    elif tile_model_spec is not None:
        raise TypeError(
            "aggregator 'tile_model_config' must be a string path or a dict. "
            f"Got: {type(tile_model_spec)!r}"
        )

    resolved_config["params"] = resolved_params
    return resolved_config


def _parse_devices(devices: Any) -> str | int | list[int]:
    """Parse Lightning ``devices`` values from YAML or CLI."""
    if devices is None:
        return "auto"
    if isinstance(devices, (int, list)):
        return devices
    if not isinstance(devices, str):
        raise TypeError(f"Unsupported devices value: {devices!r}")

    if devices == "auto":
        return "auto"
    if "," in devices:
        return [int(device.strip()) for device in devices.split(",") if device.strip()]
    return int(devices)


def _resolve_accelerator_and_devices(
    training_config: dict[str, Any],
) -> tuple[str, str | int | list[int]]:
    """Resolve Lightning accelerator/device settings from the training config."""
    accelerator = _get_training_value(training_config, "accelerator")
    devices = _get_training_value(training_config, "devices")

    if accelerator is not None or devices is not None:
        return str(accelerator or "auto"), _parse_devices(devices)

    device = _get_training_value(training_config, "device", "auto")
    if not isinstance(device, str):
        raise TypeError("training-config 'device' must be a string when provided.")

    normalized = device.strip().lower()
    if normalized in {"auto"}:
        return "auto", "auto"
    if normalized in {"cpu"}:
        return "cpu", 1
    if normalized in {"cuda", "gpu"}:
        return "gpu", 1
    if normalized.startswith("cuda:"):
        return "gpu", [int(normalized.split(":", maxsplit=1)[1])]

    return "auto", _parse_devices(device)


def _warn_on_task_mismatch(
    *,
    model: ModelABC,
    datamodule: DatasetABC,
    logger: logging.Logger,
) -> None:
    """Fail fast when the datamodule's main task or subtasks diverge from the model's.

    A main-task mismatch is fatal because the per-task loss uses a fixed
    target shape/dtype (e.g. ``(B,)`` long for ``subtyping`` cross-entropy vs.
    ``(B, D)`` float for ``sbs_regression``); raising here surfaces the
    misconfiguration at warmup instead of mid-training.
    """
    model_main = getattr(model, "main_task", None)
    dataset_main = getattr(datamodule, "main_task", None)
    logger.info("Model main_task=%s, dataset main_task=%s", model_main, dataset_main)
    if model_main and dataset_main and model_main != dataset_main:
        raise ValueError(
            f"Model main_task={model_main!r} does not match datamodule "
            f"main_task={dataset_main!r}. The per-task loss expects targets "
            "from the matching dataset task; reconfigure the dataset or "
            "model so the main tasks line up."
        )

    model_subtask = list(getattr(model, "subtasks", []) or [])
    dataset_subtask = list(getattr(datamodule, "subtasks", []) or [])
    logger.info("Model subtasks: %s", model_subtask)
    logger.info("Datamodule subtasks: %s", dataset_subtask)

    missing = sorted(set(model_subtask).difference(dataset_subtask))
    if missing:
        raise ValueError(
            "Datamodule is missing subtask(s) required by the model: "
            f"{missing}. Model subtasks={model_subtask}. "
            f"Datamodule subtasks={dataset_subtask}."
        )

    extra = sorted(set(dataset_subtask).difference(model_subtask))
    if extra:
        logger.info(
            "Datamodule provides extra subtask(s) the model ignores: %s",
            extra,
        )


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Normalize plain-state-dict and Lightning checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint to deserialize into a dict-like object.")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return dict(checkpoint["state_dict"])
    if "model_state_dict" in checkpoint and isinstance(
        checkpoint["model_state_dict"], dict
    ):
        return dict(checkpoint["model_state_dict"])
    return dict(checkpoint)


def _load_model_weights(
    model: ModelABC,
    checkpoint_path: str,
    *,
    map_location: str | torch.device,
    logger: logging.Logger,
) -> None:
    """Load aggregator weights from a plain state dict or Lightning checkpoint."""
    resolved_path = os.path.abspath(checkpoint_path)
    checkpoint = torch.load(
        resolved_path,
        map_location=map_location,
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded aggregator weights from %s", resolved_path)
    if missing_keys:
        logger.warning("Missing keys while loading %s: %s", resolved_path, missing_keys)
    if unexpected_keys:
        logger.warning(
            "Unexpected keys while loading %s: %s", resolved_path, unexpected_keys
        )


def _save_model_weights(
    model: ModelABC,
    checkpoint_path: str,
    *,
    logger: logging.Logger,
) -> None:
    """Save the aggregator weights as a plain state dict."""
    resolved_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    torch.save(model.state_dict(), resolved_path)
    logger.info("Saved aggregator state dict to %s", resolved_path)


def _resolve_resume_checkpoint_path(
    training_config: dict[str, Any],
    *,
    checkpoint_dir: str,
) -> str | None:
    """Resolve the Lightning checkpoint path used to resume training."""
    configured_resume = _get_training_value(training_config, "resume_from", None)
    default_last_checkpoint = os.path.abspath(os.path.join(checkpoint_dir, "last.ckpt"))

    if configured_resume is None:
        return (
            default_last_checkpoint if os.path.exists(default_last_checkpoint) else None
        )
    if not isinstance(configured_resume, str):
        raise TypeError("training-config 'resume_from' must be a string when provided.")

    normalized_resume = configured_resume.strip()
    if not normalized_resume or normalized_resume.lower() == "none":
        return None
    if normalized_resume.lower() in {"auto", "last"}:
        return (
            default_last_checkpoint if os.path.exists(default_last_checkpoint) else None
        )

    resolved_resume = os.path.abspath(normalized_resume)
    if not os.path.exists(resolved_resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {resolved_resume}")
    return resolved_resume


def _create_model(
    model_config: dict[str, Any],
    logger: logging.Logger,
    *,
    base_dir: str,
    resume_checkpoint_path: str | None = None,
) -> ModelABC:
    """Create a slide-level aggregator from a merged config and optional checkpoint.

    Dispatches via :func:`get_slide_module_from_config` so both ``DualCLAM``
    and ``EmbeddingMIL`` configs are handled uniformly. The tile encoder is
    built and preloaded internally by the aggregator's ``from_config`` using
    the nested ``tile_model_config`` entry, if any. The optional top-level
    aggregator ``checkpoint_path`` is then loaded here so training can
    warm-start from a previously exported aggregator. ``base_dir`` is the
    directory used to resolve any relative ``tile_model_config`` path that
    appears in the merged config.
    """
    config = _resolve_aggregator_component_paths(
        model_config,
        base_dir=base_dir,
    )
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Aggregator config 'params' must be provided as a dict.")

    model = get_slide_module_from_config(config)

    checkpoint_path = config.get("checkpoint_path")
    device = config.get("device", "cpu")
    if checkpoint_path:
        resolved_checkpoint_path = os.path.abspath(checkpoint_path)
        if resume_checkpoint_path is not None:
            logger.info(
                "Skipping aggregator weight preload from %s because training will "
                "resume from Lightning checkpoint %s",
                resolved_checkpoint_path,
                os.path.abspath(resume_checkpoint_path),
            )
        elif os.path.exists(resolved_checkpoint_path):
            _load_model_weights(
                model,
                resolved_checkpoint_path,
                map_location=device,
                logger=logger,
            )
        else:
            logger.warning(
                "Aggregator checkpoint path %s was configured but does not exist. "
                "Training will start from scratch.",
                resolved_checkpoint_path,
            )

    return model


def _load_dataset(dataset_config: dict[str, Any], logger: logging.Logger) -> DatasetABC:
    """Instantiate the dataset datamodule from a merged config dict."""
    dataset = get_dataset_from_config(dataset_config)
    logger.info(
        "Instantiated datamodule %s with main_task=%s subtasks=%s",
        type(dataset).__name__,
        getattr(dataset, "main_task", None),
        getattr(dataset, "subtasks", None),
    )
    return dataset


def _has_uninitialized_parameters(model: torch.nn.Module) -> bool:
    """Return whether the model still contains lazy parameters."""
    return any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    )


def _initialize_lazy_modules_from_dataloader(
    model: ModelABC,
    datamodule: DatasetABC,
    *,
    logger: logging.Logger,
) -> None:
    """Materialize lazy modules from one no-grad batch before Trainer/DDP setup."""
    if not _has_uninitialized_parameters(model):
        return

    logger.info(
        "Detected uninitialized lazy parameters. Running one no-grad warmup "
        "forward pass from the training dataloader before Trainer setup."
    )

    if getattr(datamodule, "train_dataset", None) is None:
        datamodule.setup(stage="fit")

    train_dataloader = datamodule.train_dataloader()
    try:
        warmup_batch = next(iter(train_dataloader))
    except StopIteration as exc:
        raise RuntimeError(
            "The training dataloader is empty, so lazy model parameters could "
            "not be initialized before Trainer/DDP setup."
        ) from exc

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(warmup_batch)
    except Exception as exc:
        raise RuntimeError(
            "The warmup forward pass used to initialize lazy model parameters "
            "failed on the first training batch."
        ) from exc
    finally:
        model.train(was_training)

    if _has_uninitialized_parameters(model):
        raise RuntimeError(
            "The warmup forward pass completed, but some lazy model parameters "
            "are still uninitialized."
        )

    logger.info("Lazy model parameters initialized successfully.")


def _build_checkpoint_callback(
    training_config: dict[str, Any],
    *,
    checkpoint_dir: str,
    run_name: str,
) -> ModelCheckpoint:
    """Build the Lightning checkpoint callback from training-config.yaml."""
    checkpoint_config = training_config.get("checkpoint", {})
    if checkpoint_config is None:
        checkpoint_config = {}
    if not isinstance(checkpoint_config, dict):
        raise TypeError("training-config 'checkpoint' must be a dict.")

    filename = checkpoint_config.get("filename", f"{run_name}" + "-{epoch:02d}")
    monitor = checkpoint_config.get("monitor", "val/loss")
    mode = checkpoint_config.get("mode", "min")
    save_top_k = int(checkpoint_config.get("save_top_k", 1))
    save_last = bool(checkpoint_config.get("save_last", True))
    every_n_epochs = int(
        checkpoint_config.get(
            "every_n_epochs",
            training_config.get("checkpoint_interval", 1),
        )
    )

    return ModelCheckpoint(
        dirpath=os.path.abspath(checkpoint_dir),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        every_n_epochs=every_n_epochs,
        auto_insert_metric_name=False,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the training script.

    The aggregator config is composed from eight partial axes under
    ``--config-dir/aggregator/`` (see
    :func:`augur.utils.config.load_aggregator_config`):

    - ``--base``: bag-level architecture (``clam`` or ``mil``).
    - ``--subtask``: zero or more aggregator-level auxiliary subtasks
      layered onto the CLAM base. Tokens drawn from ``sbs``, ``dbs``,
      ``id``, ``cnv`` (COSMIC signature classes). Multiple tokens stack
      one head per subtask. Omit for plain CLAM. Not allowed for MIL.
    - ``--variant``: variant within the base. CLAM uses ``sb`` / ``mb``;
      MIL uses ``mean`` / ``max`` / ``attention``.
    - ``--add-on``: optional attention add-on; ``gated`` is the only
      supported value and only applies with attention-based variants.
    - ``--encoder``: encoder architecture (e.g. ``resnet50``,
      ``prov-gigapath``) — controls ``enc_dim``.
    - ``--pretext``: encoder pretext task; consumed only to compose the
      aggregator's ``checkpoint_path`` (and to select the matching
      ``pretext-{name}.yaml`` marker partial).

    The optimizer + LR-scheduler recipe is pulled in automatically from
    ``configs/optimizers.yaml`` (each ``base-{base}.yaml`` extends it).

    The data config is similarly composed via
    :func:`augur.utils.config.load_dataset_config`:

    - ``--dataset``: dataset base token (e.g. ``tcga-brca``,
      ``tcga-brca-test``) → ``base-{token}.yaml``.
    - ``--precomputed`` *(flag)*: use a precomputed-feature datamodule
      (``TCGAFeatureDataset``) instead of the default slide datamodule
      (``TCGASlideDataset``); merges ``flavor-feature.yaml`` on top.
    - ``--features-dir``: explicit features directory when
      ``--precomputed`` is set; otherwise defaults to
      ``<root_dir>/features/<encoder>-<pretext>``.

    ``main_task`` and ``subtasks`` for the data config are read from the
    merged aggregator config so the two sides cannot drift.
    """
    parser = argparse.ArgumentParser(description="Train a slide-level aggregator.")

    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--aggregator-config-subdir",
        default="aggregator",
        help=(
            "Subdirectory of --config-dir holding the partial aggregator "
            "YAMLs (base-/subtask-/variant-/add-on-/encoder-/pretext-). "
            "Each `base-{base}.yaml` extends ../optimizers.yaml for the "
            "shared optimizer + LR-scheduler recipe."
        ),
    )
    parser.add_argument(
        "--dataset-config-subdir",
        default="dataset",
        help=(
            "Subdirectory of --config-dir holding the partial dataset "
            "YAMLs (base-/flavor-)."
        ),
    )
    parser.add_argument(
        "--trainer-config-subdir",
        default="trainer",
        help="Subdirectory of --config-dir holding the partial trainer YAMLs.",
    )
    parser.add_argument(
        "--trainer",
        default="test",
        choices=["default", "test", "long", "cpu", "smoketest-timeout"],
        help=(
            "Trainer recipe; selects `base-{name}.yaml` under the trainer "
            "partial dir. Default is `test` for safe interactive runs; pass "
            "`default` for the production GPU/DDP recipe."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="tcga-brca-test",
        help=(
            "Dataset base token; selects `base-{token}.yaml` under the "
            "dataset partial dir (e.g. 'tcga-brca', 'tcga-brca-test')."
        ),
    )
    parser.add_argument(
        "--precomputed",
        action="store_true",
        help=(
            "Use precomputed tile features (TCGAFeatureDataset) instead of "
            "extracting tiles on the fly (TCGASlideDataset). When set, "
            "`features_dir` is derived from --features-dir or defaults to "
            "<root_dir>/features/<encoder>-<pretext>."
        ),
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help=(
            "Explicit features directory (only used with --precomputed). "
            "Overrides the default <root_dir>/features/<encoder>-<pretext>."
        ),
    )
    parser.add_argument(
        "--base",
        default="clam",
        choices=["clam", "mil"],
        help="Bag-level architecture for the aggregator.",
    )
    parser.add_argument(
        "--subtask",
        default=[],
        nargs="*",
        choices=["sbs", "dbs", "id", "cnv", "full"],
        metavar="TOKEN",
        help=(
            "Zero or more aggregator-level auxiliary subtasks (COSMIC signature "
            "regression). Pass multiple tokens to stack heads, e.g. "
            "`--subtask sbs dbs`. Pass `full` as a shorthand for all four "
            "subtasks. The token order doesn't matter — the merge function "
            "sorts alphabetically so any permutation of the same set yields "
            "the same run name. Only valid with `--base clam`."
        ),
    )
    parser.add_argument(
        "--variant",
        default="mb",
        choices=["sb", "mb", "mean", "max", "attention"],
        help=(
            "Variant within the base: 'sb'/'mb' for CLAM; "
            "'mean'/'max'/'attention' for MIL."
        ),
    )
    parser.add_argument(
        "--add-on",
        default=None,
        choices=["gated"],
        help=(
            "Optional attention add-on. Only 'gated' is supported and only "
            "with attention-based variants (sb/mb/attention). Omit to apply "
            "no add-on."
        ),
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
        help="Encoder pretext task; used to compose the aggregator checkpoint path.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=(
            "If set, run stratified group k-fold cross-validation on the "
            "slide-level main task (currently only used by 'subtyping'). Must "
            "be paired with --fold-idx. Unknown (class 0) is excluded from the "
            "CV. When unset, the default fraction-based train/val/test split "
            "is used."
        ),
    )
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help=(
            "0-based index of the fold whose test partition is the held-out "
            "set for this run. Must be paired with --n-folds and satisfy "
            "0 <= fold_idx < n_folds. The run_name and the aggregator's "
            "checkpoint_path stem are suffixed with '-fold{idx}' so per-fold "
            "outputs do not collide."
        ),
    )

    return parser


def train(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    run_name: str,
    *,
    aggregator_config_dir: str,
) -> None:
    """Run training for the slide aggregator.

    ``model_config`` is the merged aggregator config produced by
    :func:`augur.utils.config.load_aggregator_config` from the
    base/subtask/variant/add-on/encoder/pretext partials under
    ``aggregator_config_dir`` (the base partial pulls in the shared
    optimizer + LR-scheduler recipe via its ``extends`` directive).
    Relative ``tile_model_config`` paths inside the merged dict, if any,
    are resolved against ``aggregator_config_dir``.

    ``dataset_config`` is the merged datamodule config produced by
    :func:`augur.utils.config.load_dataset_config`. Its ``main_task``
    and ``subtasks`` must already match the aggregator config (the CLI
    derives them from ``model_config`` to enforce this).

    ``training_config`` is the merged trainer recipe produced by
    :func:`augur.utils.config.load_trainer_config`.
    """
    resolved_aggregator_config_dir = os.path.abspath(aggregator_config_dir)
    accelerator, devices = _resolve_accelerator_and_devices(training_config)
    max_epochs = int(
        _get_training_value(
            training_config,
            "max_epochs",
            training_config.get("num_epochs", 10),
        )
    )
    precision = str(_get_training_value(training_config, "precision", "32-true"))
    strategy = str(_get_training_value(training_config, "strategy", "auto"))
    num_nodes = int(_get_training_value(training_config, "num_nodes", 1))
    sync_batchnorm = bool(_get_training_value(training_config, "sync_batchnorm", False))
    default_root_dir = os.path.abspath(
        _get_training_value(
            training_config,
            "default_root_dir",
            "outputs/model_training",
        )
    )
    log_every_n_steps = int(
        _get_training_value(training_config, "log_every_n_steps", 10)
    )
    accumulate_grad_batches = int(
        _get_training_value(training_config, "accumulate_grad_batches", 1)
    )
    seed = int(_get_training_value(training_config, "seed", 42))
    fast_dev_run = bool(_get_training_value(training_config, "fast_dev_run", False))
    limit_train_batches = _get_training_value(
        training_config, "limit_train_batches", 1.0
    )
    limit_val_batches = _get_training_value(training_config, "limit_val_batches", 1.0)
    limit_test_batches = _get_training_value(training_config, "limit_test_batches", 1.0)
    num_sanity_val_steps = int(
        _get_training_value(training_config, "num_sanity_val_steps", 2)
    )
    test_after_fit = bool(_get_training_value(training_config, "test_after_fit", True))
    # Wall-clock budget (e.g. "DD:HH:MM:SS") from the trainer config. Set strictly
    # below the sbatch --time so Lightning stops at a batch boundary, checkpoints,
    # and exits cleanly before SLURM's time-limit SIGKILL would tear the process
    # down mid-writeback. This is a launcher-agnostic Timer, so it is safe no
    # matter how the ranks are spawned. Resumes from last.ckpt on the next run.
    max_time = _get_training_value(training_config, "max_time", None)

    os.makedirs(default_root_dir, exist_ok=True)
    logger = _setup_logger_for_training(os.path.join(default_root_dir, "logs"))
    seed_everything(seed, workers=True)

    logger.info("Starting training run '%s'", run_name)
    logger.info("Aggregator config dir: %s", resolved_aggregator_config_dir)
    logger.info("Datamodule: %s", dataset_config.get("name"))

    checkpoint_dir = os.path.join(default_root_dir, "checkpoints", run_name)
    resume_from = _resolve_resume_checkpoint_path(
        training_config,
        checkpoint_dir=checkpoint_dir,
    )
    if resume_from is not None:
        logger.info("Resuming training from Lightning checkpoint %s", resume_from)

    model = _create_model(
        model_config,
        logger,
        base_dir=resolved_aggregator_config_dir,
        resume_checkpoint_path=resume_from,
    )
    datamodule = _load_dataset(dataset_config, logger)
    _warn_on_task_mismatch(
        model=model,
        datamodule=datamodule,
        logger=logger,
    )
    _initialize_lazy_modules_from_dataloader(
        model,
        datamodule,
        logger=logger,
    )

    checkpoint_callback = _build_checkpoint_callback(
        training_config,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )

    logger_config = training_config.get("logger", {})
    if logger_config is None:
        logger_config = {}
    if not isinstance(logger_config, dict):
        raise TypeError("training-config 'logger' must be a dict.")

    csv_logger = CSVLogger(
        save_dir=str(
            os.path.abspath(
                logger_config.get(
                    "save_dir",
                    os.path.join(default_root_dir, "lightning_logs"),
                )
            )
        ),
        name=str(logger_config.get("name", run_name)),
    )

    callbacks = [checkpoint_callback]
    if bool(_get_training_value(training_config, "enable_lr_monitor", True)):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))  # type: ignore

    # The sbatch launches one srun task per GPU (--ntasks-per-node=2 with
    # devices=2), so SLURM manages the DDP ranks and SLURMEnvironment is the
    # correct cluster environment. Disable Lightning's auto-requeue SIGTERM
    # handler: at the time limit SIGTERM then causes a prompt, clean exit (and
    # apptainer terminates) instead of being "bypassed" until SLURM escalates to
    # SIGKILL mid-write. We rely on `max_time` above for the clean, checkpointed
    # stop before the limit. (Guarded by SLURM_JOB_ID so non-SLURM/local runs are
    # unaffected.)
    plugins: list[Any] = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(SLURMEnvironment(auto_requeue=False))

    trainer = Trainer(
        default_root_dir=str(default_root_dir),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,  # type: ignore
        num_nodes=num_nodes,
        sync_batchnorm=sync_batchnorm,
        precision=precision,  # type: ignore
        max_epochs=max_epochs,
        max_time=max_time,
        plugins=plugins,  # type: ignore
        logger=csv_logger,
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_from,
    )

    if test_after_fit and trainer.is_global_zero:
        # Run test on a single device per Lightning's recommendation: avoids
        # DistributedSampler replicating samples and halves the DataLoader
        # worker fork count, which is what otherwise OOMs the cgroup at the
        # fit -> test boundary on whole-slide bags. Force num_workers=0 so the
        # test DataLoader runs in-process and we don't fork a fat parent.
        test_ckpt_path = "best" if checkpoint_callback.best_model_path else None
        datamodule.num_workers = 0
        datamodule.persistent_workers = False
        datamodule.prefetch_factor = None
        test_trainer = Trainer(
            default_root_dir=str(default_root_dir),
            accelerator=accelerator,
            devices=1,
            num_nodes=1,
            precision=precision,  # type: ignore
            logger=csv_logger,
            limit_test_batches=limit_test_batches,
        )
        test_trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=test_ckpt_path,
        )

    export_checkpoint_path = model_config.get("checkpoint_path")
    if export_checkpoint_path:
        if checkpoint_callback.best_model_path:
            _load_model_weights(
                model,
                checkpoint_callback.best_model_path,
                map_location="cpu",
                logger=logger,
            )
        _save_model_weights(
            model,
            export_checkpoint_path,
            logger=logger,
        )

    logger.info("Training run '%s' completed.", run_name)
    if checkpoint_callback.best_model_path:
        logger.info(
            "Best Lightning checkpoint: %s", checkpoint_callback.best_model_path
        )


def main() -> None:
    """CLI entrypoint for slide-aggregator training."""
    torch.set_float32_matmul_precision("high")
    args = _build_arg_parser().parse_args()

    aggregator_config_dir = os.path.join(args.config_dir, args.aggregator_config_subdir)
    dataset_config_dir = os.path.join(args.config_dir, args.dataset_config_subdir)
    trainer_config_dir = os.path.join(args.config_dir, args.trainer_config_subdir)

    model_config = load_aggregator_config(
        aggregator_config_dir,
        base=args.base,
        variant=args.variant,
        add_on=args.add_on,
        subtasks=args.subtask,
        encoder=args.encoder,
        pretext=args.pretext,
    )

    # Pull main_task + subtasks (full task names) from the merged aggregator
    # config so the datamodule cannot drift from what the model declares.
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

    training_config = load_trainer_config(
        trainer_config_dir,
        trainer=args.trainer,
    )

    # Derive the run name from the merged config's ``checkpoint_path`` so it
    # stays in lockstep with the merge function's canonicalization (subtask
    # alphabetical sort, ``full`` shorthand collapsing). This makes
    # ``--subtask sbs dbs id cnv`` and ``--subtask full`` produce the same
    # run_name regardless of argv order.
    checkpoint_path = model_config.get("checkpoint_path", "")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            "Merged aggregator config must declare a non-empty "
            "'checkpoint_path' string."
        )
    full_model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    # When running k-fold CV, suffix run_name + checkpoint_path stem with
    # `-fold{idx}` so each fold writes to its own log dir / export path.
    # TCGASlideDataset.__init__ enforces the pairing + range checks; here we
    # only need a cheap "both provided" guard for a clean CLI error.
    if (args.n_folds is None) != (args.fold_idx is None):
        raise ValueError(
            "--n-folds and --fold-idx must be provided together (or both omitted)."
        )
    if args.n_folds is not None:
        fold_suffix = f"-fold{args.fold_idx}"
        full_model_name = f"{full_model_name}{fold_suffix}"
        ckpt_dir = os.path.dirname(checkpoint_path)
        ckpt_stem, ckpt_ext = os.path.splitext(os.path.basename(checkpoint_path))
        model_config["checkpoint_path"] = os.path.join(
            ckpt_dir, f"{ckpt_stem}{fold_suffix}{ckpt_ext}"
        )

    train(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        run_name=full_model_name,
        aggregator_config_dir=aggregator_config_dir,
    )


if __name__ == "__main__":
    main()
