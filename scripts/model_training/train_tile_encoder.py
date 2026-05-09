"""Train a tile-level model with Lightning."""

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
from torch.nn.parameter import UninitializedParameter

from augur.datasets.dataset_abc import DatasetABC
from augur.datasets.factory import get_dataset_from_config
from augur.models.tile_level.tile_model import TileModel
from augur.utils.config import load_yaml_config
from augur.utils.logger import setup_logger

mp.set_sharing_strategy("file_system")


def _setup_logger_for_training(log_dir: str) -> logging.Logger:
    """Set up a process logger for training."""
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="train_tile_encoder", rank_zero_only=True)


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

    candidate_path = str(value)
    if os.path.isabs(candidate_path):
        return candidate_path

    resolved_candidate = os.path.join(config_dir, candidate_path)
    if os.path.exists(resolved_candidate):
        return str(resolved_candidate)
    return value


def _resolve_model_component_paths(
    config: dict[str, Any],
    *,
    config_path: str,
) -> dict[str, Any]:
    """Resolve nested model component config paths relative to the model config."""
    resolved_config = dict(config)
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
    model: TileModel,
    datamodule: DatasetABC,
    logger: logging.Logger,
    dataset_config_path: str,
) -> None:
    """Warn when the datamodule is configured with tasks the model will ignore."""
    dataset_tasks_value = getattr(datamodule, "tasks", None)
    if not isinstance(dataset_tasks_value, (list, tuple)):
        return

    dataset_tasks = tuple(str(task_name) for task_name in dataset_tasks_value)
    model_tasks = tuple(str(task_name) for task_name in model.decoders.keys())
    logger.info("Model decoder tasks: %s", list(model_tasks))
    logger.info("Datamodule tasks: %s", list(dataset_tasks))

    missing_dataset_tasks = sorted(set(model_tasks).difference(dataset_tasks))
    if missing_dataset_tasks:
        raise ValueError(
            "Dataset config is missing task(s) required by the model: "
            f"{missing_dataset_tasks}. Loaded datamodule tasks: {list(dataset_tasks)}. "
            f"Model decoder tasks: {list(model_tasks)}."
        )

    extra_dataset_tasks = sorted(set(dataset_tasks).difference(model_tasks))
    if not extra_dataset_tasks:
        return

    config_dir = os.path.dirname(os.path.abspath(dataset_config_path))
    primary_tasks = [
        task_name for task_name in model_tasks if task_name != "tissue_segmentation"
    ]
    suggested_config_path: str | None = None
    if len(primary_tasks) == 1:
        candidate_path = os.path.join(
            config_dir, f"tile_dataset-TCGA-BRCA-{primary_tasks[0]}.yaml"
        )
        if os.path.exists(candidate_path):
            suggested_config_path = os.path.abspath(candidate_path)

    suggestion = ""
    if suggested_config_path is not None:
        suggestion = f" Prefer using dataset config {suggested_config_path}."

    logger.warning(
        "Datamodule includes task(s) with no matching model decoder: %s. "
        "These extra tasks still incur data setup work and can make distributed "
        "startup much slower.%s",
        extra_dataset_tasks,
        suggestion,
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
    model: TileModel,
    checkpoint_path: str,
    *,
    map_location: str | torch.device,
    logger: logging.Logger,
) -> None:
    """Load model weights from a plain state dict or Lightning checkpoint."""
    resolved_path = os.path.abspath(checkpoint_path)
    checkpoint = torch.load(
        resolved_path,
        map_location=map_location,
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded model weights from %s", resolved_path)
    if missing_keys:
        logger.warning("Missing keys while loading %s: %s", resolved_path, missing_keys)
    if unexpected_keys:
        logger.warning(
            "Unexpected keys while loading %s: %s", resolved_path, unexpected_keys
        )


def _save_model_weights(
    model: TileModel,
    checkpoint_path: str,
    *,
    logger: logging.Logger,
) -> None:
    """Save the model weights as a plain state dict."""
    resolved_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    torch.save(model.state_dict(), resolved_path)
    logger.info("Saved model state dict to %s", resolved_path)


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
    config_path: str,
    logger: logging.Logger,
    *,
    resume_checkpoint_path: str | None = None,
) -> TileModel:
    """Create a ``TileModel`` from a YAML config and optional checkpoint."""
    config = _resolve_model_component_paths(
        load_yaml_config(config_path),
        config_path=config_path,
    )
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Model config 'params' must be provided as a dict.")

    model = TileModel.from_config(params)

    checkpoint_path = config.get("checkpoint_path")
    device = config.get("device", "cpu")
    if checkpoint_path:
        resolved_checkpoint_path = os.path.abspath(checkpoint_path)
        if resume_checkpoint_path is not None:
            logger.info(
                "Skipping model weight preload from %s because training will resume "
                "from Lightning checkpoint %s",
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
                "Checkpoint path %s was configured but does not exist. "
                "Training will start from scratch.",
                resolved_checkpoint_path,
            )

    return model


def _load_dataset(config_path: str, logger: logging.Logger) -> DatasetABC:
    """Instantiate the dataset datamodule from a YAML config."""
    config = load_yaml_config(config_path)
    dataset = get_dataset_from_config(config)
    logger.info("Loaded datamodule from %s", os.path.abspath(config_path))
    return dataset


def _has_uninitialized_parameters(model: torch.nn.Module) -> bool:
    """Return whether the model still contains lazy parameters."""
    return any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    )


def _initialize_lazy_modules_from_dataloader(
    model: TileModel,
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


def train(
    *,
    model_config_path: str,
    dataset_config_path: str,
    training_config_path: str,
) -> tuple[Trainer, TileModel, DatasetABC]:
    """Run training for a tile-level model."""
    resolved_model_config_path = os.path.abspath(model_config_path)
    resolved_dataset_config_path = os.path.abspath(dataset_config_path)
    resolved_training_config_path = os.path.abspath(training_config_path)

    if not os.path.exists(resolved_model_config_path):
        raise FileNotFoundError(f"Model config not found: {resolved_model_config_path}")
    if not os.path.exists(resolved_dataset_config_path):
        raise FileNotFoundError(
            f"Dataset config not found: {resolved_dataset_config_path}"
        )
    if not os.path.exists(resolved_training_config_path):
        raise FileNotFoundError(
            f"Training config not found: {resolved_training_config_path}"
        )

    training_config = load_yaml_config(resolved_training_config_path)
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
    run_name = _get_training_value(
        training_config,
        "run_name",
        os.path.splitext(os.path.basename(resolved_model_config_path))[0],
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

    os.makedirs(default_root_dir, exist_ok=True)
    logger = _setup_logger_for_training(os.path.join(default_root_dir, "logs"))
    seed_everything(seed, workers=True)

    logger.info("Starting training run '%s'", run_name)
    logger.info("Model config: %s", resolved_model_config_path)
    logger.info("Dataset config: %s", resolved_dataset_config_path)
    logger.info("Training config: %s", resolved_training_config_path)

    checkpoint_dir = os.path.join(default_root_dir, "checkpoints", str(run_name))
    resume_from = _resolve_resume_checkpoint_path(
        training_config,
        checkpoint_dir=checkpoint_dir,
    )
    if resume_from is not None:
        logger.info("Resuming training from Lightning checkpoint %s", resume_from)

    model = _create_model(
        resolved_model_config_path,
        logger,
        resume_checkpoint_path=resume_from,
    )
    datamodule = _load_dataset(resolved_dataset_config_path, logger)
    _warn_on_task_mismatch(
        model=model,
        datamodule=datamodule,
        logger=logger,
        dataset_config_path=resolved_dataset_config_path,
    )
    _initialize_lazy_modules_from_dataloader(
        model,
        datamodule,
        logger=logger,
    )

    checkpoint_callback = _build_checkpoint_callback(
        training_config,
        checkpoint_dir=checkpoint_dir,
        run_name=str(run_name),
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

    trainer = Trainer(
        default_root_dir=str(default_root_dir),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,  # type: ignore
        num_nodes=num_nodes,
        sync_batchnorm=sync_batchnorm,
        precision=precision,  # type: ignore
        max_epochs=max_epochs,
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

    if test_after_fit:
        test_ckpt_path = "best" if checkpoint_callback.best_model_path else None
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=test_ckpt_path,
        )

    model_config = load_yaml_config(resolved_model_config_path)
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
    return trainer, model, datamodule


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the training script."""
    parser = argparse.ArgumentParser(description="Train a tile-level model.")
    parser.add_argument(
        "--model-config",
        default="configs/model-resnet50-hematoxylin.yaml",
        help="Path to the model YAML config.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/tile_dataset-TCGA-BRCA-test.yaml",
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training-config.yaml",
        help="Path to the training YAML config.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for tile-model training."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    train(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        training_config_path=args.training_config,
    )


if __name__ == "__main__":
    main()
