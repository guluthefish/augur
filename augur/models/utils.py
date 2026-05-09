"""Utility functions for model construction and optimization configuration."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from torch import Tensor, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler


OptimizerFactory = Callable[..., optim.Optimizer]
LRSchedulerFactory = Callable[..., LRScheduler]

_OPTIMIZER_REGISTRY: dict[str, OptimizerFactory] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}

_DEFAULT_OPTIMIZER_KWARGS: dict[str, dict[str, Any]] = {
    "adam": {"lr": 1e-3},
    "adamw": {"lr": 1e-3},
    "sgd": {"lr": 1e-3},
    "rmsprop": {"lr": 1e-3},
}

_LR_SCHEDULER_REGISTRY: dict[str, LRSchedulerFactory] = {
    "constantlr": lr_scheduler.ConstantLR,
    "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
    "exponentiallr": lr_scheduler.ExponentialLR,
    "linearlr": lr_scheduler.LinearLR,
    "multisteplr": lr_scheduler.MultiStepLR,
    "reducelronplateau": lr_scheduler.ReduceLROnPlateau,
    "steplr": lr_scheduler.StepLR,
}


def _normalize_registry_key(name: str) -> str:
    """Normalize config names to registry keys."""
    return name.rsplit(".", maxsplit=1)[-1].lower()


def _as_dict(
    config: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate that a config object is dict-like."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"{label} config must be a dict. Got: {type(config)!r}")
    return config


def _extract_kwargs(
    config: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Read a nested kwargs dict from ``params`` or ``kwargs``."""
    raw_kwargs = config.get("params", config.get("kwargs", {}))
    if raw_kwargs is None:
        return {}
    if not isinstance(raw_kwargs, dict):
        raise TypeError(
            f"{label} params must be provided as a dict. Got: {type(raw_kwargs)!r}"
        )
    return dict(raw_kwargs)


def _extract_scheduler_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read Lightning scheduler metadata from common config keys."""
    raw_scheduler_config = config.get(
        "config",
        config.get("lightning_config", config.get("scheduler_config", {})),
    )
    if raw_scheduler_config is None:
        return {}
    if not isinstance(raw_scheduler_config, dict):
        raise TypeError(
            "lr scheduler config metadata must be provided as a dict. "
            f"Got: {type(raw_scheduler_config)!r}"
        )
    return dict(raw_scheduler_config)


def get_optimizer_from_config(
    config: dict[str, Any] | None,
) -> tuple[OptimizerFactory | None, dict[str, Any] | None]:
    """Return an optimizer factory and kwargs parsed from config.

    Supported config shapes:

    ``None`` or ``{}``
        Returns ``(None, None)``.
    ``{"name": "AdamW", "params": {"lr": 1e-4}}``
        Returns ``(torch.optim.AdamW, {"lr": 1e-4})``.

    The helper also accepts ``type`` instead of ``name``, ``kwargs`` instead of
    ``params``, and dotted class paths such as ``torch.optim.AdamW``.
    """

    config_dict = _as_dict(config, label="optimizer")
    optimizer_name = config_dict.get("name", config_dict.get("type"))
    optimizer_kwargs = _extract_kwargs(config_dict, label="optimizer")

    if optimizer_name is None:
        if optimizer_kwargs:
            raise ValueError(
                "Optimizer parameters were provided without an optimizer name."
            )
        return None, None
    if not isinstance(optimizer_name, str):
        raise TypeError(
            "Optimizer name must be provided as a string. "
            f"Got: {type(optimizer_name)!r}"
        )

    registry_key = _normalize_registry_key(optimizer_name)
    optimizer_factory = _OPTIMIZER_REGISTRY.get(registry_key)
    if optimizer_factory is None:
        supported = ", ".join(
            sorted(factory.__name__ for factory in _OPTIMIZER_REGISTRY.values())
        )
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {supported}."
        )

    merged_kwargs = dict(_DEFAULT_OPTIMIZER_KWARGS.get(registry_key, {}))
    merged_kwargs.update(optimizer_kwargs)
    return optimizer_factory, merged_kwargs


def get_lr_scheduler_from_config(
    config: dict[str, Any] | None,
) -> tuple[
    LRSchedulerFactory | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Return an LR scheduler factory, kwargs, and Lightning metadata.

    Supported config shape:

    ``{"name": "ReduceLROnPlateau", "params": {...}, "config": {...}}``

    The nested ``config`` dict is forwarded to ``ModelABC`` as Lightning
    scheduler metadata, for example ``monitor``, ``interval``, or ``frequency``.
    ``type``, ``kwargs``, dotted class paths, and ``lightning_config`` /
    ``scheduler_config`` aliases are also accepted.
    """

    config_dict = _as_dict(config, label="lr scheduler")
    scheduler_name = config_dict.get("name", config_dict.get("type"))
    scheduler_kwargs = _extract_kwargs(config_dict, label="lr scheduler")
    scheduler_config = _extract_scheduler_config(config_dict)

    if scheduler_name is None:
        if scheduler_kwargs or scheduler_config:
            raise ValueError(
                "LR scheduler parameters or config were provided without a scheduler name."
            )
        return None, None, None
    if not isinstance(scheduler_name, str):
        raise TypeError(
            "LR scheduler name must be provided as a string. "
            f"Got: {type(scheduler_name)!r}"
        )

    registry_key = _normalize_registry_key(scheduler_name)
    scheduler_factory = _LR_SCHEDULER_REGISTRY.get(registry_key)
    if scheduler_factory is None:
        supported = ", ".join(
            sorted(factory.__name__ for factory in _LR_SCHEDULER_REGISTRY.values())
        )
        raise ValueError(
            f"Unsupported lr scheduler name: {scheduler_name}. Supported schedulers: {supported}."
        )

    return scheduler_factory, scheduler_kwargs, scheduler_config
