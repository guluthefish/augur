"""Utilities for loading and composing YAML configuration files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Sequence

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dict objects."""
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(dict(base_value), value)
        else:
            merged[key] = value
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML file and ensure it contains a dict at the top level."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with path.open("r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=loader)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dict at the top level: {path}")
    return dict(data)


def _normalize_extends(
    extends: str | Sequence[str] | None,
    *,
    path: Path,
) -> list[Path]:
    """Normalize ``extends`` entries to resolved paths."""
    if extends is None:
        return []
    if isinstance(extends, str):
        return [(path.parent / extends).resolve()]
    if isinstance(extends, Sequence) and all(isinstance(item, str) for item in extends):
        return [(path.parent / item).resolve() for item in extends]

    raise TypeError(
        f"'extends' in {path} must be a string or a sequence of strings. Got: {extends!r}"
    )


def load_yaml_config(
    path: str | Path,
    *,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a YAML config file with optional recursive ``extends`` support.

    Example
    -------
    ``model.yaml``:

    ```yaml
    extends: common-params.yaml
    name: ViTEncoder
    params:
      model_name: hf_hub:prov-gigapath/prov-gigapath
      pretrained: true
    ```
    """

    resolved_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if resolved_path in seen:
        raise ValueError(f"Circular config inheritance detected at: {resolved_path}")
    seen.add(resolved_path)

    config = _load_yaml_dict(resolved_path)
    extends = config.pop("extends", config.pop("base", None))

    merged: dict[str, Any] = {}
    for base_path in _normalize_extends(extends, path=resolved_path):
        merged = _deep_merge(
            merged,
            load_yaml_config(base_path, _seen=seen),
        )

    return _deep_merge(merged, config)


_AGGREGATOR_VARIANTS_BY_BASE: dict[str, frozenset[str]] = {
    "clam": frozenset({"sb", "mb"}),
    "dual-clam": frozenset({"sb", "mb"}),
    "mil": frozenset({"mean", "max", "attention"}),
}

_AGGREGATOR_ATTENTION_VARIANTS: frozenset[str] = frozenset({"sb", "mb", "attention"})

_AGGREGATOR_SUPPORTED_ADD_ONS: frozenset[str] = frozenset({"gated"})

_AGGREGATOR_SUPPORTED_OPTIMIZERS: frozenset[str] = frozenset({"adamw"})

_AGGREGATOR_SUPPORTED_LR_SCHEDULERS: frozenset[str] = frozenset({"cosine"})


def load_aggregator_config(
    aggregator_dir: str | Path,
    *,
    base: str,
    variant: str,
    encoder: str,
    pretext: str,
    add_on: str | None = None,
    optimizer: str = "adamw",
    lr_scheduler: str = "cosine",
) -> dict[str, Any]:
    """Compose an aggregator config from partial YAMLs under ``aggregator_dir``.

    The aggregator YAML is split across seven partial axes:

    - ``optimizer-{optimizer}.yaml``: optimizer recipe (``params.optimizer``).
    - ``lr-scheduler-{lr_scheduler}.yaml``: LR schedule (``params.lr_scheduler``).
    - ``base-{base}.yaml``: bag-level architecture skeleton — ``clam``,
      ``dual-clam``, or ``mil`` — including the per-task
      ``task_weights`` / ``task_kwargs`` for the tasks that base uses.
    - ``variant-{variant}.yaml``: variant within the base. For CLAM-family
      bases the choices are ``sb`` / ``mb``; for MIL they are ``mean``,
      ``max``, ``attention``.
    - ``add-on-{add_on}.yaml`` *(optional)*: attention add-on. Only
      ``gated`` is supported and only with attention-based variants
      (``sb``, ``mb``, ``attention``).
    - ``encoder-{encoder}.yaml``: encoder-dependent fields (``enc_dim``).
    - ``pretext-{pretext}.yaml``: marker for the encoder's pretext task;
      consumed only to compose ``checkpoint_path``.

    Partials are deep-merged in the order
    ``optimizer → lr_scheduler → base → variant → add_on → encoder →
    pretext`` so that later partials override earlier ones (e.g.
    ``add-on-gated`` flips ``attn_kwargs.gated`` from ``false`` to
    ``true``). The final ``checkpoint_path`` is set to
    ``checkpoints/{base}-{variant}[-{add_on}]-{encoder}-{pretext}.pth`` —
    the optimizer / LR-scheduler choices do not appear in the checkpoint
    filename because they don't affect the saved weight structure.

    Returns a dict whose hierarchical structure matches the legacy
    single-file aggregator configs (``name``, ``params``,
    ``checkpoint_path``).
    """
    if base not in _AGGREGATOR_VARIANTS_BY_BASE:
        raise ValueError(
            f"Unsupported aggregator base={base!r}. "
            f"Choose one of {sorted(_AGGREGATOR_VARIANTS_BY_BASE)}."
        )

    valid_variants = _AGGREGATOR_VARIANTS_BY_BASE[base]
    if variant not in valid_variants:
        raise ValueError(
            f"variant={variant!r} is not supported for base={base!r}. "
            f"Choose one of {sorted(valid_variants)}."
        )

    if add_on is not None:
        if add_on not in _AGGREGATOR_SUPPORTED_ADD_ONS:
            raise ValueError(
                f"Unsupported aggregator add_on={add_on!r}. "
                f"Choose one of {sorted(_AGGREGATOR_SUPPORTED_ADD_ONS)} or "
                "omit it."
            )
        if variant not in _AGGREGATOR_ATTENTION_VARIANTS:
            raise ValueError(
                f"add_on={add_on!r} requires an attention-based variant; "
                f"got variant={variant!r}."
            )

    if optimizer not in _AGGREGATOR_SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported aggregator optimizer={optimizer!r}. "
            f"Choose one of {sorted(_AGGREGATOR_SUPPORTED_OPTIMIZERS)}."
        )

    if lr_scheduler not in _AGGREGATOR_SUPPORTED_LR_SCHEDULERS:
        raise ValueError(
            f"Unsupported aggregator lr_scheduler={lr_scheduler!r}. "
            f"Choose one of {sorted(_AGGREGATOR_SUPPORTED_LR_SCHEDULERS)}."
        )

    aggregator_dir = Path(aggregator_dir)
    partial_paths: list[Path] = [
        aggregator_dir / f"optimizer-{optimizer}.yaml",
        aggregator_dir / f"lr-scheduler-{lr_scheduler}.yaml",
        aggregator_dir / f"base-{base}.yaml",
        aggregator_dir / f"variant-{variant}.yaml",
    ]
    if add_on is not None:
        partial_paths.append(aggregator_dir / f"add-on-{add_on}.yaml")
    partial_paths.append(aggregator_dir / f"encoder-{encoder}.yaml")
    partial_paths.append(aggregator_dir / f"pretext-{pretext}.yaml")

    merged: dict[str, Any] = {}
    for partial_path in partial_paths:
        if not partial_path.exists():
            raise FileNotFoundError(
                f"Aggregator partial not found: {partial_path}"
            )
        merged = _deep_merge(merged, load_yaml_config(partial_path))

    tokens = [base, variant]
    if add_on is not None:
        tokens.append(add_on)
    tokens.extend([encoder, pretext])
    merged["checkpoint_path"] = f"checkpoints/{'-'.join(tokens)}.pth"

    return merged
