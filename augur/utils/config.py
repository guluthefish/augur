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
