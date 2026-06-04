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
    "mil": frozenset({"mean", "max", "attention"}),
}

_AGGREGATOR_SUBTASK_BASES: frozenset[str] = frozenset({"clam"})

_AGGREGATOR_ATTENTION_VARIANTS: frozenset[str] = frozenset({"sb", "mb", "attention"})

_AGGREGATOR_SUPPORTED_ADD_ONS: frozenset[str] = frozenset({"gated"})

# Map each CLI subtask token to the full task name used inside the model and
# datamodule (e.g. ``params.subtasks=[...]`` entries, ``output_dims`` /
# ``task_weights`` keys, batch task keys).
_AGGREGATOR_SUBTASK_TASK_NAMES: dict[str, str] = {
    "sbs": "sbs_regression",
    "dbs": "dbs_regression",
    "id": "id_regression",
    "cnv": "cnv_regression",
}

_AGGREGATOR_SUPPORTED_SUBTASKS: frozenset[str] = frozenset(
    _AGGREGATOR_SUBTASK_TASK_NAMES
)


def load_aggregator_config(
    aggregator_dir: str | Path,
    *,
    base: str,
    variant: str,
    encoder: str,
    pretext: str,
    add_on: str | None = None,
    subtasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compose an aggregator config from partial YAMLs under ``aggregator_dir``.

    The aggregator YAML is split across six partial axes:

    - ``base-{base}.yaml``: bag-level architecture skeleton — ``clam`` or
      ``mil`` — including the per-task ``task_weights`` / ``task_kwargs``
      for the tasks that base uses. Each base ``extends:
      ../optimizers.yaml`` so the shared optimizer + LR-scheduler recipe
      is pulled in automatically.
    - ``subtask-{token}.yaml`` *(zero or more)*: auxiliary regression
      subtask layered onto the CLAM base. Supported tokens are ``sbs``,
      ``dbs``, ``id``, ``cnv`` (COSMIC signature classes). Pass ``full``
      as a shortcut for all four. Each selected partial adds an
      ``output_dims`` entry and a ``task_weights`` entry under the
      corresponding full task name (e.g. ``sbs_regression``). Multiple
      subtasks stack — DualCLAM grows one extra head per subtask. The
      input list is alphabetically canonicalized before use, so any
      permutation of the same set yields the same merged config + run
      name. Only valid with CLAM-family bases.
    - ``variant-{variant}.yaml``: variant within the base. For CLAM the
      choices are ``sb`` / ``mb``; for MIL they are ``mean``, ``max``,
      ``attention``.
    - ``add-on-{add_on}.yaml`` *(optional)*: attention add-on. Only
      ``gated`` is supported and only with attention-based variants
      (``sb``, ``mb``, ``attention``).
    - ``encoder-{encoder}.yaml``: encoder-dependent fields (``enc_dim``).
    - ``pretext-{pretext}.yaml``: marker for the encoder's pretext task;
      consumed only to compose ``checkpoint_path``.

    Partials are deep-merged in the order
    ``base → subtask[s] (left-to-right) → variant → add_on → encoder →
    pretext`` so that later partials override earlier ones (e.g.
    ``add-on-gated`` flips ``attn_kwargs.gated`` from ``false`` to
    ``true``). After the deep merge, ``params.subtasks`` is overwritten
    with the explicit full-task-name list derived from the requested
    ``subtasks`` tokens — so requesting ``["sbs", "dbs"]`` yields
    ``params.subtasks = ["sbs_regression", "dbs_regression"]`` regardless
    of how the underlying subtask partials happen to declare the list.

    The final ``checkpoint_path`` is set to
    ``checkpoints/{base}[-{subtask}]...-{variant}[-{add_on}]-{encoder}-{pretext}.pth``
    — the subtask tokens appear in alphabetical order between the base
    and the variant. When the full set is selected they collapse to a
    single ``full`` token (so ``--subtask sbs dbs id cnv`` and
    ``--subtask full`` produce the same filename).

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

    # Expand the convenience token ``full`` to the canonical set of
    # supported subtasks (mirrors how ``load_tile_model_config`` expands
    # ``pretexts=['full']``), then canonicalize order with an
    # alphabetical sort so any permutation of the same set yields the
    # same merged config + checkpoint name.
    raw_subtask_tokens: list[str] = list(subtasks or [])
    expanded_subtasks: list[str] = []
    for token in raw_subtask_tokens:
        if token == "full":
            expanded_subtasks.extend(_AGGREGATOR_SUPPORTED_SUBTASKS)
        else:
            expanded_subtasks.append(token)

    if expanded_subtasks and base not in _AGGREGATOR_SUBTASK_BASES:
        raise ValueError(
            f"subtasks={raw_subtask_tokens!r} only supported for "
            f"base ∈ {sorted(_AGGREGATOR_SUBTASK_BASES)}; "
            f"got base={base!r}."
        )
    unsupported = [
        s for s in expanded_subtasks if s not in _AGGREGATOR_SUPPORTED_SUBTASKS
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported aggregator subtask(s)={unsupported!r}. "
            f"Choose from {sorted(_AGGREGATOR_SUPPORTED_SUBTASKS)}, use "
            "'full' to select all four, or omit the flag."
        )
    if len(set(expanded_subtasks)) != len(expanded_subtasks):
        raise ValueError(
            f"Duplicate subtasks={raw_subtask_tokens!r}. Each subtask "
            "token may appear at most once (note 'full' already covers "
            "all four)."
        )
    subtask_list: list[str] = sorted(expanded_subtasks)

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

    aggregator_dir = Path(aggregator_dir)
    partial_paths: list[Path] = [aggregator_dir / f"base-{base}.yaml"]
    for subtask_token in subtask_list:
        partial_paths.append(aggregator_dir / f"subtask-{subtask_token}.yaml")
    partial_paths.append(aggregator_dir / f"variant-{variant}.yaml")
    if add_on is not None:
        partial_paths.append(aggregator_dir / f"add-on-{add_on}.yaml")
    partial_paths.append(aggregator_dir / f"encoder-{encoder}.yaml")
    partial_paths.append(aggregator_dir / f"pretext-{pretext}.yaml")

    merged: dict[str, Any] = {}
    for partial_path in partial_paths:
        if not partial_path.exists():
            raise FileNotFoundError(f"Aggregator partial not found: {partial_path}")
        merged = _deep_merge(merged, load_yaml_config(partial_path))

    # Authoritative source for ``params.subtasks`` is the CLI/programmatic
    # ``subtasks`` argument — the per-partial ``subtasks: [...]`` entries
    # otherwise get clobbered by ``_deep_merge``'s list-replacement
    # semantics when multiple subtask partials stack.
    if subtask_list:
        params = merged.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError("Merged aggregator config 'params' must be a dict.")
        params["subtasks"] = [
            _AGGREGATOR_SUBTASK_TASK_NAMES[token] for token in subtask_list
        ]

    # Collapse the full-set selection to a single ``full`` token in the
    # checkpoint filename so `--subtask sbs dbs id cnv` (in any order) and
    # `--subtask full` produce the same name. The Python-facing
    # ``params.subtasks`` list above keeps the explicit full task names so
    # the model still gets the per-task heads it needs.
    if subtask_list and set(subtask_list) == _AGGREGATOR_SUPPORTED_SUBTASKS:
        subtask_tokens = ["full"]
    else:
        subtask_tokens = subtask_list

    tokens = [base, *subtask_tokens, variant]
    if add_on is not None:
        tokens.append(add_on)
    tokens.extend([encoder, pretext])
    merged["checkpoint_path"] = f"checkpoints/{'-'.join(tokens)}.pth"

    return merged


# Map each CLI encoder token to the encoder's Python class name. Used by
# ``load_dataset_config`` to fill ``params.expected_encoder_name`` when the
# feature-flavor partial is selected — TCGAFeatureDataset uses this to assert
# the cached features were produced by the expected backbone.
_DATASET_ENCODER_CLASSES: dict[str, str] = {
    "resnet50": "ResNetEncoder",
    "prov-gigapath": "ViTEncoder",
}

# Supported dataset flavors. ``slide`` uses the base partial as-is
# (TCGASlideDataset); ``feature`` overlays ``flavor-feature.yaml`` for
# precomputed-feature training (TCGAFeatureDataset); ``tile`` overlays
# ``flavor-tile.yaml`` for tile-level multi-task pretraining
# (TCGATileDataset).
_DATASET_SUPPORTED_FLAVORS: frozenset[str] = frozenset({"slide", "feature", "tile"})

# Pretext tokens that translate into tile-multitask ``tasks`` entries when
# the ``tile`` flavor is selected. ``tissue_segmentation`` is always
# included as the supervised target. ``full`` / ``none`` are sugar.
_DATASET_TILE_TASK_NAMES: dict[str, str] = {
    "hematoxylin": "hematoxylin",
    "jigmag": "jigmag",
    "magnification": "magnification",
}


def load_dataset_config(
    dataset_dir: str | Path,
    *,
    base: str,
    flavor: str = "slide",
    main_task: str | None = None,
    subtasks: Sequence[str] | None = None,
    pretexts: Sequence[str] | None = None,
    encoder: str | None = None,
    pretext: str | None = None,
    features_dir: str | None = None,
) -> dict[str, Any]:
    """Compose a dataset config from partial YAMLs under ``dataset_dir``.

    Two partial axes:

    - ``base-{base}.yaml``: dataset-specific common params — ``root_dir``,
      tile-extraction params (``tile_size``, ``image_size``, ``base_mpp``,
      ``stride``, ``min_tissue_fraction``, ``thumbnail_max_size``,
      ``white_threshold``), train/val/test fractions, seed, and default
      batch / worker / prefetch settings. Slide-flavored by default
      (``TCGASlideDataset``).
    - ``flavor-{flavor}.yaml`` *(optional)*: overlay applied on top of
      the base. Three flavors are supported:

      * ``"slide"`` (default) — no overlay; base is used directly.
        Required args: ``main_task``. Optional: ``subtasks``.
      * ``"feature"`` — overlay ``flavor-feature.yaml``. Switches the
        datamodule to ``TCGAFeatureDataset`` and bumps batch sizes /
        worker counts. ``params.features_dir`` is set from
        ``features_dir`` (CLI) or
        ``{root_dir}/features/{encoder}-{pretext}``; the slide-only
        tile-extraction fields remain in the merged dict but are
        unread by the feature datamodule. Required args: ``main_task``;
        plus either ``features_dir`` or both ``encoder`` and ``pretext``.
        Optional: ``subtasks``.
      * ``"tile"`` — overlay ``flavor-tile.yaml``. Switches to
        ``TCGATileDataset`` for tile-level multi-task pretraining.
        ``params.tasks`` is computed as
        ``["tissue_segmentation", *expanded_pretexts]`` from the
        ``pretexts`` argument (``"full"`` expands to all three;
        ``"none"`` to none). ``main_task`` / ``subtasks`` are ignored
        because the tile datamodule has no main task / subtask
        distinction.

    Returns a dict whose hierarchical structure matches the legacy
    single-file dataset configs (``name`` + ``params``), ready to feed
    :func:`augur.datasets.factory.get_dataset_from_config`.
    """
    if not isinstance(base, str) or not base:
        raise ValueError("dataset base must be a non-empty string.")

    if flavor not in _DATASET_SUPPORTED_FLAVORS:
        raise ValueError(
            f"Unsupported dataset flavor={flavor!r}. "
            f"Choose one of {sorted(_DATASET_SUPPORTED_FLAVORS)}."
        )

    subtask_list: list[str] = list(subtasks or [])
    if any(not isinstance(t, str) or not t for t in subtask_list):
        raise ValueError("subtasks must contain only non-empty strings.")

    dataset_dir = Path(dataset_dir)
    partial_paths: list[Path] = [dataset_dir / f"base-{base}.yaml"]
    if flavor != "slide":
        partial_paths.append(dataset_dir / f"flavor-{flavor}.yaml")

    merged: dict[str, Any] = {}
    for partial_path in partial_paths:
        if not partial_path.exists():
            raise FileNotFoundError(f"Dataset partial not found: {partial_path}")
        merged = _deep_merge(merged, load_yaml_config(partial_path))

    params = merged.setdefault("params", {})
    if not isinstance(params, dict):
        raise TypeError("Merged dataset config 'params' must be a dict.")

    if flavor in {"slide", "feature"}:
        if not isinstance(main_task, str) or not main_task:
            raise ValueError(f"flavor={flavor!r} requires a non-empty 'main_task'.")
        params["main_task"] = main_task
        if subtask_list:
            params["subtasks"] = list(subtask_list)
        else:
            params.pop("subtasks", None)

    if flavor == "feature":
        if features_dir is None:
            if encoder is None or pretext is None:
                raise ValueError(
                    "flavor='feature' requires either an explicit features_dir "
                    "or both encoder and pretext to derive "
                    "<root_dir>/features/<encoder>-<pretext>."
                )
            root_dir = params.get("root_dir")
            if not isinstance(root_dir, str) or not root_dir:
                raise ValueError(
                    "Dataset base partial must declare a non-empty "
                    "'params.root_dir' before features_dir can be derived."
                )
            features_dir = f"{root_dir.rstrip('/')}/features/{encoder}-{pretext}"
        params["features_dir"] = features_dir

        if encoder is not None and "expected_encoder_name" not in params:
            encoder_class = _DATASET_ENCODER_CLASSES.get(encoder)
            if encoder_class is not None:
                params["expected_encoder_name"] = encoder_class

    if flavor == "tile":
        # Strip slide-only fields that TCGATileDataset.from_config would
        # otherwise ignore but that are misleading in the merged dict.
        for slide_only_key in ("portion_per_sample", "stride"):
            params.pop(slide_only_key, None)
        # ``main_task`` and ``subtasks`` don't apply to TCGATileDataset; ignore
        # them rather than smuggling them into params.
        params.pop("main_task", None)
        params.pop("subtasks", None)

        pretext_tokens: list[str] = list(pretexts or [])
        expanded: list[str] = []
        for token in pretext_tokens:
            if token == "full":
                expanded.extend(["hematoxylin", "jigmag", "magnification"])
            elif token == "none":
                continue
            else:
                expanded.append(token)
        unsupported = [p for p in expanded if p not in _DATASET_TILE_TASK_NAMES]
        if unsupported:
            raise ValueError(
                f"Unsupported tile pretext(s)={unsupported!r}. "
                f"Choose from {sorted(_DATASET_TILE_TASK_NAMES)} or the special "
                "tokens 'full' / 'none'."
            )
        if len(set(expanded)) != len(expanded):
            raise ValueError(
                f"Duplicate tile pretexts={expanded!r}. Each pretext may "
                "appear at most once (note 'full' already covers all three)."
            )
        params["tasks"] = [
            "tissue_segmentation",
            *(_DATASET_TILE_TASK_NAMES[p] for p in expanded),
        ]

    return merged


_TILE_MODEL_SUPPORTED_ENCODERS: frozenset[str] = frozenset(
    {"resnet50", "prov-gigapath"}
)

# Pretext tokens that can be stacked onto a tile model. ``full`` is sugar for
# "all three"; ``none`` is sugar for the empty selection. Both pass through
# ``load_tile_model_config(pretexts=...)`` after expansion.
_TILE_MODEL_PRETEXTS: frozenset[str] = frozenset(
    {"hematoxylin", "jigmag", "magnification"}
)


def load_tile_model_config(
    tile_model_dir: str | Path,
    *,
    encoder: str,
    pretexts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compose a tile-model config from partial YAMLs under ``tile_model_dir``.

    Two partial axes:

    - ``base-{encoder}.yaml``: per-encoder skeleton — declares the
      ``params.encoder_config``, the always-on ``tissue_segmentation``
      entry under ``params.decoders_config``, and its
      ``task_weights`` / ``task_kwargs``. Supported encoders:
      ``resnet50``, ``prov-gigapath``.
    - ``pretext-{pretext}-{encoder}.yaml`` *(zero or more)*: each
      stacked pretext adds one decoder under
      ``params.decoders_config`` and one entry under
      ``params.task_weights``. Supported pretexts are
      ``hematoxylin``, ``jigmag``, ``magnification`` (the encoder
      pretexts the tile multi-task model can predict).

    Two convenience tokens are recognized in ``pretexts`` and expand
    before validation:

    - ``"full"`` → ``["hematoxylin", "jigmag", "magnification"]``.
    - ``"none"`` → ``[]``.

    They can be mixed with explicit pretexts (e.g. ``["full"]`` ≡ all
    three; ``["hematoxylin", "jigmag"]`` ≡ two specific pretexts). The
    final per-pretext partial filename is encoder-specific because the
    decoder class differs (UNet/Classifier-maxpool for ResNet vs
    DPT/Classifier-identity for ViT).

    Partials are deep-merged in the order
    ``base → pretext_1 → pretext_2 → ...``, so per-pretext entries
    accumulate under ``params.decoders_config`` and
    ``params.task_weights``. The final ``checkpoint_path`` is set to
    ``checkpoints/{encoder}-{pretext_tag}.pth`` where
    ``pretext_tag`` is ``"none"`` (no pretexts), ``"full"`` (all three
    in canonical order), or the requested pretexts joined by ``-``
    (preserving CLI order).
    """
    if encoder not in _TILE_MODEL_SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported tile-model encoder={encoder!r}. "
            f"Choose one of {sorted(_TILE_MODEL_SUPPORTED_ENCODERS)}."
        )

    pretext_tokens: list[str] = list(pretexts or [])
    expanded: list[str] = []
    for token in pretext_tokens:
        if token == "full":
            expanded.extend(["hematoxylin", "jigmag", "magnification"])
        elif token == "none":
            continue
        else:
            expanded.append(token)

    unsupported = [p for p in expanded if p not in _TILE_MODEL_PRETEXTS]
    if unsupported:
        raise ValueError(
            f"Unsupported tile-model pretext(s)={unsupported!r}. "
            f"Choose from {sorted(_TILE_MODEL_PRETEXTS)} or the special "
            "tokens 'full' / 'none'."
        )
    if len(set(expanded)) != len(expanded):
        raise ValueError(
            f"Duplicate tile-model pretexts={expanded!r}. Each pretext may "
            "appear at most once (note 'full' already covers all three)."
        )

    tile_model_dir = Path(tile_model_dir)
    partial_paths: list[Path] = [tile_model_dir / f"base-{encoder}.yaml"]
    for pretext in expanded:
        partial_paths.append(tile_model_dir / f"pretext-{pretext}-{encoder}.yaml")

    merged: dict[str, Any] = {}
    for partial_path in partial_paths:
        if not partial_path.exists():
            raise FileNotFoundError(f"Tile-model partial not found: {partial_path}")
        merged = _deep_merge(merged, load_yaml_config(partial_path))

    # Pretext tag for checkpoint naming. Use 'none' / 'full' for the
    # canonical extremes and the CLI-order list otherwise.
    if not expanded:
        pretext_tag = "none"
    elif set(expanded) == _TILE_MODEL_PRETEXTS:
        pretext_tag = "full"
    else:
        pretext_tag = "-".join(expanded)
    merged["checkpoint_path"] = f"checkpoints/{encoder}-{pretext_tag}.pth"

    return merged


def load_trainer_config(
    trainer_dir: str | Path,
    *,
    trainer: str = "default",
) -> dict[str, Any]:
    """Load a trainer recipe partial from ``trainer_dir``.

    Single-axis partial: ``base-{trainer}.yaml`` holds the Lightning
    ``trainer``/``checkpoint``/``logger`` block (plus top-level
    ``seed`` / ``test_after_fit`` / ``enable_lr_monitor``). The run-name
    -derived fields (``checkpoint.filename``, ``logger.name``) are left
    unset here — the training script fills them from the composed run
    name so a single trainer partial fits every model/dataset
    combination.

    ``trainer`` selects which recipe to load (``default`` for production
    GPU/DDP, ``test`` for fast-dev). Returns the loaded dict unchanged.
    """
    trainer_dir = Path(trainer_dir)
    partial_path = trainer_dir / f"base-{trainer}.yaml"
    if not partial_path.exists():
        raise FileNotFoundError(f"Trainer partial not found: {partial_path}")
    return load_yaml_config(partial_path)
