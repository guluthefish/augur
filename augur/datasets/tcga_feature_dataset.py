"""Lightning datamodule that yields cached encoder features per slide.

Drop-in replacement for :class:`TCGASlideDataset` when training a slide-level
aggregator on top of a *frozen* tile encoder. Instead of opening WSIs and
re-encoding tiles every epoch, ``__getitem__`` loads a per-slide
``<slide_id>.pt`` produced by
``scripts/model_training/precompute_tile_features.py`` and returns a
``(K, D)`` feature bag with the same downstream contract as ``_SlideDataset``
— including ``target`` and per-subtask target entries.

The same :func:`pad_bag_collate` from the slide dataset works unchanged: it
pads along the first axis regardless of trailing shape. After collation,
``batch["image"]`` is ``(B, K_max, D)`` — exactly the pre-encoded path already
handled by ``DualCLAM._encode_bag`` / ``EmbeddingMIL._encode_bag`` via the
``image.ndim == 3`` branch.

Compared to the slide dataset, this datamodule:

- Skips slides that have no cached ``<slide_id>.pt`` (with a warning), so a
  partial cache build still trains.
- Caps each sampled bag at ``max_tiles_per_bag`` after applying
  ``portion_per_sample`` — the OOM mitigation that was unavailable when the
  encoder was inline in the training loop.
- Validates that the cache agrees on ``encoder_name`` and ``enc_dim``;
  mixing caches from different encoders would otherwise silently produce
  garbage features.
"""

from __future__ import annotations

import logging
import os
import random
import time
import traceback
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from augur.datasets.cancer_subtyping import load_subtyping_labels
from augur.datasets.dataset_abc import DatasetABC
from augur.datasets.mutational_signature import (
    SUPPORTED_SUBTASKS,
    load_signature_labels,
)
from augur.datasets.tcga_slide_dataset import (
    SUPPORTED_MAIN_TASKS,
    pad_bag_collate,
)
from augur.datasets.utils import (
    SlideRecord,
    load_slide_records,
    resolve_manifest_path,
    resolve_slide_main_label_path,
    resolve_slide_subtask_label_path,
    split_slide_records,
)
from augur.utils.logger import setup_logger


def _setup_logger_for_module() -> logging.Logger:
    """Create the dataset logger shared by the datamodule and worker loaders."""
    log_dir = os.path.join("logs", "datasets")
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="TCGAFeatureDataset", rank_zero_only=True)


# ----------------------------------------------------------------------------
# Inner Dataset
# ----------------------------------------------------------------------------


class _FeatureBagDataset(Dataset[dict[str, Any]]):
    """Map-style dataset that loads one cached feature bag per slide.

    Each entry yields::

        {
            "image":     Tensor[K, D] float32,   # sampled tile features
            "target":    Tensor[scalar long],    # main-task class index
            "metadata":  dict[str, Any],
            "<subtask>": {"target": Tensor[D'] float},  # one per task
        }

    where ``K = min(max_tiles_per_bag, max(1, floor(K_total * portion_per_sample)))``.
    """

    _GETITEM_MAX_RETRIES = 3

    def __init__(
        self: _FeatureBagDataset,
        *,
        slide_records: Sequence[SlideRecord],
        feature_paths_by_slide_id: dict[str, str],
        main_submitter_labels: dict[str, int],
        main_label_names: Sequence[str],
        subtasks: list[str] | None = None,
        subtask_submitter_labels: dict[str, dict[str, np.ndarray]] | None = None,
        subtask_label_names: dict[str, tuple[str, ...]] | None = None,
        portion_per_sample: float = 1.0,
        max_tiles_per_bag: int | None = None,
        task: str = "subtyping",
        logger: logging.Logger | None = None,
    ) -> None:
        if not 0.0 < float(portion_per_sample) <= 1.0:
            raise ValueError(
                f"portion_per_sample must be in (0, 1]. Got: {portion_per_sample}"
            )
        if max_tiles_per_bag is not None and (
            not isinstance(max_tiles_per_bag, int) or max_tiles_per_bag <= 0
        ):
            raise ValueError(
                "max_tiles_per_bag must be a positive integer or None. "
                f"Got: {max_tiles_per_bag}"
            )

        self.slide_records = list(slide_records)
        self.feature_paths_by_slide_id = dict(feature_paths_by_slide_id)
        self.main_submitter_labels = main_submitter_labels
        self.main_label_names = tuple(main_label_names)
        self.subtasks: list[str] = list(subtasks or [])
        self.subtask_submitter_labels: dict[str, dict[str, np.ndarray]] = (
            subtask_submitter_labels or {}
        )
        self.subtask_label_names: dict[str, tuple[str, ...]] = dict(
            subtask_label_names or {}
        )
        self.portion_per_sample = float(portion_per_sample)
        self.max_tiles_per_bag = max_tiles_per_bag
        self.task = task
        self.logger = logger or _setup_logger_for_module()

    def __len__(self: _FeatureBagDataset) -> int:
        return len(self.slide_records)

    def _sampled_size(self: _FeatureBagDataset, k_total: int) -> int:
        """How many tiles to sample from a slide with ``k_total`` cached features."""
        if k_total <= 0:
            return 0
        size = max(1, int(k_total * self.portion_per_sample))
        if self.max_tiles_per_bag is not None:
            size = min(size, self.max_tiles_per_bag)
        return min(size, k_total)

    def __getitem__(self: _FeatureBagDataset, index: int) -> dict[str, Any]:
        worker_pid = os.getpid()
        current_index = index
        for attempt in range(self._GETITEM_MAX_RETRIES + 1):
            slide_record = self.slide_records[current_index]
            self.logger.info(
                "[pid=%d] __getitem__ start: index=%d attempt=%d slide_id=%s",
                worker_pid,
                current_index,
                attempt,
                slide_record.slide_id,
            )
            start_time = time.monotonic()
            try:
                sample = self._load_sample(slide_record)
            except Exception as exc:  # pylint: disable=broad-except
                elapsed = time.monotonic() - start_time
                self.logger.error(
                    "[pid=%d] __getitem__ FAILED: index=%d attempt=%d "
                    "slide_id=%s elapsed=%.2fs error=%s\n%s",
                    worker_pid,
                    current_index,
                    attempt,
                    slide_record.slide_id,
                    elapsed,
                    exc,
                    traceback.format_exc(),
                )
                if attempt >= self._GETITEM_MAX_RETRIES:
                    raise RuntimeError(
                        f"_FeatureBagDataset.__getitem__ failed after "
                        f"{self._GETITEM_MAX_RETRIES + 1} attempts starting at "
                        f"index {index}; last failure on slide "
                        f"{slide_record.slide_id}."
                    ) from exc
                current_index = random.randrange(len(self.slide_records))
                continue

            elapsed = time.monotonic() - start_time
            self.logger.info(
                "[pid=%d] __getitem__ done: index=%d slide_id=%s K=%d elapsed=%.3fs",
                worker_pid,
                current_index,
                slide_record.slide_id,
                int(sample["image"].shape[0]),
                elapsed,
            )
            return sample

        raise RuntimeError("unreachable in _FeatureBagDataset.__getitem__")

    def _load_sample(
        self: _FeatureBagDataset, slide_record: SlideRecord
    ) -> dict[str, Any]:
        """Load one cached bag and assemble its sample dict."""
        cache_path = self.feature_paths_by_slide_id[slide_record.slide_id]
        # weights_only=False because the payload is a dict with str + Tensor values.
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)

        features = payload.get("features")
        if not isinstance(features, Tensor) or features.ndim != 2:
            raise ValueError(
                f"Expected 'features' tensor of shape (K, D) in {cache_path}. "
                f"Got: {type(features)!r} ndim="
                f"{getattr(features, 'ndim', '?')}"
            )
        if features.dtype != torch.float32:
            features = features.to(torch.float32)

        tile_centers = payload.get("tile_centers")
        if not isinstance(tile_centers, Tensor) or tile_centers.ndim != 2:
            raise ValueError(
                f"Expected 'tile_centers' tensor of shape (K, 2) in {cache_path}."
            )

        k_total = int(features.shape[0])
        sample_size = self._sampled_size(k_total)
        if sample_size == 0:
            raise RuntimeError(
                f"Cached slide {slide_record.slide_id} has zero features."
            )
        if sample_size < k_total:
            indices = random.sample(range(k_total), sample_size)
        else:
            indices = list(range(k_total))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        sampled_features = features.index_select(0, index_tensor).contiguous()
        sampled_centers = tile_centers.index_select(0, index_tensor).contiguous()

        target = torch.tensor(
            self.main_submitter_labels[slide_record.submitter_id],
            dtype=torch.long,
        )

        metadata: dict[str, Any] = {
            "slide_id": slide_record.slide_id,
            "submitter_id": slide_record.submitter_id,
            "task": self.task,
            "encoder_name": payload.get("encoder_name"),
            "enc_dim": int(features.shape[1]),
            "base_mpp": float(payload.get("base_mpp", 0.0)),
            "tile_size": int(payload.get("tile_size", 0)),
            "image_size": int(payload.get("image_size", 0)),
            "tile_centers": sampled_centers,          # (K, 2) long
            "selected_indices": index_tensor,         # (K,)  long
            "k_total": int(k_total),
        }

        sample: dict[str, Any] = {
            "image": sampled_features,
            "target": target,
            "metadata": metadata,
        }

        for subtask in self.subtasks:
            task_submitter_labels = self.subtask_submitter_labels.get(subtask)
            if task_submitter_labels is None:
                raise RuntimeError(
                    f"Subtask task '{subtask}' is missing label mappings."
                )
            subtask_vector = task_submitter_labels[slide_record.submitter_id]
            sample[subtask] = {
                "target": torch.from_numpy(
                    np.ascontiguousarray(subtask_vector)
                ).float()
            }

        return sample


# ----------------------------------------------------------------------------
# LightningDataModule
# ----------------------------------------------------------------------------


class TCGAFeatureDataset(DatasetABC):
    """Lightning datamodule that serves cached encoder features per slide.

    Mirrors :class:`TCGASlideDataset` for label loading and patient-level
    splitting, but each ``__getitem__`` loads a precomputed ``(K, D)`` feature
    tensor from ``features_dir/<slide_id>.pt`` instead of opening the WSI.

    Expected cache layout (produced by
    ``scripts/model_training/precompute_tile_features.py``)::

        <features_dir>/
            <slide_id>.pt        # dict with 'features', 'tile_centers', ...
            _manifest.tsv        # produced by the precompute script (ignored)

    Each ``.pt`` payload is a dict with at minimum ``features``
    (Tensor[K, D]) and ``tile_centers`` (Tensor[K, 2] long).

    Parameters
    ----------
    root_dir
        Root TCGA directory (same one used by ``TCGASlideDataset``). Used to
        resolve the manifest and label tables.
    features_dir
        Directory containing per-slide ``<slide_id>.pt`` feature caches.
    main_task, subtasks
        Slide-level main task (currently ``"subtyping"``) and optional list of
        subtasks.
    manifest_path, main_labels_path, subtask_labels_paths, ordered_data_dir
        Optional explicit overrides; otherwise resolved from ``root_dir``.
    portion_per_sample
        Fraction of cached tiles to sample per slide, in ``(0, 1]``.
    max_tiles_per_bag
        Optional hard cap on bag size after applying ``portion_per_sample``.
    train_fraction, val_fraction, test_fraction, random_seed, max_slides
        Patient-level split controls, identical to :class:`TCGASlideDataset`.
    enc_dim
        Optional declared feature dim; if provided, validated against the
        cache on setup. If omitted, inferred from the first cache file.
    expected_encoder_name
        Optional declared encoder class name; if provided, refuses caches
        from a different encoder.
    logger
        Optional logger override.
    batch_size, val_batch_size, test_batch_size, predict_batch_size,
    num_workers, pin_memory, persistent_workers, prefetch_factor,
    shuffle_train, drop_last_train, collate_fn
        Standard Lightning datamodule wiring.
    """

    SUPPORTED_MAIN_TASKS = SUPPORTED_MAIN_TASKS
    SUPPORTED_SUBTASKS = SUPPORTED_SUBTASKS

    def __init__(
        self: TCGAFeatureDataset,
        root_dir: str,
        features_dir: str,
        *,
        main_task: str = "subtyping",
        subtasks: list[str] | None = None,
        manifest_path: str | None = None,
        main_labels_path: str | None = None,
        subtask_labels_paths: dict[str, str] | None = None,
        ordered_data_dir: str | None = None,
        portion_per_sample: float = 1.0,
        max_tiles_per_bag: int | None = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        random_seed: int = 42,
        max_slides: int | None = None,
        enc_dim: int | None = None,
        expected_encoder_name: str | None = None,
        logger: logging.Logger | None = None,
        batch_size: int = 8,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        predict_batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            predict_batch_size=predict_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            shuffle_train=shuffle_train,
            drop_last_train=drop_last_train,
            collate_fn=collate_fn if collate_fn is not None else pad_bag_collate,
        )

        self.logger = logger or _setup_logger_for_module()

        if main_task not in self.SUPPORTED_MAIN_TASKS:
            raise ValueError(
                f"Unsupported main_task: {main_task!r}. "
                f"Must be one of {self.SUPPORTED_MAIN_TASKS}."
            )
        if subtasks is not None:
            if not isinstance(subtasks, list) or any(
                not isinstance(task, str) or not task for task in subtasks
            ):
                raise ValueError(
                    "subtasks must be a list of non-empty strings or None."
                )
            if len(set(subtasks)) != len(subtasks):
                raise ValueError("subtasks must not contain duplicates.")
            for task in subtasks:
                if task not in self.SUPPORTED_SUBTASKS:
                    raise ValueError(
                        f"Unsupported subtask: {task!r}. "
                        f"Must be one of {self.SUPPORTED_SUBTASKS}."
                    )

        if not isinstance(features_dir, str) or not features_dir:
            raise ValueError(
                "features_dir is required and must be a non-empty string."
            )
        if not os.path.isdir(features_dir):
            raise FileNotFoundError(
                f"features_dir does not exist or is not a directory: {features_dir}"
            )
        if not 0.0 < float(portion_per_sample) <= 1.0:
            raise ValueError(
                f"portion_per_sample must be in (0, 1]. Got: {portion_per_sample}"
            )
        if max_tiles_per_bag is not None and (
            not isinstance(max_tiles_per_bag, int) or max_tiles_per_bag <= 0
        ):
            raise ValueError(
                "max_tiles_per_bag must be a positive integer or None. "
                f"Got: {max_tiles_per_bag}"
            )
        if enc_dim is not None and (not isinstance(enc_dim, int) or enc_dim <= 0):
            raise ValueError(
                f"enc_dim must be a positive integer or None. Got: {enc_dim}"
            )
        if max_slides is not None and (
            not isinstance(max_slides, int) or max_slides <= 0
        ):
            raise ValueError("max_slides must be a positive integer or None.")
        split_sum = train_fraction + val_fraction + test_fraction
        if not np.isclose(split_sum, 1.0):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1."
            )

        self.root_dir = root_dir
        self.features_dir = os.path.abspath(features_dir)
        self.main_task = main_task
        self.subtasks: list[str] = list(subtasks or [])
        self.manifest_path = manifest_path
        self.main_labels_path = main_labels_path
        self.subtask_labels_paths: dict[str, str] = dict(subtask_labels_paths or {})
        self.ordered_data_dir = (
            ordered_data_dir
            if ordered_data_dir is not None
            else os.path.join(root_dir, "ordered_data")
        )
        self.portion_per_sample = float(portion_per_sample)
        self.max_tiles_per_bag = max_tiles_per_bag
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.random_seed = int(random_seed)
        self.max_slides = max_slides
        self._declared_enc_dim = enc_dim
        self._declared_encoder_name = expected_encoder_name

        # Resolved-on-setup state, mirroring TCGASlideDataset.
        self._resolved_manifest_path: str | None = None
        self._resolved_main_labels_path: str | None = None
        self._resolved_subtask_labels_paths: dict[str, str] = {}
        self._main_submitter_labels: dict[str, int] | None = None
        self._main_label_names: tuple[str, ...] | None = None
        self._subtask_submitter_labels: dict[str, dict[str, np.ndarray]] = {}
        self._subtask_label_names: dict[str, tuple[str, ...]] = {}
        self._slide_splits: dict[str, list[SlideRecord]] | None = None
        self._feature_paths_by_slide_id: dict[str, str] | None = None

        # Discovered on first cache load (or validated against declared).
        self._inferred_enc_dim: int | None = None
        self._inferred_encoder_name: str | None = None

    # -- properties exposed to downstream code -------------------------------

    @property
    def enc_dim(self: TCGAFeatureDataset) -> int | None:
        """Feature dim discovered from the cache (or declared via config)."""
        return self._inferred_enc_dim or self._declared_enc_dim

    @property
    def encoder_name(self: TCGAFeatureDataset) -> str | None:
        """Encoder class name discovered from the cache (or declared)."""
        return self._inferred_encoder_name or self._declared_encoder_name

    @property
    def num_main_labels(self: TCGAFeatureDataset) -> int:
        """Number of main-task classes (after labels are loaded)."""
        if self._main_label_names is None:
            raise RuntimeError(
                "Call prepare_data()/setup() before reading num_main_labels."
            )
        return len(self._main_label_names)

    @property
    def num_subtask_labels(self: TCGAFeatureDataset) -> dict[str, int]:
        """Dim of each subtask label vector (after labels are loaded)."""
        return {task: len(names) for task, names in self._subtask_label_names.items()}

    # -- LightningDataModule API ---------------------------------------------

    def prepare_data(self: TCGAFeatureDataset) -> None:
        """Resolve manifest and label paths on one process."""
        self.logger.info(
            "Preparing TCGA feature dataset for main task: %s", self.main_task
        )
        self._resolved_manifest_path = resolve_manifest_path(
            self.root_dir, self.manifest_path, self.logger
        )
        self._resolved_main_labels_path = resolve_slide_main_label_path(
            self.root_dir, self.main_task, self.main_labels_path, self.logger
        )
        for subtask in self.subtasks:
            self._resolved_subtask_labels_paths[subtask] = (
                resolve_slide_subtask_label_path(
                    self.root_dir,
                    subtask,
                    self.subtask_labels_paths.get(subtask),
                    self.logger,
                )
            )

    def setup(self: TCGAFeatureDataset, stage: str | None = None) -> None:
        """Load labels, discover the feature cache, and build split datasets."""
        valid_stages = {None, "fit", "validate", "test", "predict"}
        if stage not in valid_stages:
            raise ValueError(f"Unsupported stage: {stage}")

        self.logger.info(
            "Starting setup(stage=%s) for main task: %s", stage, self.main_task
        )

        self._ensure_main_labels_loaded()
        self._ensure_subtask_labels_loaded()
        slide_splits = self._get_slide_splits()
        feature_paths_by_slide_id = self._discover_feature_cache(slide_splits)
        self._validate_cache_consistency(feature_paths_by_slide_id)

        needed_splits: list[str] = []
        if stage in (None, "fit"):
            if self.train_dataset is None:
                needed_splits.append("train")
            if self.val_dataset is None:
                needed_splits.append("val")
        if stage in (None, "validate") and self.val_dataset is None:
            needed_splits.append("val")
        if stage in (None, "test") and self.test_dataset is None:
            needed_splits.append("test")
        if stage in (None, "predict") and self.predict_dataset is None:
            needed_splits.append("predict")

        if "train" in needed_splits:
            self.train_dataset = self._build_split_dataset(slide_splits["train"])
        if "val" in needed_splits:
            self.val_dataset = self._build_split_dataset(slide_splits["val"])
        if "test" in needed_splits:
            self.test_dataset = self._build_split_dataset(slide_splits["test"])
        if "predict" in needed_splits:
            self.predict_dataset = self._build_split_dataset(slide_splits["predict"])

        self.logger.info(
            "Finished setup(stage=%s): train=%d val=%d test=%d predict=%d "
            "enc_dim=%s encoder=%s",
            stage,
            0 if self.train_dataset is None else len(self.train_dataset),  # type: ignore[arg-type]
            0 if self.val_dataset is None else len(self.val_dataset),  # type: ignore[arg-type]
            0 if self.test_dataset is None else len(self.test_dataset),  # type: ignore[arg-type]
            0 if self.predict_dataset is None else len(self.predict_dataset),  # type: ignore[arg-type]
            self.enc_dim,
            self.encoder_name,
        )

    # -- internals -----------------------------------------------------------

    def _ensure_main_labels_loaded(self: TCGAFeatureDataset) -> None:
        if (
            self._main_submitter_labels is not None
            and self._main_label_names is not None
        ):
            return
        if self._resolved_main_labels_path is None:
            self._resolved_main_labels_path = resolve_slide_main_label_path(
                self.root_dir,
                self.main_task,
                self.main_labels_path,
                self.logger,
            )
        match self.main_task:
            case "subtyping":
                submitter_labels, label_names = load_subtyping_labels(
                    self._resolved_main_labels_path, logger=self.logger
                )
            case _:
                raise ValueError(f"Unsupported main task: {self.main_task}")
        self._main_submitter_labels = submitter_labels
        self._main_label_names = label_names
        self.logger.info(
            "Loaded slide main labels: %d submitter(s), %d class(es) for task '%s'.",
            len(submitter_labels),
            len(label_names),
            self.main_task,
        )

    def _ensure_subtask_labels_loaded(self: TCGAFeatureDataset) -> None:
        for subtask in self.subtasks:
            if (
                subtask in self._subtask_submitter_labels
                and subtask in self._subtask_label_names
            ):
                continue
            resolved_path = self._resolved_subtask_labels_paths.get(subtask)
            if resolved_path is None:
                resolved_path = resolve_slide_subtask_label_path(
                    self.root_dir,
                    subtask,
                    self.subtask_labels_paths.get(subtask),
                    self.logger,
                )
                self._resolved_subtask_labels_paths[subtask] = resolved_path
            if subtask not in SUPPORTED_SUBTASKS:
                raise ValueError(f"Unsupported subtask: {subtask}")
            submitter_labels, mutation_names = load_signature_labels(
                resolved_path, logger=self.logger
            )
            self._subtask_submitter_labels[subtask] = submitter_labels
            self._subtask_label_names[subtask] = mutation_names
            self.logger.info(
                "Loaded slide subtask labels: %d submitter(s), %d label(s) for task '%s'.",
                len(submitter_labels),
                len(mutation_names),
                subtask,
            )

    def _get_slide_splits(
        self: TCGAFeatureDataset,
    ) -> dict[str, list[SlideRecord]]:
        if self._slide_splits is not None:
            return self._slide_splits
        if self._resolved_manifest_path is None:
            self._resolved_manifest_path = resolve_manifest_path(
                self.root_dir, self.manifest_path, self.logger
            )
        assert self._main_submitter_labels is not None
        subtask_labels_by_task = self._subtask_submitter_labels

        slide_records = load_slide_records(
            manifest_path=self._resolved_manifest_path,
            ordered_data_dir=self.ordered_data_dir,
            max_slides=None,
        )
        self.logger.info(
            "Loaded %d slide record(s) from manifest %s.",
            len(slide_records),
            self._resolved_manifest_path,
        )

        labelled_records = [
            record
            for record in slide_records
            if record.submitter_id in self._main_submitter_labels
            and all(
                record.submitter_id in subtask_labels_by_task[subtask]
                for subtask in self.subtasks
            )
        ]
        dropped = len(slide_records) - len(labelled_records)
        if dropped:
            self.logger.warning(
                "Dropped %d slide(s) without all required labels.", dropped
            )
        if not labelled_records:
            raise RuntimeError("No slides have all requested labels available.")
        if self.max_slides is not None:
            labelled_records = labelled_records[: self.max_slides]

        splits = split_slide_records(
            labelled_records,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            seed=self.random_seed,
        )
        splits["predict"] = list(labelled_records)
        self._slide_splits = splits
        return splits

    def _discover_feature_cache(
        self: TCGAFeatureDataset, slide_splits: dict[str, list[SlideRecord]]
    ) -> dict[str, str]:
        """Filter splits to slides with a cached ``<slide_id>.pt`` on disk.

        Mutates ``slide_splits`` in place to drop missing slides, and returns
        the ``slide_id -> path`` mapping used by the inner dataset.
        """
        if self._feature_paths_by_slide_id is not None:
            return self._feature_paths_by_slide_id

        all_records = {record.slide_id: record for record in slide_splits["predict"]}
        feature_paths: dict[str, str] = {}
        missing: list[str] = []
        for slide_id in all_records:
            candidate = os.path.join(self.features_dir, f"{slide_id}.pt")
            if os.path.isfile(candidate):
                feature_paths[slide_id] = candidate
            else:
                missing.append(slide_id)
        if missing:
            self.logger.warning(
                "Dropping %d slide(s) without a cached feature file under %s. "
                "First few: %s",
                len(missing),
                self.features_dir,
                missing[:5],
            )
        if not feature_paths:
            raise RuntimeError(
                f"No cached feature files found under {self.features_dir}. "
                "Run scripts/model_training/precompute_tile_features.py first."
            )

        for split_name, records in slide_splits.items():
            slide_splits[split_name] = [
                record for record in records if record.slide_id in feature_paths
            ]
        self._feature_paths_by_slide_id = feature_paths
        return feature_paths

    def _validate_cache_consistency(
        self: TCGAFeatureDataset, feature_paths_by_slide_id: dict[str, str]
    ) -> None:
        """Load one cached file and check encoder_name / enc_dim agreement."""
        sample_slide_id = next(iter(feature_paths_by_slide_id))
        sample_path = feature_paths_by_slide_id[sample_slide_id]
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        features = payload.get("features")
        if not isinstance(features, Tensor) or features.ndim != 2:
            raise ValueError(
                f"Cache sample {sample_path} has malformed 'features' "
                "(expected 2D Tensor)."
            )
        cache_enc_dim = int(features.shape[1])
        cache_encoder_name = payload.get("encoder_name")

        if (
            self._declared_enc_dim is not None
            and self._declared_enc_dim != cache_enc_dim
        ):
            raise ValueError(
                f"Declared enc_dim={self._declared_enc_dim} does not match "
                f"cached enc_dim={cache_enc_dim} in {sample_path}. Was the "
                "cache built with a different encoder?"
            )
        if (
            self._declared_encoder_name is not None
            and cache_encoder_name is not None
            and self._declared_encoder_name != cache_encoder_name
        ):
            raise ValueError(
                f"Declared expected_encoder_name="
                f"{self._declared_encoder_name!r} does not match cached "
                f"encoder_name={cache_encoder_name!r} in {sample_path}."
            )

        self._inferred_enc_dim = cache_enc_dim
        self._inferred_encoder_name = (
            cache_encoder_name
            if isinstance(cache_encoder_name, str)
            else self._declared_encoder_name
        )
        self.logger.info(
            "Cache sanity check passed (sample=%s enc_dim=%d encoder=%s).",
            sample_slide_id,
            cache_enc_dim,
            self._inferred_encoder_name,
        )

    def _build_split_dataset(
        self: TCGAFeatureDataset, records: Sequence[SlideRecord]
    ) -> _FeatureBagDataset:
        assert self._feature_paths_by_slide_id is not None
        assert self._main_submitter_labels is not None
        assert self._main_label_names is not None
        return _FeatureBagDataset(
            slide_records=records,
            feature_paths_by_slide_id=self._feature_paths_by_slide_id,
            main_submitter_labels=self._main_submitter_labels,
            main_label_names=self._main_label_names,
            subtasks=self.subtasks,
            subtask_submitter_labels=self._subtask_submitter_labels,
            subtask_label_names=self._subtask_label_names,
            portion_per_sample=self.portion_per_sample,
            max_tiles_per_bag=self.max_tiles_per_bag,
            task=self.main_task,
            logger=self.logger,
        )

    # -- from_config ---------------------------------------------------------

    @staticmethod
    def from_config(config: dict[str, Any]) -> TCGAFeatureDataset:
        """Create a TCGAFeatureDataset from a config dict."""
        root_dir = config.get("root_dir")
        if not isinstance(root_dir, str):
            raise ValueError("root_dir is required and must be a string.")
        features_dir = config.get("features_dir")
        if not isinstance(features_dir, str):
            raise ValueError("features_dir is required and must be a string.")

        main_task = config.get("main_task", "subtyping")
        if (
            not isinstance(main_task, str)
            or main_task not in TCGAFeatureDataset.SUPPORTED_MAIN_TASKS
        ):
            raise ValueError(
                f"main_task must be one of {TCGAFeatureDataset.SUPPORTED_MAIN_TASKS}."
            )

        subtasks = config.get("subtasks", None)
        if subtasks is not None:
            if not isinstance(subtasks, list) or any(
                not isinstance(task, str) or not task for task in subtasks
            ):
                raise ValueError(
                    "subtasks must be a list of non-empty strings or None."
                )
            for task in subtasks:
                if task not in TCGAFeatureDataset.SUPPORTED_SUBTASKS:
                    raise ValueError(
                        f"Unsupported subtask: {task!r}. Must be one of "
                        f"{TCGAFeatureDataset.SUPPORTED_SUBTASKS}."
                    )

        subtask_labels_paths = config.get("subtask_labels_paths", None)
        if subtask_labels_paths is not None and (
            not isinstance(subtask_labels_paths, dict)
            or any(
                not isinstance(task, str) or not isinstance(path, str)
                for task, path in subtask_labels_paths.items()
            )
        ):
            raise ValueError(
                "subtask_labels_paths must be a dict of task name to path string."
            )

        def _optional_str(key: str) -> str | None:
            value = config.get(key, None)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string when provided.")
            return value

        def _positive_int(key: str, default: int) -> int:
            value = config.get(key, default)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
            return int(value)

        def _fraction(key: str, default: float) -> float:
            value = config.get(key, default)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
            return float(value)

        max_slides = config.get("max_slides", None)
        if max_slides is not None and (
            not isinstance(max_slides, int) or max_slides <= 0
        ):
            raise ValueError("max_slides must be a positive integer or None.")

        max_tiles_per_bag = config.get("max_tiles_per_bag", None)
        if max_tiles_per_bag is not None and (
            not isinstance(max_tiles_per_bag, int) or max_tiles_per_bag <= 0
        ):
            raise ValueError("max_tiles_per_bag must be a positive integer or None.")

        enc_dim = config.get("enc_dim", None)
        if enc_dim is not None and (not isinstance(enc_dim, int) or enc_dim <= 0):
            raise ValueError("enc_dim must be a positive integer or None.")

        expected_encoder_name = config.get("expected_encoder_name", None)
        if expected_encoder_name is not None and not isinstance(
            expected_encoder_name, str
        ):
            raise ValueError("expected_encoder_name must be a string or None.")

        prefetch_factor = config.get("prefetch_factor", None)
        if prefetch_factor is not None and (
            not isinstance(prefetch_factor, int) or prefetch_factor <= 0
        ):
            raise ValueError("prefetch_factor must be a positive integer or None.")

        persistent_workers = config.get("persistent_workers", None)
        if persistent_workers is not None and not isinstance(persistent_workers, bool):
            raise ValueError("persistent_workers must be a boolean or None.")

        portion_per_sample_value = config.get("portion_per_sample", 1.0)
        if (
            not isinstance(portion_per_sample_value, (int, float))
            or not 0.0 < float(portion_per_sample_value) <= 1.0
        ):
            raise ValueError(
                f"portion_per_sample must be in (0, 1]. Got: {portion_per_sample_value}"
            )

        val_batch_size = config.get("val_batch_size", None)
        test_batch_size = config.get("test_batch_size", None)
        predict_batch_size = config.get("predict_batch_size", None)
        for name, value in (
            ("val_batch_size", val_batch_size),
            ("test_batch_size", test_batch_size),
            ("predict_batch_size", predict_batch_size),
        ):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")

        return TCGAFeatureDataset(
            root_dir=root_dir,
            features_dir=features_dir,
            main_task=main_task,
            subtasks=subtasks,
            manifest_path=_optional_str("manifest_path"),
            main_labels_path=_optional_str("main_labels_path"),
            subtask_labels_paths=subtask_labels_paths,
            ordered_data_dir=_optional_str("ordered_data_dir"),
            portion_per_sample=float(portion_per_sample_value),
            max_tiles_per_bag=max_tiles_per_bag,
            train_fraction=_fraction("train_fraction", 0.8),
            val_fraction=_fraction("val_fraction", 0.1),
            test_fraction=_fraction("test_fraction", 0.1),
            random_seed=int(config.get("random_seed", 42)),
            max_slides=max_slides,
            enc_dim=enc_dim,
            expected_encoder_name=expected_encoder_name,
            batch_size=_positive_int("batch_size", 8),
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            predict_batch_size=predict_batch_size,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", True)),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            shuffle_train=bool(config.get("shuffle_train", True)),
            drop_last_train=bool(config.get("drop_last_train", False)),
        )
