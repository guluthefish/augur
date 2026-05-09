"""TCGA Lightning datamodule that treats each slide as one dataset entry.

Main task is slide-level subtyping (classification). Pretext tasks are SBS
mutational-signature exposure vectors (regression / multilabel variants).
"""

from __future__ import annotations

import logging
import os
import random
from collections import OrderedDict
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from openslide import OpenSlide
import pandas as pd
import torch
from torch.utils.data import Dataset

from VexDR.datasets.dataset_abc import DatasetABC
from VexDR.datasets.utils import (
    SlideRecord,
    enumerate_slide_tile_centers,
    load_slide_records,
    read_tile_from_record,
    resolve_manifest_path,
    resolve_slide_main_label_path,
    resolve_slide_pretext_label_path,
    split_slide_records,
    _make_tile_record_for_mpp,
)
from VexDR.utils.logger import setup_logger

SUPPORTED_MAIN_TASKS: tuple[str, ...] = ("subtyping",)
SUPPORTED_PRETEXT_TASKS: tuple[str, ...] = (
    "sbs_regression",
    "sbs_thresholded_multilabel",
    "sbs_ranked_multilabel",
)
UNKNOWN_SUBTYPE_CLASS = "Other, specify"


def _normalize_subtype_label(value: Any) -> str:
    """Normalize a TCGA histologic-subtype value into a class label."""
    label = str(value).strip()
    if label.lower() == "nan" or label in {"", "[Not Available]", "[Not Applicable]"}:
        return UNKNOWN_SUBTYPE_CLASS

    parts = [part.strip() for part in label.split("|") if part.strip()]
    unique_parts = list(dict.fromkeys(parts))
    if len(unique_parts) == 1:
        label = unique_parts[0]

    if label == UNKNOWN_SUBTYPE_CLASS:
        return UNKNOWN_SUBTYPE_CLASS
    return label


def _load_subtyping_labels(
    labels_path: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Load slide-level subtyping labels with ``Other, specify`` fixed at index 0."""
    labels_df = pd.read_table(labels_path, dtype=str)
    required_columns = {"bcr_patient_barcode", "histologic_diagnosis"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        raise ValueError(
            f"Subtyping label table is missing column(s): {sorted(missing_columns)}"
        )

    submitter_ids = labels_df["bcr_patient_barcode"].astype(str).str.strip()
    raw_values = labels_df["histologic_diagnosis"]

    class_to_index: dict[str, int] = {UNKNOWN_SUBTYPE_CLASS: 0}
    submitter_labels: dict[str, int] = {}
    raw_submitter_labels: dict[str, str] = {}

    for submitter_id, raw_label in zip(submitter_ids, raw_values):
        if not submitter_id.startswith("TCGA-"):
            continue

        label = _normalize_subtype_label(raw_label)
        if label != UNKNOWN_SUBTYPE_CLASS and label not in class_to_index:
            class_to_index[label] = len(class_to_index)

        class_index = class_to_index[label]
        previous_label = raw_submitter_labels.get(submitter_id)
        if previous_label is not None and previous_label != label:
            if logger is not None:
                logger.warning(
                    "Conflicting subtyping labels for submitter %s: %s vs %s. "
                    "Using unknown class index 0.",
                    submitter_id,
                    previous_label,
                    label,
                )
            submitter_labels[submitter_id] = 0
            raw_submitter_labels[submitter_id] = UNKNOWN_SUBTYPE_CLASS
            continue

        submitter_labels[submitter_id] = class_index
        raw_submitter_labels[submitter_id] = label

    if not submitter_labels:
        raise RuntimeError(f"No TCGA subtyping labels were found in: {labels_path}")

    return submitter_labels, tuple(class_to_index)


def _load_sbs_labels(
    labels_path: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Load an SBS exposure table into ``(submitter_id -> vector, mutation_names)``.

    The first column is treated as the submitter id (renamed to
    ``submitter_id`` if necessary); the remaining columns are stored as a
    per-submitter ``float32`` vector.
    """
    labels_df = pd.read_table(labels_path)
    if "submitter_id" not in labels_df.columns:
        labels_df = labels_df.rename(columns={labels_df.columns[0]: "submitter_id"})
    labels_df["submitter_id"] = labels_df["submitter_id"].astype(str).str.strip()

    mutation_columns = [
        column for column in labels_df.columns if column != "submitter_id"
    ]
    if not mutation_columns:
        raise ValueError(f"SBS labels file has no label columns: {labels_path}")

    label_matrix = labels_df[mutation_columns].to_numpy(dtype=np.float32)
    submitter_ids = labels_df["submitter_id"].tolist()
    submitter_labels = {
        submitter_id: label_matrix[idx]
        for idx, submitter_id in enumerate(submitter_ids)
    }
    if logger is not None:
        logger.info(
            "Loaded SBS labels from %s: %d submitter(s), %d mutation column(s).",
            labels_path,
            len(submitter_labels),
            len(mutation_columns),
        )
    return submitter_labels, tuple(mutation_columns)


def _setup_logger_for_module() -> logging.Logger:
    """Create a logger for this module."""
    log_dir = os.path.join("logs", "datasets")
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="TCGASlideDataset", rank_zero_only=True)


def _pad_first_dim(tensors: Sequence[torch.Tensor], max_first: int) -> torch.Tensor:
    """Stack ``(K_i, *rest)`` tensors into ``(B, max_first, *rest)`` with zero padding."""
    first = tensors[0]
    out = torch.zeros((len(tensors), max_first, *first.shape[1:]), dtype=first.dtype)
    for i, tensor in enumerate(tensors):
        if tensor.shape[0] > 0:
            out[i, : tensor.shape[0]] = tensor
    return out


def _collate_metadata(
    metadata_list: Sequence[dict[str, Any]], max_K: int
) -> dict[str, Any]:
    """Collate per-sample metadata dicts, padding tile-aligned tensors to ``max_K``."""
    if not metadata_list:
        return {}
    out: dict[str, Any] = {}
    for key in metadata_list[0]:
        values = [item[key] for item in metadata_list]
        if all(isinstance(value, torch.Tensor) for value in values):
            shapes = {tuple(value.shape) for value in values}
            if len(shapes) == 1:
                out[key] = torch.stack(values)
            else:
                out[key] = _pad_first_dim(values, max_K)
        else:
            out[key] = values
    return out


def pad_bag_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate for variable-length tile bags.

    Pads each sample's ``image`` to the batch's max ``K`` along the tile axis,
    emits a parallel boolean ``mask: (B, K_max)`` flagging valid tiles, stacks
    fixed-shape tensors normally, and pads any tile-aligned metadata entries.
    """
    if not samples:
        return {}

    bag_sizes = [int(sample["image"].shape[0]) for sample in samples]
    max_K = max(bag_sizes) if bag_sizes else 0

    images = [sample["image"] for sample in samples]
    batched: dict[str, Any] = {"image": _pad_first_dim(images, max_K)}

    mask = torch.zeros(len(samples), max_K, dtype=torch.bool)
    for i, k in enumerate(bag_sizes):
        if k > 0:
            mask[i, :k] = True
    batched["mask"] = mask

    if "target" in samples[0]:
        batched["target"] = torch.stack([sample["target"] for sample in samples])

    if "metadata" in samples[0]:
        batched["metadata"] = _collate_metadata(
            [sample["metadata"] for sample in samples], max_K
        )

    handled = {"image", "target", "metadata", "mask"}
    for key in samples[0]:
        if key in handled:
            continue
        sub_samples = [sample[key] for sample in samples]
        if isinstance(sub_samples[0], dict):
            sub_batched: dict[str, Any] = {}
            for sub_key in sub_samples[0]:
                values = [item[sub_key] for item in sub_samples]
                if all(isinstance(value, torch.Tensor) for value in values):
                    sub_batched[sub_key] = torch.stack(values)
                else:
                    sub_batched[sub_key] = values
            batched[key] = sub_batched
        else:
            batched[key] = sub_samples

    return batched


class _SlideDataset(Dataset[dict[str, Any]]):
    """Map-style dataset where each entry is one slide and yields K sampled tiles.

    Tissue candidate centers are precomputed once per slide by the outer
    datamodule (with the configured ``stride``) and passed in via
    ``centers_by_slide_id``. ``__getitem__`` randomly samples
    ``floor(T * portion_per_sample)`` of those centers without replacement,
    where ``T`` is the per-slide candidate count. The returned tile stack has
    a per-slide ``K`` (variable across slides) — use a padding collate
    function when batching.

    Each sample emits the main subtyping target at ``sample["target"]`` (a
    scalar ``long`` class index) and one nested entry per pretext SBS task at
    ``sample[pretext_task]["target"]`` (a float vector of mutation exposures).
    """

    def __init__(
        self: _SlideDataset,
        *,
        slide_records: Sequence[SlideRecord],
        centers_by_slide_id: dict[str, list[tuple[int, int]]],
        main_submitter_labels: dict[str, int],
        main_label_names: Sequence[str],
        pretext_tasks: list[str] | None = None,
        pretext_submitter_labels: dict[str, dict[str, np.ndarray]] | None = None,
        pretext_label_names: dict[str, tuple[str, ...]] | None = None,
        portion_per_sample: float,
        tile_size: int,
        image_size: int,
        base_mpp: float,
        task: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.slide_records = list(slide_records)
        self.centers_by_slide_id = centers_by_slide_id
        self.main_submitter_labels = main_submitter_labels
        self.main_label_names = tuple(main_label_names)
        self.pretext_tasks: list[str] = list(pretext_tasks or [])
        self.pretext_submitter_labels: dict[str, dict[str, np.ndarray]] = (
            pretext_submitter_labels or {}
        )
        self.pretext_label_names: dict[str, tuple[str, ...]] = dict(
            pretext_label_names or {}
        )
        self.portion_per_sample = float(portion_per_sample)
        self.tile_size = int(tile_size)
        self.image_size = int(image_size)
        self.base_mpp = float(base_mpp)
        self.task = task
        self.logger = logger or _setup_logger_for_module()
        # LRU cache of OpenSlide handles, bounded to avoid unbounded growth
        # over a long fit run with shuffled access. Each handle holds an
        # internal tile-pyramid cache; uncapped, this leaks ~30-50 MB per
        # unique slide visited and eventually exhausts the cgroup memory.
        self._slides: OrderedDict[str, OpenSlide] = OrderedDict()
        self._max_open_slides: int = 16

    def __len__(self: _SlideDataset) -> int:
        return len(self.slide_records)

    def __getstate__(self: _SlideDataset) -> dict[str, Any]:
        """Drop open slide handles when dataloader workers are forked/pickled."""
        state = self.__dict__.copy()
        state["_slides"] = OrderedDict()
        return state

    def close(self: _SlideDataset) -> None:
        """Close any cached slide handles held by the current worker."""
        for slide in self._slides.values():
            slide.close()
        self._slides = OrderedDict()

    def _get_slide(self: _SlideDataset, slide_path: str) -> OpenSlide:
        slide = self._slides.get(slide_path)
        if slide is None:
            while len(self._slides) >= self._max_open_slides:
                _, evicted = self._slides.popitem(last=False)
                evicted.close()
            slide = OpenSlide(slide_path)
            self._slides[slide_path] = slide
        else:
            self._slides.move_to_end(slide_path)
        return slide

    def _sample_centers(
        self: _SlideDataset, centers: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if not centers:
            return []
        sample_size = max(1, int(len(centers) * self.portion_per_sample))
        sample_size = min(sample_size, len(centers))
        return random.sample(list(centers), sample_size)

    def __getitem__(self: _SlideDataset, index: int) -> dict[str, Any]:
        slide_record = self.slide_records[index]
        centers = self.centers_by_slide_id[slide_record.slide_id]
        slide = self._get_slide(slide_record.slide_path)

        sampled = self._sample_centers(centers)

        tile_images: list[np.ndarray] = []
        tile_xy: list[tuple[int, int]] = []
        tile_levels: list[int] = []
        tile_sizes: list[int] = []
        for center_x, center_y in sampled:
            record = _make_tile_record_for_mpp(
                slide_record,
                slide,
                center_x=center_x,
                center_y=center_y,
                output_size=self.tile_size,
                target_mpp=self.base_mpp,
                logger=self.logger,
            )
            image = read_tile_from_record(slide, record)
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                image = cv2.resize(  # pylint: disable=no-member
                    image,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
                )
            tile_images.append(image.astype(np.float32) / 255.0)
            tile_xy.append((record.x, record.y))
            tile_levels.append(record.level)
            tile_sizes.append(record.size)

        images_np = np.stack(tile_images, axis=0)  # (K, H, W, 3)
        images = torch.from_numpy(np.ascontiguousarray(images_np)).permute(0, 3, 1, 2)

        target = torch.tensor(
            self.main_submitter_labels[slide_record.submitter_id],
            dtype=torch.long,
        )

        metadata: dict[str, Any] = {
            "slide_id": slide_record.slide_id,
            "submitter_id": slide_record.submitter_id,
            "base_mpp": self.base_mpp,
            "task": self.task,
            "tile_centers": torch.tensor(sampled, dtype=torch.long),
            "tile_xy": torch.tensor(tile_xy, dtype=torch.long),
            "tile_level": torch.tensor(tile_levels, dtype=torch.long),
            "tile_size": torch.tensor(tile_sizes, dtype=torch.long),
        }
        sample: dict[str, Any] = {
            "image": images,
            "target": target,
            "metadata": metadata,
        }

        if self.pretext_tasks:
            metadata["pretext_tasks"] = list(self.pretext_tasks)
            for pretext_task in self.pretext_tasks:
                task_submitter_labels = self.pretext_submitter_labels.get(pretext_task)
                if task_submitter_labels is None:
                    raise RuntimeError(
                        f"Pretext task '{pretext_task}' is missing label mappings."
                    )
                pretext_vector = task_submitter_labels[slide_record.submitter_id]
                sample[pretext_task] = {
                    "target": torch.from_numpy(
                        np.ascontiguousarray(pretext_vector)
                    ).float()
                }

        return sample


class TCGASlideDataset(DatasetABC):
    """Lightning datamodule that yields one multi-tile sample per slide.

    Main task is slide-level subtyping (classification): the per-submitter
    histologic-subtype class is read from
    ``<root_dir>/atlases/slide_main_atlas.txt`` (or ``main_labels_path``) and
    emitted as a scalar ``long`` at ``batch["target"]``. ``Other, specify`` is
    fixed at class index 0 (treat as the unknown class for ignore-index losses).

    Pretext tasks are slide-level SBS exposure vectors keyed by the entries
    of ``slide_pretext_atlas.txt`` — currently ``sbs_regression``,
    ``sbs_thresholded_multilabel``, and ``sbs_ranked_multilabel``. Each
    configured pretext task adds a nested ``batch[pretext_task]["target"]``
    entry holding a float vector of length ``num_pretext_labels[pretext_task]``.

    Per-slide bag size ``K`` varies (it depends on the per-slide tissue area
    and the configured ``stride`` / ``portion_per_sample``); the default
    :func:`pad_bag_collate` pads each batch to its max ``K`` and emits a
    parallel ``mask: (B, K_max)`` boolean.

    Parameters
    ----------
    root_dir
        Root directory containing the manifest and labels files, or their
        parent if their paths are relative.
    main_task
        Slide-level main task. Currently only ``"subtyping"`` is supported.
    pretext_tasks
        Optional list of slide-level pretext tasks. Each entry must be one of
        :attr:`SUPPORTED_PRETEXT_TASKS`.
    manifest_path, main_labels_path, pretext_labels_paths, ordered_data_dir
        Optional explicit paths. When omitted, the manifest is resolved from
        ``atlases/manifest_atlas.txt``, the main label table from
        ``atlases/slide_main_atlas.txt``, and each pretext task's table from
        ``atlases/slide_pretext_atlas.txt``.
    portion_per_sample
        Fraction of valid tissue candidates to keep per slide. Each
        ``__getitem__`` randomly samples
        ``max(1, floor(T * portion_per_sample))`` of them without replacement,
        where ``T`` is the per-slide candidate count. Must be in ``(0, 1]``.
    stride
        Spacing between consecutive candidate tile centers, in the same units
        as ``tile_size`` (pixels at ``base_mpp``). Defaults to ``tile_size``
        (non-overlapping tiles).
    tile_size, image_size, base_mpp, min_tissue_fraction, thumbnail_max_size,
    white_threshold
        Tile-extraction parameters. ``tile_size`` is the read size at
        ``base_mpp`` before resizing to ``image_size``. ``min_tissue_fraction``
        and ``white_threshold`` filter candidates against a thumbnail tissue
        mask.
    train_fraction, val_fraction, test_fraction, random_seed
        Patient-level (submitter-level) splits. Fractions must sum to 1.
    max_slides
        Optional cap applied after dropping slides without all configured
        labels.
    logger, batch_size, val_batch_size, test_batch_size, predict_batch_size,
    num_workers, pin_memory, persistent_workers, prefetch_factor,
    shuffle_train, drop_last_train
        Standard Lightning datamodule wiring.
    collate_fn
        Override the default :func:`pad_bag_collate`. Use ``None`` to keep the
        padding collate that handles variable-length bags.
    """

    SUPPORTED_MAIN_TASKS = SUPPORTED_MAIN_TASKS
    SUPPORTED_PRETEXT_TASKS = SUPPORTED_PRETEXT_TASKS

    def __init__(
        self: TCGASlideDataset,
        root_dir: str,
        *,
        main_task: str = "subtyping",
        pretext_tasks: list[str] | None = None,
        manifest_path: str | None = None,
        main_labels_path: str | None = None,
        pretext_labels_paths: dict[str, str] | None = None,
        ordered_data_dir: str | None = None,
        portion_per_sample: float = 1.0,
        stride: int | None = None,
        tile_size: int = 512,
        image_size: int = 256,
        base_mpp: float = 0.25,
        min_tissue_fraction: float = 0.5,
        thumbnail_max_size: int = 1024,
        white_threshold: float = 0.85,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        random_seed: int = 42,
        max_slides: int | None = None,
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
            self.logger.error("Unsupported main task: %s", main_task)
            raise ValueError(
                f"Unsupported main_task: {main_task}. "
                f"Must be one of {self.SUPPORTED_MAIN_TASKS}."
            )
        if pretext_tasks is not None:
            if not isinstance(pretext_tasks, list) or any(
                not isinstance(task, str) or not task for task in pretext_tasks
            ):
                raise ValueError(
                    "pretext_tasks must be a list of non-empty strings or None."
                )
            if len(set(pretext_tasks)) != len(pretext_tasks):
                raise ValueError("pretext_tasks must not contain duplicates.")
            for task in pretext_tasks:
                if task not in self.SUPPORTED_PRETEXT_TASKS:
                    self.logger.error("Unsupported pretext task: %s", task)
                    raise ValueError(
                        f"Unsupported pretext task: {task}. "
                        f"Must be one of {self.SUPPORTED_PRETEXT_TASKS}."
                    )
        if pretext_labels_paths is not None:
            if not isinstance(pretext_labels_paths, dict) or any(
                not isinstance(task, str) or not isinstance(path, str)
                for task, path in pretext_labels_paths.items()
            ):
                raise ValueError(
                    "pretext_labels_paths must be a dict of task name to path string."
                )
        if (
            not isinstance(portion_per_sample, (int, float))
            or not 0.0 < float(portion_per_sample) <= 1.0
        ):
            raise ValueError(
                "portion_per_sample must be a number in (0, 1]. "
                f"Got: {portion_per_sample}"
            )
        if stride is not None and (not isinstance(stride, int) or stride <= 0):
            raise ValueError(
                f"stride must be a positive integer or None. Got: {stride}"
            )
        if tile_size <= 0:
            raise ValueError("tile_size must be a positive integer.")
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if base_mpp <= 0:
            raise ValueError("base_mpp must be positive.")
        if not 0 <= min_tissue_fraction <= 1:
            raise ValueError("min_tissue_fraction must be between 0 and 1.")
        if not 0 <= white_threshold <= 1:
            raise ValueError("white_threshold must be between 0 and 1.")
        if max_slides is not None and max_slides <= 0:
            raise ValueError("max_slides must be a positive integer or None.")
        split_sum = train_fraction + val_fraction + test_fraction
        if not np.isclose(split_sum, 1.0):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1."
            )

        self.root_dir = root_dir
        self.main_task = main_task
        self.pretext_tasks: list[str] = list(pretext_tasks or [])
        self.manifest_path = manifest_path
        self.main_labels_path = main_labels_path
        self.pretext_labels_paths: dict[str, str] = dict(pretext_labels_paths or {})
        self.ordered_data_dir = (
            ordered_data_dir
            if ordered_data_dir is not None
            else os.path.join(root_dir, "ordered_data")
        )
        self.portion_per_sample = float(portion_per_sample)
        self.stride: int = int(stride) if stride is not None else int(tile_size)
        self.tile_size = int(tile_size)
        self.image_size = int(image_size)
        self.base_mpp = float(base_mpp)
        self.min_tissue_fraction = float(min_tissue_fraction)
        self.thumbnail_max_size = int(thumbnail_max_size)
        self.white_threshold = float(white_threshold)
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.random_seed = int(random_seed)
        self.max_slides = max_slides

        self._resolved_manifest_path: str | None = None
        self._resolved_main_labels_path: str | None = None
        self._resolved_pretext_labels_paths: dict[str, str] = {}
        self._main_submitter_labels: dict[str, int] | None = None
        self._main_label_names: tuple[str, ...] | None = None
        self._pretext_submitter_labels: dict[str, dict[str, np.ndarray]] = {}
        self._pretext_label_names: dict[str, tuple[str, ...]] = {}
        self._slide_splits: dict[str, list[SlideRecord]] | None = None
        self._centers_by_slide_id: dict[str, list[tuple[int, int]]] | None = None

    @staticmethod
    def from_config(config: dict[str, Any]) -> TCGASlideDataset:
        """Create a TCGASlideDataset from a config dict."""
        root_dir = config.get("root_dir")
        if not isinstance(root_dir, str):
            raise ValueError("root_dir is required and must be a string.")

        main_task = config.get("main_task", "subtyping")
        if (
            not isinstance(main_task, str)
            or main_task not in TCGASlideDataset.SUPPORTED_MAIN_TASKS
        ):
            raise ValueError(
                f"main_task must be one of {TCGASlideDataset.SUPPORTED_MAIN_TASKS}."
            )

        pretext_tasks = config.get("pretext_tasks", None)
        if pretext_tasks is not None:
            if not isinstance(pretext_tasks, list) or any(
                not isinstance(task, str) or not task for task in pretext_tasks
            ):
                raise ValueError(
                    "pretext_tasks must be a list of non-empty strings or None."
                )
            for task in pretext_tasks:
                if task not in TCGASlideDataset.SUPPORTED_PRETEXT_TASKS:
                    raise ValueError(
                        f"Unsupported pretext task: {task}. Must be one of "
                        f"{TCGASlideDataset.SUPPORTED_PRETEXT_TASKS}."
                    )

        pretext_labels_paths = config.get("pretext_labels_paths", None)
        if pretext_labels_paths is not None:
            if not isinstance(pretext_labels_paths, dict) or any(
                not isinstance(task, str) or not isinstance(path, str)
                for task, path in pretext_labels_paths.items()
            ):
                raise ValueError(
                    "pretext_labels_paths must be a dict of task name to path string."
                )

        def _optional_str(key: str) -> str | None:
            value = config.get(key, None)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or None.")
            return value

        def _positive_int(key: str, default: int) -> int:
            value = config.get(key, default)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
            return value

        def _fraction(key: str, default: float) -> float:
            value = config.get(key, default)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
            return float(value)

        base_mpp = config.get("base_mpp", 0.25)
        if not isinstance(base_mpp, (int, float)) or base_mpp <= 0:
            raise ValueError("base_mpp must be a positive number.")

        max_slides = config.get("max_slides", None)
        if max_slides is not None and (
            not isinstance(max_slides, int) or max_slides <= 0
        ):
            raise ValueError("max_slides must be a positive integer or None.")

        predict_batch_size = config.get("predict_batch_size", None)
        val_batch_size = config.get("val_batch_size", None)
        test_batch_size = config.get("test_batch_size", None)
        for name, value in (
            ("val_batch_size", val_batch_size),
            ("test_batch_size", test_batch_size),
            ("predict_batch_size", predict_batch_size),
        ):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")

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
                "portion_per_sample must be a number in (0, 1]. "
                f"Got: {portion_per_sample_value}"
            )

        stride_value = config.get("stride", None)
        if stride_value is not None and (
            not isinstance(stride_value, int) or stride_value <= 0
        ):
            raise ValueError(
                f"stride must be a positive integer or None. Got: {stride_value}"
            )

        return TCGASlideDataset(
            root_dir=root_dir,
            main_task=main_task,
            pretext_tasks=pretext_tasks,
            manifest_path=_optional_str("manifest_path"),
            main_labels_path=_optional_str("main_labels_path"),
            pretext_labels_paths=pretext_labels_paths,
            ordered_data_dir=_optional_str("ordered_data_dir"),
            portion_per_sample=float(portion_per_sample_value),
            stride=stride_value,
            tile_size=_positive_int("tile_size", 512),
            image_size=_positive_int("image_size", 256),
            base_mpp=float(base_mpp),
            min_tissue_fraction=_fraction("min_tissue_fraction", 0.5),
            thumbnail_max_size=_positive_int("thumbnail_max_size", 1024),
            white_threshold=_fraction("white_threshold", 0.85),
            train_fraction=_fraction("train_fraction", 0.8),
            val_fraction=_fraction("val_fraction", 0.1),
            test_fraction=_fraction("test_fraction", 0.1),
            random_seed=int(config.get("random_seed", 42)),
            max_slides=max_slides,
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

    def prepare_data(self: TCGASlideDataset) -> None:
        """Resolve manifest and label paths on one process."""
        self.logger.info("Preparing TCGA slide data for main task: %s", self.main_task)
        self._resolved_manifest_path = resolve_manifest_path(
            self.root_dir, self.manifest_path, self.logger
        )
        self._resolved_main_labels_path = resolve_slide_main_label_path(
            self.root_dir, self.main_task, self.main_labels_path, self.logger
        )
        for pretext_task in self.pretext_tasks:
            self._resolved_pretext_labels_paths[pretext_task] = (
                resolve_slide_pretext_label_path(
                    self.root_dir,
                    pretext_task,
                    self.pretext_labels_paths.get(pretext_task),
                    self.logger,
                )
            )
        self.logger.info("Finished prepare_data for main task: %s", self.main_task)

    def setup(self: TCGASlideDataset, stage: str | None = None) -> None:
        """Build slide splits, load labels, and precompute tissue centers."""
        valid_stages = {None, "fit", "validate", "test", "predict"}
        if stage not in valid_stages:
            raise ValueError(f"Unsupported stage: {stage}")

        self.logger.info(
            "Starting setup(stage=%s) for main task: %s", stage, self.main_task
        )

        self._ensure_main_labels_loaded()
        self._ensure_pretext_labels_loaded()
        slide_splits = self._get_slide_splits()

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

        self._ensure_tile_centers(
            [record for split in set(needed_splits) for record in slide_splits[split]]
        )

        if "train" in needed_splits:
            self.train_dataset = self._build_slide_dataset(slide_splits["train"])
        if "val" in needed_splits:
            self.val_dataset = self._build_slide_dataset(slide_splits["val"])
        if "test" in needed_splits:
            self.test_dataset = self._build_slide_dataset(slide_splits["test"])
        if "predict" in needed_splits:
            self.predict_dataset = self._build_slide_dataset(slide_splits["predict"])

        self.logger.info(
            "Finished setup(stage=%s): train=%d val=%d test=%d predict=%d",
            stage,
            0 if self.train_dataset is None else len(self.train_dataset),  # type: ignore[arg-type]
            0 if self.val_dataset is None else len(self.val_dataset),  # type: ignore[arg-type]
            0 if self.test_dataset is None else len(self.test_dataset),  # type: ignore[arg-type]
            0 if self.predict_dataset is None else len(self.predict_dataset),  # type: ignore[arg-type]
        )

    def teardown(self: TCGASlideDataset, stage: str | None = None) -> None:
        """Close cached slide handles kept by split datasets."""
        del stage
        for dataset in (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.predict_dataset,
        ):
            if isinstance(dataset, _SlideDataset):
                dataset.close()

    def _ensure_main_labels_loaded(self: TCGASlideDataset) -> None:
        """Load the slide-level main-task labels once."""
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
                submitter_labels, label_names = _load_subtyping_labels(
                    self._resolved_main_labels_path,
                    logger=self.logger,
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

    def _ensure_pretext_labels_loaded(self: TCGASlideDataset) -> None:
        """Load optional slide-level pretext SBS labels once per task."""
        for pretext_task in self.pretext_tasks:
            if (
                pretext_task in self._pretext_submitter_labels
                and pretext_task in self._pretext_label_names
            ):
                continue

            resolved_path = self._resolved_pretext_labels_paths.get(pretext_task)
            if resolved_path is None:
                resolved_path = resolve_slide_pretext_label_path(
                    self.root_dir,
                    pretext_task,
                    self.pretext_labels_paths.get(pretext_task),
                    self.logger,
                )
                self._resolved_pretext_labels_paths[pretext_task] = resolved_path

            if pretext_task in SUPPORTED_PRETEXT_TASKS:
                submitter_labels, mutation_names = _load_sbs_labels(
                    resolved_path, logger=self.logger
                )
            else:
                raise ValueError(f"Unsupported pretext task: {pretext_task}")

            self._pretext_submitter_labels[pretext_task] = submitter_labels
            self._pretext_label_names[pretext_task] = mutation_names
            self.logger.info(
                "Loaded slide pretext labels: %d submitter(s), %d label(s) for task '%s'.",
                len(submitter_labels),
                len(mutation_names),
                pretext_task,
            )

    def _get_slide_splits(self: TCGASlideDataset) -> dict[str, list[SlideRecord]]:
        """Load slides, drop unlabelled ones, then split by submitter."""
        if self._slide_splits is not None:
            return self._slide_splits

        if self._resolved_manifest_path is None:
            self._resolved_manifest_path = resolve_manifest_path(
                self.root_dir, self.manifest_path, self.logger
            )
        self._ensure_main_labels_loaded()
        self._ensure_pretext_labels_loaded()
        assert self._main_submitter_labels is not None
        pretext_labels_by_task = self._pretext_submitter_labels

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
                record.submitter_id in pretext_labels_by_task[pretext_task]
                for pretext_task in self.pretext_tasks
            )
        ]
        dropped = len(slide_records) - len(labelled_records)
        if dropped:
            self.logger.warning(
                "Dropped %d slide(s) without matching label(s) for main task '%s'.",
                dropped,
                self.main_task,
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

    def _ensure_tile_centers(
        self: TCGASlideDataset, slide_records: Sequence[SlideRecord]
    ) -> None:
        """Precompute tissue centers for the requested slides once."""
        if self._centers_by_slide_id is None:
            self._centers_by_slide_id = {}

        pending = [
            record
            for record in slide_records
            if record.slide_id not in self._centers_by_slide_id
        ]
        total = len(pending)
        for index, record in enumerate(pending, start=1):
            if index == 1 or index == total or index % 25 == 0:
                self.logger.info(
                    "Enumerating tissue tile centers for slide %d/%d (%s).",
                    index,
                    total,
                    record.slide_id,
                )
            centers = enumerate_slide_tile_centers(
                record,
                output_size=self.tile_size,
                context_mpp=self.base_mpp,
                min_tissue_fraction=self.min_tissue_fraction,
                thumbnail_max_size=self.thumbnail_max_size,
                white_threshold=self.white_threshold,
                stride=self.stride,
                logger=self.logger,
            )
            if not centers:
                self.logger.warning(
                    "No tissue regions found for slide %s; it will be dropped.",
                    record.slide_id,
                )
            self._centers_by_slide_id[record.slide_id] = centers

    def _build_slide_dataset(
        self: TCGASlideDataset, slide_records: Sequence[SlideRecord]
    ) -> _SlideDataset:
        assert self._centers_by_slide_id is not None
        assert self._main_submitter_labels is not None
        assert self._main_label_names is not None

        usable_records = [
            record
            for record in slide_records
            if self._centers_by_slide_id.get(record.slide_id)
        ]
        return _SlideDataset(
            slide_records=usable_records,
            centers_by_slide_id=self._centers_by_slide_id,
            main_submitter_labels=self._main_submitter_labels,
            main_label_names=self._main_label_names,
            pretext_tasks=self.pretext_tasks,
            pretext_submitter_labels=self._pretext_submitter_labels,
            pretext_label_names=self._pretext_label_names,
            portion_per_sample=self.portion_per_sample,
            tile_size=self.tile_size,
            image_size=self.image_size,
            base_mpp=self.base_mpp,
            task=self.main_task,
            logger=self.logger,
        )

    @property
    def main_label_names(self: TCGASlideDataset) -> tuple[str, ...]:
        """Class names for the configured slide-level main task."""
        if self._main_label_names is None:
            self._ensure_main_labels_loaded()
        assert self._main_label_names is not None
        return self._main_label_names

    @property
    def num_main_labels(self: TCGASlideDataset) -> int:
        """Number of classes for the configured slide-level main task."""
        return len(self.main_label_names)

    @property
    def pretext_label_names(self: TCGASlideDataset) -> dict[str, tuple[str, ...]]:
        """Mutation/column names per configured slide-level pretext task."""
        if not self.pretext_tasks:
            return {}
        if any(task not in self._pretext_label_names for task in self.pretext_tasks):
            self._ensure_pretext_labels_loaded()
        return {task: self._pretext_label_names[task] for task in self.pretext_tasks}

    @property
    def num_pretext_labels(self: TCGASlideDataset) -> dict[str, int]:
        """Vector length per configured slide-level pretext task."""
        return {task: len(names) for task, names in self.pretext_label_names.items()}
