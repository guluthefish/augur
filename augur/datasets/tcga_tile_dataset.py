"""Unified TCGA Lightning datamodule with multi-task tile processors."""

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

from augur.datasets.dataset_abc import DatasetABC
from augur.datasets.hematoxylin import process_hematoxylin_task
from augur.datasets.jigmag import process_jigmag_task
from augur.datasets.magnification import process_magnification_task
from augur.datasets.utils import (
    TileRecord,
    SlideRecord,
    as_image_tensor,
    as_mask_tensor,
    derive_bcss_slide_name,
    load_slide_records,
    load_tissue_mask_label_for_free_tile,
    tile_record_center_l0,
    read_tile_from_record,
    resolve_manifest_path,
    resolve_tissue_label_metadata_path,
    sample_tile_records,
    split_slide_records,
    split_slide_records_with_budget,
)
from augur.utils.logger import setup_logger

SUPPORTED_TASKS = (
    "tissue_segmentation",
    "tumor_classification",
    "jigmag",
    "magnification",
    "hematoxylin",
)


def _setup_logger_for_module() -> logging.Logger:
    """Create a logger for this module."""
    log_dir = os.path.join("logs", "datasets")
    os.makedirs(log_dir, exist_ok=True)
    return setup_logger(log_dir, name="TCGATileDataset", rank_zero_only=True)


class _TileDataset(Dataset[dict[str, Any]]):
    """
    Map-style dataset that yields one multi-task TCGA sample per tile.

    Parameters
    ----------
    records:
        A list of TileRecord objects specifying the base tiles to sample from.
    tasks:
        A list of task names to build for each sample. Supported tasks are:
        - "tissue_segmentation": Predict a tissue segmentation mask for the tile.
        - "tumor_classification": Predict a binary tumor vs normal label for the tile.
        - "hematoxylin": Predict the hematoxylin channel of the tile.
        - "magnification": Predict the magnification level of the tile from a set of candidates.
        - "jigmag": Predict the permutation of a 2x2 puzzle of tiles at different magnifications.
    tile_size:
        The width and height of the output tile image in pixels.
        This is the size that will be used to sample tiles before resizing to image_size.
    image_size:
        The width and height of the final output image in pixels after any resizing.
    base_mpp:
        The mpp at which the base tiles in the records are defined.
        This is used to calculate the appropriate level to read from the slide.
    magnification_mpps:
        The mpp values to use as candidates for the magnification task.
        The target mpp will be randomly selected from this list.
        If empty or None, the magnification task will not be built.
    jigmag_mpps:
        The mpp values to use for the 4 tiles in the jigmag task.
        Must contain exactly 4 values since the puzzle is always 2x2.
        If empty or None, the jigmag task will not be built.
    tissue_segmentation_n_classes:
        The number of classes for the tissue segmentation task.
        If None, the task will not be built.
    random_seed:
        The random seed to use for any random operations in the dataset,
        such as sampling tiles or selecting magnification levels.
    root_dir:
        The root directory where slide files and labels are stored.
        This is used for loading tissue segmentation masks.
        If None, tissue segmentation tasks will not be built since labels cannot be loaded.
    task_transforms:
        A dict of task names to callables that take and return a task output dict.
        These transforms will be applied to each task's output after it is built and before tensorization.
    sample_transform:
        A callable that takes and returns a full sample dict.
        This will be applied after all tasks are built and tensorized, allowing for joint transforms across tasks.
    logger:
        A logger instance to use for logging messages from this dataset.
        If None, a default logger will be created.
    """

    def __init__(
        self: _TileDataset,
        *,
        records: Sequence[TileRecord],
        tasks: Sequence[str],
        tile_size: int,
        image_size: int,
        base_mpp: float,
        magnification_mpps: Sequence[float] | None = None,
        jigmag_mpps: Sequence[float] | None = None,
        tissue_segmentation_n_classes: int | None = None,
        bcss_roi_groups: dict[str, pd.DataFrame] | None = None,
        slide_cache_size: int = 16,
        random_seed: int = 42,
        root_dir: str | None = None,
        task_transforms: (
            dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None
        ) = None,
        sample_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if slide_cache_size <= 0:
            raise ValueError("slide_cache_size must be a positive integer.")
        self.records = list(records)
        self.tasks = list(tasks)
        self.tile_size = tile_size
        self.image_size = image_size
        self.base_mpp = float(base_mpp)
        self.root_dir = root_dir
        self.magnification_mpps = (
            tuple(float(mpp) for mpp in magnification_mpps)
            if magnification_mpps is not None
            else None
        )
        self.jigmag_mpps = (
            tuple(float(mpp) for mpp in jigmag_mpps)
            if jigmag_mpps is not None
            else None
        )
        self.tissue_segmentation_n_classes = tissue_segmentation_n_classes
        self.bcss_roi_groups = bcss_roi_groups
        self.slide_cache_size = slide_cache_size
        self.random_seed = random_seed
        self.task_transforms = task_transforms or {}
        self.sample_transform = sample_transform
        self.logger = logger or _setup_logger_for_module()
        self._slides: OrderedDict[str, OpenSlide] = OrderedDict()

        # Check task validity at the dataset level since tasks are shared across all samples.
        unknown_tasks = set(self.tasks).difference(SUPPORTED_TASKS)
        if unknown_tasks:
            self.logger.error(f"Unsupported tasks requested: {sorted(unknown_tasks)}")
            raise ValueError(f"Unsupported tasks requested: {sorted(unknown_tasks)}")

    def __len__(self: _TileDataset) -> int:
        """Return the number of sampled base tiles."""
        return len(self.records)

    def __getstate__(self: _TileDataset) -> dict[str, Any]:
        """Drop open slide handles when dataloader workers are forked/pickled."""
        state = self.__dict__.copy()
        state["_slides"] = OrderedDict()
        return state

    def close(self: _TileDataset) -> None:
        """Close any cached slide handles held by the current worker."""
        for slide in self._slides.values():
            slide.close()
        self._slides = OrderedDict()

    def _get_slide(self: _TileDataset, slide_path: str) -> OpenSlide:
        """Return a cached slide handle, opening and LRU-evicting as needed."""
        slide = self._slides.get(slide_path)
        if slide is not None:
            self._slides.move_to_end(slide_path)
            return slide
        slide = OpenSlide(slide_path)
        self._slides[slide_path] = slide
        while len(self._slides) > self.slide_cache_size:
            _, evicted = self._slides.popitem(last=False)
            evicted.close()
        return slide

    def __getitem__(self: _TileDataset, index: int) -> dict[str, Any]:
        """Build a multi-task sample from one base tile record."""
        tile_record = self.records[index]
        slide = self._get_slide(tile_record.slide_path)
        base_image = read_tile_from_record(slide, tile_record)
        center_x, center_y = tile_record_center_l0(slide, tile_record)
        rng = random.Random(f"{self.random_seed}:{index}")

        sample: dict[str, Any] = {
            "metadata": {
                "slide_id": tile_record.slide_id,
                "submitter_id": tile_record.submitter_id,
                "x": tile_record.x,
                "y": tile_record.y,
                "level": tile_record.level,
                "size": tile_record.size,
                "center_x": center_x,
                "center_y": center_y,
                "base_mpp": self.base_mpp,
            }
        }
        if tile_record.roi_name is not None:
            sample["metadata"]["roi_name"] = tile_record.roi_name
        if tile_record.roi_xmin is not None:
            sample["metadata"]["roi_xmin"] = tile_record.roi_xmin
        if tile_record.roi_ymin is not None:
            sample["metadata"]["roi_ymin"] = tile_record.roi_ymin
        if tile_record.roi_xmax is not None:
            sample["metadata"]["roi_xmax"] = tile_record.roi_xmax
        if tile_record.roi_ymax is not None:
            sample["metadata"]["roi_ymax"] = tile_record.roi_ymax
        for task_name in self.tasks:
            task_output = self._build_task(
                task_name,
                base_image=base_image,
                tile_record=tile_record,
                slide=slide,
                center_x=center_x,
                center_y=center_y,
                rng=rng,
            )
            task_output = self._tensorize_task_output(task_name, task_output)
            if task_name in self.task_transforms:
                task_output = self.task_transforms[task_name](task_output)
            sample[task_name] = task_output

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample

    def _build_main_task(
        self: _TileDataset,
        *,
        task_name: str,
        base_image: np.ndarray,
        tile_record: TileRecord,
        slide: OpenSlide,
    ) -> dict[str, Any]:
        """Build a base-scale task such as tissue segmentation or classification."""

        resized_image = cv2.resize(  # pylint: disable=no-member
            base_image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
        )

        match task_name:
            case "tissue_segmentation":
                if self.root_dir is None:
                    self.logger.error(
                        "root_dir must be set on TCGATileDataset for tissue segmentation."
                    )
                    raise ValueError(
                        "root_dir must be set on TCGATileDataset for tissue segmentation."
                    )
                if self.tissue_segmentation_n_classes is None:
                    self.logger.error(
                        "tissue_segmentation_n_classes must be set for tissue segmentation task."
                    )
                    raise ValueError(
                        "tissue_segmentation_n_classes must be set for tissue segmentation task."
                    )

                slide_name = derive_bcss_slide_name(
                    tile_record.slide_path, tile_record.submitter_id
                )
                slide_rois = (
                    self.bcss_roi_groups.get(slide_name)
                    if self.bcss_roi_groups is not None
                    else None
                )
                target_mask = load_tissue_mask_label_for_free_tile(
                    root_dir=self.root_dir,
                    tile_record=tile_record,
                    slide=slide,
                    base_mpp=self.base_mpp,
                    output_size=self.image_size,
                    slide_name=slide_name,
                    slide_rois=slide_rois,
                    logger=self.logger,
                )

                target_one_hot = np.zeros(
                    (
                        self.tissue_segmentation_n_classes,
                        self.image_size,
                        self.image_size,
                    ),
                    dtype=np.uint8,
                )
                for class_idx in range(self.tissue_segmentation_n_classes):
                    target_one_hot[class_idx] = (target_mask == class_idx).astype(
                        np.uint8
                    )

                return {
                    "image": resized_image.astype(np.float32) / 255.0,
                    "target": target_one_hot,
                }
            case "tumor_classification":
                self.logger.error(
                    "tumor_classification is not implemented in TCGATileDataset."
                )
                raise NotImplementedError(
                    "tumor_classification is not implemented in TCGATileDataset."
                )
            case _:
                self.logger.error("Unsupported main task requested: %s", task_name)
                raise ValueError(f"Unsupported main task requested: {task_name}")

    def _build_task(
        self: _TileDataset,
        task_name: str,
        *,
        base_image: np.ndarray,
        tile_record: TileRecord,
        slide: OpenSlide,
        center_x: int,
        center_y: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        """Dispatch one task processor."""

        def _resize_image(image: np.ndarray, interpolation: int) -> np.ndarray:
            return cv2.resize(  # pylint: disable=no-member
                image,
                (self.image_size, self.image_size),
                interpolation=interpolation,
            )

        match task_name:
            case "tissue_segmentation" | "tumor_classification":
                return self._build_main_task(
                    task_name=task_name,
                    base_image=base_image,
                    tile_record=tile_record,
                    slide=slide,
                )
            case "hematoxylin":
                task_output = process_hematoxylin_task(
                    base_image=base_image, logger=self.logger
                )
                task_output["image"] = _resize_image(
                    task_output["image"],
                    interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
                )
                task_output["target"] = _resize_image(
                    task_output["target"],
                    interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
                )
                return task_output
            case "magnification":
                if not self.magnification_mpps:
                    self.logger.error(
                        "magnification_mpps must be set for magnification task."
                    )
                    raise ValueError(
                        "magnification_mpps must be set for magnification task."
                    )
                task_output = process_magnification_task(
                    slide=slide,
                    center_x=center_x,
                    center_y=center_y,
                    output_size=self.tile_size,
                    target_mpps=self.magnification_mpps,
                    rng=rng,
                    logger=self.logger,
                )
                task_output["image"] = _resize_image(
                    task_output["image"],
                    interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
                )

                return task_output

            case "jigmag":
                if not self.jigmag_mpps:
                    self.logger.error("jigmag_mpps must be set for jigmag task.")
                    raise ValueError("jigmag_mpps must be set for jigmag task.")
                task_output = process_jigmag_task(
                    slide=slide,
                    center_x=center_x,
                    center_y=center_y,
                    output_size=self.tile_size,
                    target_mpps=self.jigmag_mpps,
                    rng=rng,
                    logger=self.logger,
                )
                task_output["image"] = _resize_image(
                    task_output["image"],
                    interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
                )
                return task_output
            case _:
                self.logger.error("Unsupported task requested: %s", task_name)
                raise ValueError(f"Unsupported task requested: {task_name}")

    def _tensorize_task_output(
        self: _TileDataset,
        task_name: str,
        task_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert task outputs into default-collate-friendly tensor structures."""
        tensorized: dict[str, Any] = {}
        for key, value in task_output.items():
            if key == "image":
                tensorized[key] = as_image_tensor(value, self.logger)
                continue
            if key == "target":
                tensorized[key] = self._tensorize_target(task_name, value)
                continue
            if isinstance(value, np.ndarray):
                tensorized[key] = torch.from_numpy(np.ascontiguousarray(value))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                tensorized[key] = torch.tensor(value)
            elif (
                isinstance(value, (tuple, list))
                and value
                and all(
                    isinstance(item, (int, float, np.integer, np.floating))
                    for item in value
                )
            ):
                tensorized[key] = torch.tensor(value)
            else:
                tensorized[key] = value
        return tensorized

    def _tensorize_target(self: _TileDataset, task_name: str, value: Any) -> Any:
        """Tensorize a task target according to the task type."""
        match task_name:
            case "hematoxylin" | "tissue_segmentation":
                return as_mask_tensor(value, self.logger)
            case "magnification":
                if not self.magnification_mpps:
                    self.logger.error(
                        "magnification_mpps must be set for magnification task."
                    )
                    raise ValueError(
                        "magnification_mpps must be set for magnification task."
                    )
                one_hot_target = torch.zeros(
                    len(self.magnification_mpps), dtype=torch.float32
                )
                one_hot_target[int(value)] = 1.0
                return one_hot_target
            case "jigmag":
                if not self.jigmag_mpps:
                    self.logger.error("jigmag_mpps must be set for jigmag task.")
                    raise ValueError("jigmag_mpps must be set for jigmag task.")
                one_hot_target = torch.zeros(4 * 3 * 2 * 1, dtype=torch.float32)
                one_hot_target[int(value)] = 1.0
                return one_hot_target
            case _:
                self.logger.warning(
                    "No tensorization rules defined for task %s, returning target as-is.",
                    task_name,
                )
                return value


class TCGATileDataset(DatasetABC):
    """Lightning datamodule that builds multi-task TCGA tile datasets."""

    SUPPORTED_TASKS = SUPPORTED_TASKS

    def __init__(
        self: TCGATileDataset,
        root_dir: str,
        *,
        tasks: Sequence[str],
        manifest_path: str | None = None,
        ordered_data_dir: str | None = None,
        tile_size: int = 512,
        image_size: int = 256,
        base_mpp: float = 0.25,
        magnification_mpps: Sequence[float] | None = None,
        jigmag_mpps: Sequence[float] | None = None,
        tiles_per_slide: int = 64,
        min_tissue_fraction: float = 0.5,
        thumbnail_max_size: int = 1024,
        white_threshold: float = 0.85,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        train_budget: float = 0.4,
        val_budget: float = 0.2,
        test_budget: float = 0.4,
        random_seed: int = 42,
        max_slides: int | None = None,
        slide_cache_size: int | None = None,
        task_transforms: (
            dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None
        ) = None,
        sample_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
        batch_size: int = 32,
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
            collate_fn=collate_fn,
        )

        self.logger = logger or _setup_logger_for_module()

        normalized_tasks = tuple(tasks)

        if not normalized_tasks:
            self.logger.error("tasks must contain at least one task name.")
            raise ValueError("tasks must contain at least one task name.")

        unknown_tasks = set(normalized_tasks).difference(self.SUPPORTED_TASKS)

        if unknown_tasks:
            self.logger.error("Unsupported tasks requested: %s", sorted(unknown_tasks))
            raise ValueError(f"Unsupported tasks requested: {sorted(unknown_tasks)}")
        if tile_size <= 0:
            self.logger.error("tile_size must be a positive integer.")
            raise ValueError("tile_size must be a positive integer.")
        if base_mpp <= 0:
            self.logger.error("base_mpp must be positive.")
            raise ValueError("base_mpp must be positive.")
        if magnification_mpps is not None and any(
            mpp <= 0 for mpp in magnification_mpps
        ):
            self.logger.error("magnification_mpps must contain only positive values.")
            raise ValueError("magnification_mpps must contain only positive values.")
        if jigmag_mpps is not None and any(mpp <= 0 for mpp in jigmag_mpps):
            self.logger.error("jigmag_mpps must contain only positive values.")
            raise ValueError("jigmag_mpps must contain only positive values.")
        if not magnification_mpps and "magnification" in normalized_tasks:
            raise ValueError(
                "magnification_mpps must contain at least one value for magnification."
            )
        if "jigmag" in normalized_tasks and (
            jigmag_mpps is None or len(jigmag_mpps) != 4
        ):
            self.logger.error("jigmag_mpps must contain exactly 4 values.")
            raise ValueError("jigmag_mpps must contain exactly 4 values.")
        if tiles_per_slide <= 0:
            self.logger.error("tiles_per_slide must be a positive integer.")
            raise ValueError("tiles_per_slide must be a positive integer.")
        if not 0 <= min_tissue_fraction <= 1:
            self.logger.error("min_tissue_fraction must be between 0 and 1.")
            raise ValueError("min_tissue_fraction must be between 0 and 1.")
        if not 0 <= white_threshold <= 1:
            self.logger.error("white_threshold must be between 0 and 1.")
            raise ValueError("white_threshold must be between 0 and 1.")
        if max_slides is not None and max_slides <= 0:
            self.logger.error("max_slides must be a positive integer or None.")
            raise ValueError("max_slides must be a positive integer or None.")
        if slide_cache_size is not None and slide_cache_size <= 0:
            self.logger.error("slide_cache_size must be a positive integer or None.")
            raise ValueError("slide_cache_size must be a positive integer or None.")
        split_sum = train_fraction + val_fraction + test_fraction
        if not np.isclose(split_sum, 1.0):
            self.logger.error(
                "train_fraction + val_fraction + test_fraction must sum to 1."
            )
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1."
            )
        budget_sum = train_budget + val_budget + test_budget
        if not np.isclose(budget_sum, 1.0):
            self.logger.error("train_budget + val_budget + test_budget must sum to 1.")
            raise ValueError("train_budget + val_budget + test_budget must sum to 1.")

        self.root_dir = root_dir
        self.tasks = normalized_tasks
        self.manifest_path = manifest_path
        self.ordered_data_dir = (
            ordered_data_dir
            if ordered_data_dir is not None
            else os.path.join(root_dir, "ordered_data")
        )
        self.tile_size = tile_size
        self.image_size = image_size
        self.base_mpp = float(base_mpp)
        self.magnification_mpps = (
            tuple(float(mpp) for mpp in magnification_mpps)
            if magnification_mpps is not None
            else ()
        )
        self.jigmag_mpps = (
            tuple(float(mpp) for mpp in jigmag_mpps) if jigmag_mpps is not None else ()
        )
        self.tiles_per_slide = tiles_per_slide
        self.min_tissue_fraction = min_tissue_fraction
        self.thumbnail_max_size = thumbnail_max_size
        self.white_threshold = white_threshold
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.train_budget = train_budget
        self.val_budget = val_budget
        self.test_budget = test_budget
        self.random_seed = random_seed
        self.max_slides = max_slides
        if slide_cache_size is None:
            effective_workers = max(num_workers, 1)
            effective_prefetch = prefetch_factor if prefetch_factor is not None else 2
            slide_cache_size = effective_workers * effective_prefetch * batch_size
        self.slide_cache_size = slide_cache_size
        self.task_transforms = dict(task_transforms or {})
        self.sample_transform = sample_transform
        self._resolved_manifest_path: str | None = None
        self._slide_splits: dict[str, list[SlideRecord]] | None = None
        self._bcss_gt_codes: dict[str, str] | None = None
        self._bcss_slide_metadata: pd.DataFrame | None = None
        self._bcss_roi_metadata: pd.DataFrame | None = None
        self._bcss_roi_groups: dict[str, pd.DataFrame] | None = None
        self._labeled_submitter_ids: set[str] | None = None
        self._labeled_slide_paths: set[str] | None = None

    @staticmethod
    def from_config(config: dict[str, Any]) -> TCGATileDataset:
        """Create a TCGATileDataset from a config dict."""
        # Validate config keys
        root_dir = config.get("root_dir", None)
        if root_dir is None:
            raise ValueError(
                "root_dir is required in config to create TCGATileDataset."
            )
        if not isinstance(root_dir, str):
            raise ValueError("root_dir must be a string in config.")

        tasks = config.get("tasks", None)
        if tasks is None:
            raise ValueError("tasks is required in config to create TCGATileDataset.")
        if not isinstance(tasks, (list, tuple)) or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError("tasks must be a list of strings in config.")
        if not set(tasks).issubset(TCGATileDataset.SUPPORTED_TASKS):
            raise ValueError(
                f"tasks must be a subset of {TCGATileDataset.SUPPORTED_TASKS} in config."
            )

        manifest_path = config.get("manifest_path", None)
        if manifest_path is not None and not isinstance(manifest_path, str):
            raise ValueError("manifest_path must be a string or None in config.")
        ordered_data_dir = config.get("ordered_data_dir", None)
        if ordered_data_dir is not None and not isinstance(ordered_data_dir, str):
            raise ValueError("ordered_data_dir must be a string or None in config.")

        tile_size = config.get("tile_size", 512)
        if not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer in config.")
        image_size = config.get("image_size", 256)
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer in config.")
        base_mpp = config.get("base_mpp", 0.25)
        if not isinstance(base_mpp, (int, float)) or base_mpp <= 0:
            raise ValueError("base_mpp must be a positive number in config.")

        magnification_mpps = config.get("magnification_mpps", None)
        if magnification_mpps is not None and (
            not isinstance(magnification_mpps, (list, tuple))
            or not all(
                isinstance(mpp, (int, float)) and mpp > 0 for mpp in magnification_mpps
            )
        ):
            raise ValueError(
                "magnification_mpps must be a list of positive numbers in config."
            )
        if "magnification" in tasks and not magnification_mpps:
            raise ValueError(
                "magnification_mpps must contain at least one value for magnification task in config."
            )

        jigmag_mpps = config.get("jigmag_mpps", None)
        if jigmag_mpps is not None and (
            not isinstance(jigmag_mpps, (list, tuple))
            or not all(isinstance(mpp, (int, float)) and mpp > 0 for mpp in jigmag_mpps)
        ):
            raise ValueError(
                "jigmag_mpps must be a list of positive numbers in config."
            )
        if "jigmag" in tasks and (not jigmag_mpps or len(jigmag_mpps) != 4):
            raise ValueError(
                "jigmag_mpps must contain exactly 4 values for jigmag task in config."
            )

        tiles_per_slide = config.get("tiles_per_slide", 64)
        if not isinstance(tiles_per_slide, int) or tiles_per_slide <= 0:
            raise ValueError("tiles_per_slide must be a positive integer in config.")
        min_tissue_fraction = config.get("min_tissue_fraction", 0.5)
        if (
            not isinstance(min_tissue_fraction, (int, float))
            or not 0 <= min_tissue_fraction <= 1
        ):
            raise ValueError("min_tissue_fraction must be between 0 and 1 in config.")

        thumbnail_max_size = config.get("thumbnail_max_size", 1024)
        if not isinstance(thumbnail_max_size, int) or thumbnail_max_size <= 0:
            raise ValueError("thumbnail_max_size must be a positive integer in config.")
        white_threshold = config.get("white_threshold", 0.85)
        if (
            not isinstance(white_threshold, (int, float))
            or not 0 <= white_threshold <= 1
        ):
            raise ValueError("white_threshold must be between 0 and 1 in config.")

        train_fraction = config.get("train_fraction", 0.8)
        val_fraction = config.get("val_fraction", 0.1)
        test_fraction = config.get("test_fraction", 0.1)
        if not all(
            isinstance(frac, (int, float)) and 0 <= frac <= 1
            for frac in (train_fraction, val_fraction, test_fraction)
        ):
            raise ValueError(
                "train_fraction, val_fraction, and test_fraction must be between 0 and 1 in config."
            )

        train_budget = config.get("train_budget", 0.4)
        val_budget = config.get("val_budget", 0.2)
        test_budget = config.get("test_budget", 0.4)
        if not all(
            isinstance(budget, (int, float)) and 0 <= budget <= 1
            for budget in (train_budget, val_budget, test_budget)
        ):
            raise ValueError(
                "train_budget, val_budget, and test_budget must be between 0 and 1 in config."
            )

        random_seed = config.get("random_seed", 42)
        if not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer in config.")

        max_slides = config.get("max_slides", None)
        if max_slides is not None and (
            not isinstance(max_slides, int) or max_slides <= 0
        ):
            raise ValueError("max_slides must be a positive integer or None in config.")

        slide_cache_size = config.get("slide_cache_size", None)
        if slide_cache_size is not None and (
            not isinstance(slide_cache_size, int) or slide_cache_size <= 0
        ):
            raise ValueError(
                "slide_cache_size must be a positive integer or None in config."
            )

        # task_transforms: (
        #     dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None
        # ) = None,
        # sample_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        # logger: logging.Logger | None = None,

        batch_size = config.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer in config.")
        val_batch_size = config.get("val_batch_size", None)
        if val_batch_size is not None and (
            not isinstance(val_batch_size, int) or val_batch_size <= 0
        ):
            raise ValueError(
                "val_batch_size must be a positive integer or None in config."
            )
        test_batch_size = config.get("test_batch_size", None)
        if test_batch_size is not None and (
            not isinstance(test_batch_size, int) or test_batch_size <= 0
        ):
            raise ValueError(
                "test_batch_size must be a positive integer or None in config."
            )
        predict_batch_size = config.get("predict_batch_size", None)
        if predict_batch_size is not None and (
            not isinstance(predict_batch_size, int) or predict_batch_size <= 0
        ):
            raise ValueError(
                "predict_batch_size must be a positive integer or None in config."
            )

        num_workers = config.get("num_workers", 0)
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer in config.")
        pin_memory = config.get("pin_memory", True)
        if not isinstance(pin_memory, bool):
            raise ValueError("pin_memory must be a boolean in config.")
        persistent_workers = config.get("persistent_workers", None)
        if persistent_workers is not None and not isinstance(persistent_workers, bool):
            raise ValueError("persistent_workers must be a boolean or None in config.")
        prefetch_factor = config.get("prefetch_factor", None)
        if prefetch_factor is not None and (
            not isinstance(prefetch_factor, int) or prefetch_factor <= 0
        ):
            raise ValueError(
                "prefetch_factor must be a positive integer or None in config."
            )
        shuffle_train = config.get("shuffle_train", True)
        if not isinstance(shuffle_train, bool):
            raise ValueError("shuffle_train must be a boolean in config.")
        drop_last_train = config.get("drop_last_train", False)
        if not isinstance(drop_last_train, bool):
            raise ValueError("drop_last_train must be a boolean in config.")

        # collate_fn: Callable[[list[Any]], Any] | None = None,

        return TCGATileDataset(
            root_dir=root_dir,
            tasks=tasks,
            manifest_path=manifest_path,
            ordered_data_dir=ordered_data_dir,
            tile_size=tile_size,
            image_size=image_size,
            base_mpp=base_mpp,
            magnification_mpps=magnification_mpps,
            jigmag_mpps=jigmag_mpps,
            tiles_per_slide=tiles_per_slide,
            min_tissue_fraction=min_tissue_fraction,
            thumbnail_max_size=thumbnail_max_size,
            white_threshold=white_threshold,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_budget=train_budget,
            val_budget=val_budget,
            test_budget=test_budget,
            random_seed=random_seed,
            max_slides=max_slides,
            slide_cache_size=slide_cache_size,
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
        )

    def prepare_data(self) -> None:
        """Validate manifest resolution and slide discovery on one process."""
        self.logger.info("Preparing TCGA tile data for tasks: %s", list(self.tasks))
        self._resolved_manifest_path = resolve_manifest_path(
            self.root_dir, self.manifest_path, self.logger
        )
        self._load_candidate_slide_records()
        if "tissue_segmentation" in self.tasks:
            self._load_bcss_metadata()
        self.logger.info("Finished prepare_data for tasks: %s", list(self.tasks))

    def setup(self: TCGATileDataset, stage: str | None = None) -> None:
        """Populate dataset splits requested by the current Lightning stage."""
        valid_stages = {None, "fit", "validate", "test", "predict"}
        if stage not in valid_stages:
            raise ValueError(f"Unsupported stage: {stage}")

        self.logger.info(
            "Starting setup(stage=%s) for tasks: %s", stage, list(self.tasks)
        )
        slide_splits = self._get_slide_splits()

        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_tile_dataset(slide_splits["train"])
            if self.val_dataset is None:
                self.val_dataset = self._build_tile_dataset(slide_splits["val"])
        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = self._build_tile_dataset(slide_splits["val"])
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self._build_tile_dataset(slide_splits["test"])
        if stage in (None, "predict") and self.predict_dataset is None:
            self.predict_dataset = self._build_tile_dataset(slide_splits["predict"])

        self.logger.info(
            "Finished setup(stage=%s): train=%d val=%d test=%d predict=%d",
            stage,
            0 if self.train_dataset is None else len(self.train_dataset),  # type: ignore[union-attr]
            0 if self.val_dataset is None else len(self.val_dataset),  # type: ignore[union-attr]
            0 if self.test_dataset is None else len(self.test_dataset),  # type: ignore[union-attr]
            0 if self.predict_dataset is None else len(self.predict_dataset),  # type: ignore[union-attr]
        )

    def teardown(self: TCGATileDataset, stage: str | None = None) -> None:
        """Close cached slide handles kept by split datasets."""
        del stage
        for dataset in (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.predict_dataset,
        ):
            if isinstance(dataset, _TileDataset):
                dataset.close()

    def _get_slide_splits(self) -> dict[str, list[SlideRecord]]:
        """Load slides and compute patient-level splits once."""
        if self._slide_splits is not None:
            return self._slide_splits

        candidate_slide_records = self._load_candidate_slide_records()
        labeled_submitter_ids = self._labeled_submitter_ids or set()
        labeled_slide_paths = self._labeled_slide_paths or set()
        if "tissue_segmentation" in self.tasks:
            split_records = split_slide_records_with_budget(
                candidate_slide_records,
                labeled_submitter_ids=labeled_submitter_ids,
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                train_budget=self.train_budget,
                val_budget=self.val_budget,
                test_budget=self.test_budget,
                seed=self.random_seed,
                logger=self.logger,
            )
        else:
            split_records = split_slide_records(
                candidate_slide_records,
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                seed=self.random_seed,
            )
        split_records["predict"] = list(candidate_slide_records)
        self._slide_splits = split_records

        for split_name in ("train", "val", "test"):
            split_slides = split_records[split_name]
            labeled_count = sum(
                1 for slide in split_slides if slide.slide_path in labeled_slide_paths
            )
            unlabeled_count = len(split_slides) - labeled_count
            self.logger.info(
                "Split %s: %d slide(s) (%d labelled, %d unlabelled).",
                split_name,
                len(split_slides),
                labeled_count,
                unlabeled_count,
            )

        return self._slide_splits

    def _infer_context_mpp(self) -> float:
        """Determine the largest field of view required by the selected tasks."""
        context_mpp = self.base_mpp
        if "magnification" in self.tasks and self.magnification_mpps:
            context_mpp = max(context_mpp, max(self.magnification_mpps))
        if "jigmag" in self.tasks and self.jigmag_mpps:
            context_mpp = max(context_mpp, max(self.jigmag_mpps))
        return float(context_mpp)

    def _build_tile_dataset(
        self: TCGATileDataset, slide_records: Sequence[SlideRecord]
    ) -> _TileDataset:
        """Sample base tiles for one split and wrap them in a map-style dataset."""
        records = sample_tile_records(
            slide_records,
            output_size=self.tile_size,
            base_mpp=self.base_mpp,
            context_mpp=self._infer_context_mpp(),
            tiles_per_slide=self.tiles_per_slide,
            min_tissue_fraction=self.min_tissue_fraction,
            thumbnail_max_size=self.thumbnail_max_size,
            white_threshold=self.white_threshold,
            seed=self.random_seed,
            logger=self.logger,
        )

        tissue_segmentation_n_classes: int | None = None
        bcss_roi_groups: dict[str, pd.DataFrame] | None = None
        if "tissue_segmentation" in self.tasks:
            if not self._bcss_gt_codes:
                self.logger.error(
                    "BCSS metadata could not be loaded, tissue segmentation requires it."
                )
                raise RuntimeError(
                    "BCSS metadata could not be loaded, tissue segmentation requires it."
                )
            tissue_segmentation_n_classes = len(self._bcss_gt_codes)
            bcss_roi_groups = self._get_bcss_roi_groups()

        return _TileDataset(
            records=records,
            tasks=self.tasks,
            tile_size=self.tile_size,
            image_size=self.image_size,
            base_mpp=self.base_mpp,
            root_dir=self.root_dir,
            magnification_mpps=self.magnification_mpps,
            jigmag_mpps=self.jigmag_mpps,
            random_seed=self.random_seed,
            task_transforms=self.task_transforms,
            sample_transform=self.sample_transform,
            logger=self.logger,
            tissue_segmentation_n_classes=tissue_segmentation_n_classes,
            bcss_roi_groups=bcss_roi_groups,
            slide_cache_size=self.slide_cache_size,
        )

    def _get_bcss_roi_groups(self) -> dict[str, pd.DataFrame]:
        """Group BCSS ROI rows by slide_name for per-tile mask compositing."""
        if self._bcss_roi_groups is not None:
            return self._bcss_roi_groups
        _, _, roi_df = self._load_bcss_metadata()
        self._bcss_roi_groups = {
            str(slide_name): group.reset_index(drop=True)
            for slide_name, group in roi_df.groupby("slide_name", sort=False)
        }
        return self._bcss_roi_groups

    def _load_candidate_slide_records(self) -> list[SlideRecord]:
        """Load slide records and, if needed, restrict them to BCSS-labelled slides."""
        if self._resolved_manifest_path is None:
            self._resolved_manifest_path = resolve_manifest_path(
                self.root_dir, self.manifest_path, self.logger
            )

        all_slide_records = load_slide_records(
            manifest_path=self._resolved_manifest_path,
            ordered_data_dir=self.ordered_data_dir,
            max_slides=None,
        )
        self.logger.info(
            "Loaded %d slide record(s) from manifest %s.",
            len(all_slide_records),
            self._resolved_manifest_path,
        )

        if self.max_slides is not None:
            all_slide_records = all_slide_records[: self.max_slides]

        if "tissue_segmentation" in self.tasks:
            labeled_submitter_ids, labeled_slide_paths = (
                self._compute_labeled_submitter_ids(all_slide_records)
            )
            self._labeled_submitter_ids = labeled_submitter_ids
            self._labeled_slide_paths = labeled_slide_paths
            if not labeled_submitter_ids:
                self.logger.error(
                    "No BCSS-labelled slides from bcss_slide_magnifications.csv were found in the manifest/ordered_data."
                )
                raise RuntimeError(
                    "No BCSS-labelled slides from bcss_slide_magnifications.csv were found in the manifest/ordered_data."
                )

        return all_slide_records

    def _load_bcss_metadata(self) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
        """Load BCSS slide metadata and ROI bounds from ``root_dir/metadata``."""
        if (
            self._bcss_gt_codes is not None
            and self._bcss_slide_metadata is not None
            and self._bcss_roi_metadata is not None
        ):
            return (
                self._bcss_gt_codes,
                self._bcss_slide_metadata,
                self._bcss_roi_metadata,
            )

        gt_codes_path, roi_bounds_path, slide_metadata_path = (
            resolve_tissue_label_metadata_path(
                root_dir=self.root_dir, logger=self.logger
            )
        )

        if not os.path.exists(slide_metadata_path):
            raise FileNotFoundError(
                f"BCSS slide metadata file not found: {slide_metadata_path}"
            )
        if not os.path.exists(roi_bounds_path):
            raise FileNotFoundError(
                f"BCSS ROI bounds file not found: {roi_bounds_path}"
            )
        if not os.path.exists(gt_codes_path):
            raise FileNotFoundError(
                f"BCSS ground truth codes file not found: {gt_codes_path}"
            )

        gt_codes_df = pd.read_csv(gt_codes_path, sep="\t", dtype=str)
        gt_codes_df = gt_codes_df.rename(
            columns={"label": "label", "GT_code": "gt_code"}
        )
        self._bcss_gt_codes = dict(zip(gt_codes_df["label"], gt_codes_df["gt_code"]))

        slide_df = pd.read_csv(slide_metadata_path, dtype=str)
        slide_df = slide_df.rename(
            columns={
                slide_df.columns[0]: "submitter_id",
                "name": "filename",
                "_id": "bcss_item_id",
            }
        )
        required_slide_columns = {"submitter_id", "filename", "magnification"}
        missing_slide_columns = required_slide_columns.difference(slide_df.columns)
        if missing_slide_columns:
            raise ValueError(
                "BCSS slide metadata is missing required columns: "
                f"{sorted(missing_slide_columns)}"
            )

        slide_df["submitter_id"] = slide_df["submitter_id"].astype(str).str.strip()
        slide_df["filename"] = slide_df["filename"].astype(str).str.strip()
        slide_df["slide_name"] = slide_df.apply(
            lambda row: derive_bcss_slide_name(row["filename"], row["submitter_id"]),
            axis=1,
        )

        roi_df = pd.read_csv(roi_bounds_path, dtype=str)
        roi_df = roi_df.rename(columns={roi_df.columns[0]: "slide_name"})
        required_roi_columns = {"slide_name", "xmin", "ymin", "xmax", "ymax"}
        missing_roi_columns = required_roi_columns.difference(roi_df.columns)
        if missing_roi_columns:
            raise ValueError(
                f"BCSS ROI bounds file is missing required columns: {sorted(missing_roi_columns)}"
            )

        roi_df["slide_name"] = roi_df["slide_name"].astype(str).str.strip()
        for column in ("xmin", "ymin", "xmax", "ymax"):
            roi_df[column] = pd.to_numeric(roi_df[column], errors="coerce")
        if roi_df[["xmin", "ymin", "xmax", "ymax"]].isna().any().any():
            raise ValueError("BCSS ROI bounds file contains invalid coordinates.")

        self.logger.info(
            "Loaded BCSS metadata: %d slide rows, %d ROI rows, %d GT codes.",
            len(slide_df),
            len(roi_df),
            len(self._bcss_gt_codes),
        )
        self._bcss_slide_metadata = slide_df
        self._bcss_roi_metadata = roi_df
        return self._bcss_gt_codes, self._bcss_slide_metadata, self._bcss_roi_metadata

    def _compute_labeled_submitter_ids(
        self: TCGATileDataset, slide_records: Sequence[SlideRecord]
    ) -> tuple[set[str], set[str]]:
        """Return labelled submitter IDs and slide paths matched in BCSS metadata."""
        _, slide_df, _ = self._load_bcss_metadata()
        expected_submitter_by_filename = dict(
            zip(slide_df["filename"], slide_df["submitter_id"])
        )

        labeled_submitter_ids: set[str] = set()
        labeled_slide_paths: set[str] = set()
        for slide_record in slide_records:
            filename = os.path.basename(slide_record.slide_path)
            expected_submitter_id = expected_submitter_by_filename.get(filename)
            if expected_submitter_id == slide_record.submitter_id:
                labeled_submitter_ids.add(slide_record.submitter_id)
                labeled_slide_paths.add(slide_record.slide_path)

        self.logger.info(
            "Matched %d BCSS-labelled slide(s) from %d manifest slide(s) "
            "(%d unique labelled submitter(s)).",
            len(labeled_slide_paths),
            len(slide_records),
            len(labeled_submitter_ids),
        )
        return labeled_submitter_ids, labeled_slide_paths

