"""Common abstractions for Lightning datamodules used in VexDR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DatasetABC(LightningDataModule, ABC):
    """Abstract base class for Lightning datamodules.

    Subclasses are expected to assign split datasets during :meth:`setup`:

    - ``self.train_dataset``
    - ``self.val_dataset``
    - ``self.test_dataset``
    - ``self.predict_dataset`` (optional)
    """

    def __init__(
        self: DatasetABC,
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
        """Initialize shared datamodule configuration.

        Parameters
        ----------
        batch_size:
            Default batch size used for training and, unless overridden, for all
            other dataloaders.
        val_batch_size:
            Optional validation batch size. Falls back to ``batch_size``.
        test_batch_size:
            Optional test batch size. Falls back to ``batch_size``.
        predict_batch_size:
            Optional prediction batch size. Falls back to ``batch_size``.
        num_workers:
            Number of worker processes used by each ``DataLoader``.
        pin_memory:
            Whether dataloaders should pin memory.
        persistent_workers:
            Whether worker processes are kept alive between epochs. If omitted,
            it is enabled automatically whenever ``num_workers > 0``.
        prefetch_factor:
            Number of batches prefetched by each worker. Only applied when
            ``num_workers > 0``.
        shuffle_train:
            Whether the training dataloader should shuffle samples.
        drop_last_train:
            Whether to drop the last incomplete training batch.
        collate_fn:
            Optional custom collate function shared by all dataloaders.
        """
        super().__init__()

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if num_workers < 0:
            raise ValueError("num_workers must be greater than or equal to zero.")
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer or None.")
        if persistent_workers and num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers to be greater than zero."
            )

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.predict_batch_size = predict_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.prefetch_factor = prefetch_factor
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train
        self.collate_fn = collate_fn

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self.predict_dataset: Dataset[Any] | None = None

        self.save_hyperparameters(
            {
                "batch_size": batch_size,
                "val_batch_size": self.val_batch_size,
                "test_batch_size": self.test_batch_size,
                "predict_batch_size": self.predict_batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": prefetch_factor,
                "shuffle_train": shuffle_train,
                "drop_last_train": drop_last_train,
            }
        )

    def prepare_data(self: DatasetABC) -> None:
        """Optionally download or preprocess data on a single process."""

    @abstractmethod
    def setup(self: DatasetABC, stage: str | None = None) -> None:
        """Populate the dataset splits required for the requested stage."""

    def _build_dataloader(
        self: DatasetABC,
        dataset: Dataset[Any] | None,
        *,
        split_name: str,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader[Any]:
        """Create a ``DataLoader`` for a prepared dataset split."""
        if dataset is None:
            raise RuntimeError(
                f"{split_name} dataset has not been initialized. "
                "Make sure setup() assigns the split before requesting a dataloader."
            )

        dataloader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": drop_last,
        }

        if self.collate_fn is not None:
            dataloader_kwargs["collate_fn"] = self.collate_fn
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self: DatasetABC) -> DataLoader[Any]:
        """Return the training dataloader."""
        return self._build_dataloader(
            self.train_dataset,
            split_name="train",
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self: DatasetABC) -> DataLoader[Any]:
        """Return the validation dataloader."""
        return self._build_dataloader(
            self.val_dataset,
            split_name="validation",
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self: DatasetABC) -> DataLoader[Any]:
        """Return the testing dataloader."""
        return self._build_dataloader(
            self.test_dataset,
            split_name="test",
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self: DatasetABC) -> DataLoader[Any]:
        """Return the prediction dataloader."""
        return self._build_dataloader(
            self.predict_dataset,
            split_name="predict",
            batch_size=self.predict_batch_size,
            shuffle=False,
            drop_last=False,
        )
