"""Unit test for training tile encoder."""

from __future__ import annotations

import logging
import os
import tempfile

import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import Dataset

from VexDR.datasets.dataset_abc import DatasetABC
from VexDR.models.tile_level.classifier import Classifier
from VexDR.models.tile_level.tile_model import TileModel
from VexDR.models.tile_level.unet_encoder import UNetEncoder
from VexDR.scripts.model_training.train_tile_encoder import (
    _initialize_lazy_modules_from_dataloader,
    _resolve_resume_checkpoint_path,
    train as train_tile_encoder,
)


class _WarmupDataset(Dataset):
    """Small synthetic dataset used to initialize lazy classifier heads."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.samples = []
        for _ in range(2):
            image = torch.randn(3, 32, 32, generator=generator)
            target = torch.zeros(4, dtype=torch.float32)
            target[torch.randint(0, 4, (1,), generator=generator).item()] = 1.0
            self.samples.append(
                {
                    "magnification": {
                        "image": image,
                        "target": target,
                    }
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class _WarmupDataModule(DatasetABC):
    """Minimal datamodule for lazy-module warmup tests."""

    def __init__(self) -> None:
        super().__init__(batch_size=2, num_workers=0, pin_memory=False)

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is None:
            self.train_dataset = _WarmupDataset()


def _test_resolve_resume_checkpoint_path():
    """Resume helper should auto-discover and validate Lightning checkpoints."""
    print("Testing resume checkpoint path resolution...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")

        assert (
            _resolve_resume_checkpoint_path({}, checkpoint_dir=checkpoint_dir) is None
        ), "Expected no resume checkpoint when last.ckpt is absent."

        with open(last_checkpoint, "wb") as handle:
            handle.write(b"checkpoint")

        assert _resolve_resume_checkpoint_path(
            {},
            checkpoint_dir=checkpoint_dir,
        ) == os.path.abspath(
            last_checkpoint
        ), "Expected auto-resume to pick up checkpoint_dir/last.ckpt."

        assert _resolve_resume_checkpoint_path(
            {"resume_from": "auto"},
            checkpoint_dir=checkpoint_dir,
        ) == os.path.abspath(
            last_checkpoint
        ), "Expected explicit auto-resume to use checkpoint_dir/last.ckpt."

        explicit_checkpoint = os.path.join(tmpdir, "manual.ckpt")
        with open(explicit_checkpoint, "wb") as handle:
            handle.write(b"manual")

        assert _resolve_resume_checkpoint_path(
            {"resume_from": explicit_checkpoint},
            checkpoint_dir=checkpoint_dir,
        ) == os.path.abspath(
            explicit_checkpoint
        ), "Expected explicit resume path to be preserved."

    print("[OK] resume checkpoint path resolution test passed.")


def test_initialize_lazy_modules_from_dataloader():
    """A warmup batch should materialize lazy classifier parameters."""
    print("Testing lazy-module warmup from the training dataloader...")

    model = TileModel(
        encoder=UNetEncoder(feature_channels=(8, 8, 16, 32, 64)),
        decoders={
            "magnification": Classifier(
                input_dim=None,
                hidden_dims=[16],
                output_dim=4,
                preproc=torch.nn.AdaptiveMaxPool2d(1),
            )
        },
    )
    datamodule = _WarmupDataModule()

    assert any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    ), "Expected the classifier head to start with uninitialized lazy parameters."

    _initialize_lazy_modules_from_dataloader(
        model,
        datamodule,
        logger=logging.getLogger(__name__),
    )

    assert not any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    ), "Expected the warmup batch to initialize all lazy parameters."

    warmup_batch = next(iter(datamodule.train_dataloader()))
    model.eval()
    with torch.no_grad():
        predictions = model(warmup_batch)

    assert "magnification" in predictions, "Expected a magnification prediction."
    assert predictions["magnification"].shape == (
        2,
        4,
    ), f"Expected batched classifier logits of shape (2, 4). Got: {predictions['magnification'].shape}"

    print("[OK] lazy-module warmup test passed.")


def test_train_tile_encoder():
    """Test the training loop for the tile encoder."""
    print("Testing tile encoder training loop...")
    trainer, tile_model, tile_dataset = train_tile_encoder(
        model_config_path="configs/model-resnet50-hematoxylin.yaml",
        dataset_config_path="configs/tile_dataset-TCGA-BRCA-test.yaml",
        training_config_path="configs/training-config.yaml",
    )

    dataloader = tile_dataset.train_dataloader()
    test_sample = dataloader.dataset[0]

    device = next(tile_model.parameters()).device
    batched_sample = {}
    for task_name, task_sample in test_sample.items():
        if not isinstance(task_sample, dict):
            continue
        image = task_sample.get("image")
        if isinstance(image, torch.Tensor):
            batched_sample[task_name] = {"image": image.unsqueeze(0).to(device)}

    tile_model.eval()
    with torch.no_grad():
        predictions = tile_model(batched_sample)

    assert isinstance(predictions, dict), "Predictions should be a dictionary."
    expected_keys = {"hematoxylin", "tissue_segmentation"}
    assert expected_keys.issubset(
        predictions.keys()
    ), f"Predictions should contain keys: {expected_keys}"

    for task_name in expected_keys:
        assert task_name in predictions, f"Missing prediction for task: {task_name}"
        prediction = predictions[task_name]
        assert isinstance(
            prediction, torch.Tensor
        ), f"Prediction for {task_name} should be a tensor."
        assert (
            prediction.ndim >= 3
        ), f"Prediction for {task_name} should have at least 3 dimensions (batch, channels, ...)."

    # Optionally visualize the input, target, and prediction for each task
    # import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # def _to_plot_array(tensor: torch.Tensor):
    #     tensor = tensor.detach().cpu()
    #     if tensor.ndim == 4:
    #         tensor = tensor[0]
    #     if tensor.ndim == 3 and tensor.shape[0] == 3:
    #         return tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    #     if tensor.ndim == 3 and tensor.shape[0] == 1:
    #         return tensor.squeeze(0).numpy()
    #     if tensor.ndim == 3:
    #         return tensor.argmax(dim=0).numpy()
    #     return tensor.numpy()

    # plot_tasks = list(predictions.keys())
    # figure, axes = plt.subplots(
    #     nrows=len(plot_tasks),
    #     ncols=3,
    #     figsize=(9, 3 * len(plot_tasks)),
    #     squeeze=False,
    # )

    # for row_index, task_name in enumerate(plot_tasks):
    #     task_sample = test_sample[task_name]
    #     input_image = _to_plot_array(task_sample["image"])
    #     target = _to_plot_array(task_sample["target"])
    #     prediction = _to_plot_array(predictions[task_name])

    #     axes[row_index][0].imshow(input_image)
    #     axes[row_index][0].set_title(f"{task_name} input")
    #     axes[row_index][1].imshow(target)
    #     axes[row_index][1].set_title(f"{task_name} target")
    #     axes[row_index][2].imshow(prediction)
    #     axes[row_index][2].set_title(f"{task_name} prediction")

    #     for axis in axes[row_index]:
    #         axis.axis("off")

    # figure.tight_layout()
    # plt.show()
    # plt.close(figure)

    print("[OK] train_tile_encoder test passed.")


if __name__ == "__main__":
    _test_resolve_resume_checkpoint_path()
    test_initialize_lazy_modules_from_dataloader()
    test_train_tile_encoder()
