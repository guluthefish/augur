"""Factory module to create dataset instances based on configuration."""

from typing import Any

from augur.datasets.dataset_abc import DatasetABC
from augur.datasets.tcga_feature_dataset import TCGAFeatureDataset
from augur.datasets.tcga_slide_dataset import TCGASlideDataset
from augur.datasets.tcga_tile_dataset import TCGATileDataset


def get_dataset_from_config(config: dict[str, Any]) -> DatasetABC:
    """Factory function to create dataset instances based on config."""

    dataset_name = config.get("name")
    match dataset_name:
        case "TCGATileDataset":
            return TCGATileDataset.from_config(config.get("params", {}))
        case "TCGASlideDataset":
            return TCGASlideDataset.from_config(config.get("params", {}))
        case "TCGAFeatureDataset":
            return TCGAFeatureDataset.from_config(config.get("params", {}))
        case _:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
