"""Factory for creating slide-level modules from configuration."""

from typing import Any

from VexDR.models.model_abc import ModelABC
from VexDR.models.slide_level.dual_clam import DualCLAM
from VexDR.models.slide_level.mil import EmbeddingMIL


def get_module_from_config(config: dict[str, Any]) -> ModelABC:
    """Factory function to create slide-level modules based on config."""
    module_name = config.get("name", None)
    match module_name:
        case "DualCLAM":
            return DualCLAM.from_config(config.get("params", {}))
        case "EmbeddingMIL":
            return EmbeddingMIL.from_config(config.get("params", {}))
        case _:
            raise ValueError(f"Unsupported slide-level module name: {module_name}")
