"""Unit tests for the shared Lightning model base class."""

from __future__ import annotations

from typing import Any

import torch

from augur.models.model_abc import ModelABC


class _DummyModel(ModelABC):
    """Minimal model used to exercise shared logging behavior."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def from_config(config: dict[str, Any]) -> _DummyModel:
        return _DummyModel()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def model_step(
        self,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del batch_idx, stage
        batch_size = batch["image"].shape[0]
        return torch.tensor(1.0), {
            "aux_metric": torch.tensor(float(batch_size)),
        }


def _test_shared_step_sync_dist_logging():
    """Epoch-level metrics should synchronize across ranks in distributed runs."""
    print("Testing ModelABC distributed metric logging...")
    model = _DummyModel()
    model._trainer = type("TrainerStub", (), {"world_size": 2})()  # type: ignore[attr-defined] pylint: disable=protected-access

    logged_calls: dict[str, Any] = {}

    def _capture_log(name: str, value: Any, **kwargs: Any) -> None:
        logged_calls["log"] = {"name": name, "value": value, "kwargs": kwargs}

    def _capture_log_dict(metrics: dict[str, Any], **kwargs: Any) -> None:
        logged_calls["log_dict"] = {"metrics": metrics, "kwargs": kwargs}

    model.log = _capture_log  # type: ignore[method-assign]
    model.log_dict = _capture_log_dict  # type: ignore[method-assign]

    batch = {"image": torch.randn(3, 3, 8, 8)}
    loss = model._shared_step(batch, batch_idx=0, stage="val", on_step=False)

    assert isinstance(loss, torch.Tensor), (
        "Expected _shared_step() to return a loss tensor."
    )
    assert logged_calls["log"]["kwargs"]["sync_dist"] is True, (
        "Loss logging should synchronize across ranks when world_size > 1."
    )
    assert logged_calls["log_dict"]["kwargs"]["sync_dist"] is True, (
        "Metric logging should synchronize across ranks when world_size > 1."
    )

    model._trainer = type("TrainerStub", (), {"world_size": 1})()  # type: ignore[attr-defined] pylint: disable=protected-access
    logged_calls.clear()
    model._shared_step(batch, batch_idx=0, stage="val", on_step=False)

    assert logged_calls["log"]["kwargs"]["sync_dist"] is False, (
        "Loss logging should stay local for single-process runs."
    )
    assert logged_calls["log_dict"]["kwargs"]["sync_dist"] is False, (
        "Metric logging should stay local for single-process runs."
    )
    print("[OK] ModelABC distributed metric logging test passed.")


def test_all_models_model_abc():
    """Run all ModelABC tests."""
    print("Running ModelABC tests...")
    _test_shared_step_sync_dist_logging()
    print("All ModelABC tests passed!")


if __name__ == "__main__":
    test_all_models_model_abc()
