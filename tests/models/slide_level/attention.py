"""Unit tests for slide-level attention aggregators."""

from __future__ import annotations

from typing import Any, Callable

import torch

from VexDR.models.model_abc import ModelABC
from VexDR.models.slide_level.attention import Attention, GatedAttention


def _assert_raises(
    expected_exception: type[BaseException],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Assert that a callable raises the expected exception type."""
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, got {type(exc).__name__}."
        ) from exc
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised.")


def _feature_bag(
    batch_size: int = 2,
    num_tiles: int = 4,
    input_dim: int = 3,
) -> torch.Tensor:
    """Create a deterministic feature bag shaped like encoded slide tiles."""
    values = torch.arange(
        batch_size * num_tiles * input_dim,
        dtype=torch.float32,
    )
    return values.view(batch_size, num_tiles, input_dim) / 10.0


def _assert_attention_forward_contract(
    model: Attention | GatedAttention,
    h: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Validate the common attention pooling contract."""
    aggregated, attention_weights = model(h, mask)

    expected_output_shape = (h.shape[0], model.num_heads * h.shape[-1])
    expected_attention_shape = (h.shape[0], model.num_heads, h.shape[1])

    assert aggregated.shape == expected_output_shape, (
        f"Expected aggregated shape {expected_output_shape}. "
        f"Got: {aggregated.shape}."
    )
    assert attention_weights.shape == expected_attention_shape, (
        f"Expected attention weight shape {expected_attention_shape}. "
        f"Got: {attention_weights.shape}."
    )

    expected_sums = torch.ones(h.shape[0], model.num_heads)
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        expected_sums,
        atol=1e-6,
    ), "Attention weights should sum to 1 across tiles for every bag and head."

    if mask is not None:
        expanded_mask = mask.bool().unsqueeze(1).expand_as(attention_weights)
        masked_weights = attention_weights.masked_select(~expanded_mask)
        assert torch.all(
            masked_weights == 0
        ), "Masked tiles should receive zero attention weight."

    expected_aggregated = torch.einsum(
        "bmk,bkd->bmd",
        attention_weights,
        h,
    ).flatten(start_dim=1)
    assert torch.allclose(
        aggregated,
        expected_aggregated,
        atol=1e-6,
    ), "Aggregated embeddings should match the weighted sum of tile features."


def _test_init() -> None:
    """Attention modules should inherit the shared base and store constructor args."""
    print("Testing attention aggregator initialization...")

    attention = Attention(
        input_dim=3,
        hidden_dim=5,
        num_heads=2,
        dropout=0.25,
        optimizer_factory=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1},
    )
    gated_attention = GatedAttention(
        input_dim=3,
        hidden_dim=5,
        num_heads=2,
        dropout=0.25,
        optimizer_factory=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )

    for model in (attention, gated_attention):
        assert isinstance(
            model, ModelABC
        ), f"Expected {model.__class__.__name__} to inherit ModelABC."
        assert model.input_dim == 3, f"Expected input_dim=3. Got: {model.input_dim}."
        assert (
            model.hidden_dim == 5
        ), f"Expected hidden_dim=5. Got: {model.hidden_dim}."
        assert model.num_heads == 2, f"Expected num_heads=2. Got: {model.num_heads}."
        assert model.dropout == 0.25, f"Expected dropout=0.25. Got: {model.dropout}."

    assert isinstance(
        attention.configure_optimizers(),
        torch.optim.SGD,
    ), "Expected Attention to configure an SGD optimizer."
    assert isinstance(
        gated_attention.configure_optimizers(),
        torch.optim.AdamW,
    ), "Expected GatedAttention to configure an AdamW optimizer."

    print("[OK] Attention aggregator initialization test passed.")


def _test_forward() -> None:
    """Attention modules should pool deterministic feature bags per head."""
    print("Testing attention aggregator forward passes...")

    h = _feature_bag(batch_size=2, num_tiles=4, input_dim=3)
    attention = Attention(input_dim=3, hidden_dim=5, num_heads=2)
    gated_attention = GatedAttention(input_dim=3, hidden_dim=5, num_heads=2)

    _assert_attention_forward_contract(attention, h)
    _assert_attention_forward_contract(gated_attention, h)

    print("[OK] Attention aggregator forward pass test passed.")


def _test_masked_forward() -> None:
    """Masked tiles should be ignored while valid tiles still normalize per head."""
    print("Testing attention aggregator masked forward passes...")

    h = _feature_bag(batch_size=2, num_tiles=4, input_dim=3)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
        ]
    )
    attention = Attention(input_dim=3, hidden_dim=5, num_heads=2)
    gated_attention = GatedAttention(input_dim=3, hidden_dim=5, num_heads=2)

    _assert_attention_forward_contract(attention, h, mask)
    _assert_attention_forward_contract(gated_attention, h, mask)

    print("[OK] Attention aggregator masked forward pass test passed.")


def _test_lazy_input_dim() -> None:
    """Lazy attention layers should infer input_dim on the first forward pass."""
    print("Testing attention aggregator lazy input dimension inference...")

    h = _feature_bag(batch_size=2, num_tiles=3, input_dim=4)
    attention = Attention(input_dim=None, hidden_dim=6, num_heads=2)
    gated_attention = GatedAttention(input_dim=None, hidden_dim=6, num_heads=2)

    _assert_attention_forward_contract(attention, h)
    _assert_attention_forward_contract(gated_attention, h)

    assert attention.input_dim == 4, (
        f"Attention should infer input_dim=4 from the feature bag. "
        f"Got: {attention.input_dim}."
    )
    assert gated_attention.input_dim == 4, (
        f"GatedAttention should infer input_dim=4 from the feature bag. "
        f"Got: {gated_attention.input_dim}."
    )

    print("[OK] Attention aggregator lazy input dimension test passed.")


def _test_from_config() -> None:
    """Config constructors should parse model and optimization settings."""
    print("Testing attention aggregator from_config() constructors...")

    config = {
        "input_dim": 3,
        "hidden_dim": 5,
        "num_heads": 2,
        "dropout": 0.1,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "params": {"lr": 5e-4, "weight_decay": 0.01},
        },
        "lr_scheduler": {
            "name": "torch.optim.lr_scheduler.StepLR",
            "params": {"step_size": 3, "gamma": 0.5},
            "config": {"interval": "epoch", "frequency": 1},
        },
    }

    attention = Attention.from_config(config)
    gated_attention = GatedAttention.from_config(config)

    for model in (attention, gated_attention):
        assert model.input_dim == 3, f"Expected input_dim=3. Got: {model.input_dim}."
        assert (
            model.hidden_dim == 5
        ), f"Expected hidden_dim=5. Got: {model.hidden_dim}."
        assert model.num_heads == 2, f"Expected num_heads=2. Got: {model.num_heads}."
        assert model.dropout == 0.1, f"Expected dropout=0.1. Got: {model.dropout}."
        assert (
            model.optimizer_factory is torch.optim.AdamW
        ), f"Expected AdamW optimizer factory. Got: {model.optimizer_factory}."
        assert model.optimizer_kwargs == {
            "lr": 5e-4,
            "weight_decay": 0.01,
        }, f"Expected AdamW kwargs from config. Got: {model.optimizer_kwargs}."
        assert (
            model.lr_scheduler_factory is torch.optim.lr_scheduler.StepLR
        ), f"Expected StepLR scheduler factory. Got: {model.lr_scheduler_factory}."
        assert model.lr_scheduler_kwargs == {
            "step_size": 3,
            "gamma": 0.5,
        }, f"Expected StepLR kwargs from config. Got: {model.lr_scheduler_kwargs}."
        assert model.lr_scheduler_config == {
            "interval": "epoch",
            "frequency": 1,
        }, f"Expected scheduler metadata from config. Got: {model.lr_scheduler_config}."

        optimizers = model.configure_optimizers()
        assert isinstance(
            optimizers["optimizer"],
            torch.optim.AdamW,
        ), f"Expected AdamW optimizer from config. Got: {type(optimizers['optimizer'])}."
        assert isinstance(
            optimizers["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.StepLR,
        ), (
            "Expected StepLR scheduler from config. "
            f"Got: {type(optimizers['lr_scheduler']['scheduler'])}."
        )

    print("[OK] Attention aggregator from_config() test passed.")


def _test_error_handling() -> None:
    """Invalid inputs should fail with explicit exceptions."""
    print("Testing attention aggregator error handling...")

    invalid_constructor_args = (
        {"input_dim": 0, "hidden_dim": 5},
        {"input_dim": 3, "hidden_dim": 0},
        {"input_dim": 3, "hidden_dim": 5, "num_heads": 0},
        {"input_dim": 3, "hidden_dim": 5, "dropout": 1.0},
    )
    for model_cls in (Attention, GatedAttention):
        for kwargs in invalid_constructor_args:
            _assert_raises(ValueError, model_cls, **kwargs)

        model = model_cls(input_dim=3, hidden_dim=5)
        _assert_raises(ValueError, model, torch.randn(4, 3))
        _assert_raises(
            ValueError,
            model,
            _feature_bag(batch_size=2, num_tiles=4, input_dim=3),
            torch.ones(2, 3, dtype=torch.bool),
        )
        _assert_raises(
            NotImplementedError,
            model.model_step,
            {"image": torch.randn(1, 4, 3)},
            0,
            "train",
        )

    print("[OK] Attention aggregator error handling test passed.")


def test_Attention() -> None:
    """Run all slide-level attention aggregator unit tests."""
    print("Running slide-level attention aggregator tests...")
    _test_init()
    _test_forward()
    _test_masked_forward()
    _test_lazy_input_dim()
    _test_from_config()
    _test_error_handling()
    print("All slide-level attention aggregator tests passed!")
