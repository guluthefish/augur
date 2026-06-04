"""Attention-based Deep Multiple Instance Learning aggregators (Ilse et al., 2018)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from augur.models.model_abc import ModelABC


class Attention(ModelABC):
    """
    Attention-based Deep MIL pooling (Ilse et al., 2018).

    Given a bag of ``K`` instance embeddings per slide ``h_1, ..., h_K`` of
    dimensionality ``D``, computes attention weights
    ``a_k = exp(w^T tanh(V h_k)) / Z`` and returns the attention-weighted sum
    ``z = sum_k a_k h_k``. When ``num_heads > 1``, independent attention
    distributions are learned and their per-head pooled representations are
    concatenated along the feature dimension.

    Parameters
    ----------
    input_dim
        Instance embedding dimensionality ``D``. If omitted, the first linear
        layer is lazily initialized on the first forward pass.
    hidden_dim
        Attention hidden dimensionality ``L`` in the original paper.
    num_heads
        Number of independent attention distributions (branches).
    dropout
        Dropout rate applied after the attention non-linearity.
    optimizer_factory
        An optional factory function for constructing the optimizer.
    optimizer_kwargs
        Keyword arguments forwarded to ``optimizer_factory``.
    lr_scheduler_factory
        Optional callable that builds a learning-rate scheduler from the optimizer
        returned by ``optimizer_factory``.
    lr_scheduler_kwargs
        Keyword arguments forwarded to ``lr_scheduler_factory``.
    lr_scheduler_config
        Optional configuration for the learning-rate scheduler.
    """

    def __init__(
        self: Attention,
        input_dim: int | None,
        hidden_dim: int,
        *,
        num_heads: int = 1,
        dropout: float = 0.0,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

        _validate_attention_args(input_dim, hidden_dim, num_heads, dropout)

        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.V = (
            nn.LazyLinear(self.hidden_dim)
            if self.input_dim is None
            else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.attn_dropout = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()
        )
        self.w = nn.Linear(self.hidden_dim, self.num_heads, bias=False)

    @staticmethod
    def from_config(config: dict[str, Any]) -> Attention:
        # Attention is a sub-component of DualCLAM / EmbeddingMIL; its
        # optimizer config is owned by the parent and is ignored here.
        input_dim, hidden_dim, num_heads, dropout = _parse_attention_config(config)
        return Attention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(  # pylint: disable=arguments-differ
        self: Attention,
        h: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate a bag of instance embeddings by attention pooling.

        Parameters
        ----------
        h
            Bag tensor of shape ``(B, K, D)``.
        mask
            Optional validity mask of shape ``(B, K)`` with ``1`` for valid
            instances and ``0`` for padded positions.

        Returns
        -------
        aggregated
            Pooled embedding of shape ``(B, num_heads * D)``.
        attention_weights
            Normalized attention weights of shape ``(B, num_heads, K)``.
        """
        if h.ndim != 3:
            raise ValueError(
                f"Expected h to have shape (B, K, D). Got: {tuple(h.shape)}"
            )
        if self.input_dim is None:
            self.input_dim = int(h.shape[-1])

        logits = self.w(self.attn_dropout(torch.tanh(self.V(h))))
        return _attention_pool(h, logits, mask)

    def model_step(
        self: Attention, batch: Any, batch_idx: int, stage: str
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "Attention is an aggregator-only Lightning module. Override "
            "model_step() in a task-specific module to compute a loss."
        )

    def configure_optimizers(self: Attention) -> None:
        """Sub-component — optimizer is owned by the parent DualCLAM/EmbeddingMIL."""
        raise NotImplementedError(
            "Attention is a pooling sub-component, not a top-level Lightning "
            "module. Its optimizer is configured by the parent slide-level "
            "model (DualCLAM / EmbeddingMIL); train the parent module instead "
            "of this one directly."
        )


class GatedAttention(ModelABC):
    """
    Gated Attention-based Deep MIL pooling (Ilse et al., 2018).

    Replaces the ``tanh(V h)`` non-linearity of :class:`Attention` with the
    gated formulation ``tanh(V h) * sigmoid(U h)`` to increase attention
    expressiveness while preserving the same pooling interface.

    Parameters
    ----------
    input_dim
        Instance embedding dimensionality ``D``. If omitted, the first linear
        layer is lazily initialized on the first forward pass.
    hidden_dim
        Attention hidden dimensionality ``L`` in the original paper.
    num_heads
        Number of independent attention distributions (branches).
    dropout
        Dropout rate applied after the gated attention non-linearity.
    optimizer_factory
        An optional factory function for constructing the optimizer.
    optimizer_kwargs
        Keyword arguments forwarded to ``optimizer_factory``.
    lr_scheduler_factory
        Optional callable that builds a learning-rate scheduler from the optimizer
        returned by ``optimizer_factory``.
    lr_scheduler_kwargs
        Keyword arguments forwarded to ``lr_scheduler_factory``.
    lr_scheduler_config
        Optional configuration for the learning-rate scheduler.
    """

    def __init__(
        self: GatedAttention,
        input_dim: int | None,
        hidden_dim: int,
        *,
        num_heads: int = 1,
        dropout: float = 0.0,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

        _validate_attention_args(input_dim, hidden_dim, num_heads, dropout)

        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.V = (
            nn.LazyLinear(self.hidden_dim)
            if self.input_dim is None
            else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.U = (
            nn.LazyLinear(self.hidden_dim)
            if self.input_dim is None
            else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.attn_dropout = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()
        )
        self.w = nn.Linear(self.hidden_dim, self.num_heads, bias=False)

    @staticmethod
    def from_config(config: dict[str, Any]) -> GatedAttention:
        # GatedAttention is a sub-component of DualCLAM / EmbeddingMIL; its
        # optimizer config is owned by the parent and is ignored here.
        input_dim, hidden_dim, num_heads, dropout = _parse_attention_config(config)
        return GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(  # pylint: disable=arguments-differ
        self: GatedAttention,
        h: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate a bag of instance embeddings by gated attention pooling.

        Parameters
        ----------
        h
            Bag tensor of shape ``(B, K, D)``.
        mask
            Optional validity mask of shape ``(B, K)`` with ``1`` for valid
            instances and ``0`` for padded positions.

        Returns
        -------
        aggregated
            Pooled embedding of shape ``(B, num_heads * D)``.
        attention_weights
            Normalized attention weights of shape ``(B, num_heads, K)``.
        """
        if h.ndim != 3:
            raise ValueError(
                f"Expected h to have shape (B, K, D). Got: {tuple(h.shape)}"
            )
        if self.input_dim is None:
            self.input_dim = int(h.shape[-1])

        gated = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        logits = self.w(self.attn_dropout(gated))
        return _attention_pool(h, logits, mask)

    def model_step(
        self: GatedAttention, batch: Any, batch_idx: int, stage: str
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "GatedAttention is an aggregator-only Lightning module. Override "
            "model_step() in a task-specific module to compute a loss."
        )

    def configure_optimizers(self: GatedAttention) -> None:
        """Sub-component — optimizer is owned by the parent DualCLAM/EmbeddingMIL."""
        raise NotImplementedError(
            "GatedAttention is a pooling sub-component, not a top-level "
            "Lightning module. Its optimizer is configured by the parent "
            "slide-level model (DualCLAM / EmbeddingMIL); train the parent "
            "module instead of this one directly."
        )


def _validate_attention_args(
    input_dim: int | None,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
) -> None:
    if input_dim is not None and (not isinstance(input_dim, int) or input_dim <= 0):
        raise ValueError(
            f"input_dim must be a positive integer or None. Got: {input_dim}"
        )
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be a positive integer. Got: {hidden_dim}")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(f"num_heads must be a positive integer. Got: {num_heads}")
    if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
        raise ValueError(
            f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
        )


def _parse_attention_config(
    config: dict[str, Any],
) -> tuple[int | None, int, int, float]:
    input_dim = config.get("input_dim", None)
    if input_dim is not None:
        assert (
            isinstance(input_dim, int) and input_dim > 0
        ), "input_dim must be a positive integer when provided"
    hidden_dim = config.get("hidden_dim", None)
    assert hidden_dim is not None, "hidden_dim must be specified in the config"
    assert (
        isinstance(hidden_dim, int) and hidden_dim > 0
    ), "hidden_dim must be a positive integer"
    num_heads = config.get("num_heads", 1)
    assert (
        isinstance(num_heads, int) and num_heads > 0
    ), "num_heads must be a positive integer"
    dropout = config.get("dropout", 0.0)
    assert (
        isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0
    ), "dropout must be a float in the range [0.0, 1.0)"
    return input_dim, hidden_dim, num_heads, float(dropout)


def _attention_pool(
    h: Tensor,
    logits: Tensor,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Softmax-normalize attention logits over instances and pool ``h`` per head."""
    if mask is not None:
        if mask.shape != h.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match bag shape "
                f"{tuple(h.shape[:2])}."
            )
        logits = logits.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))

    attn = torch.softmax(logits, dim=1)
    aggregated = torch.einsum("bkm,bkd->bmd", attn, h).flatten(start_dim=1)
    attn_weights = attn.transpose(1, 2).contiguous()
    return aggregated, attn_weights
