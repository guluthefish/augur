"""Lightning-compatible MLP classifier head for tile-level predictions."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from augur.models.model_abc import ModelABC


class Classifier(ModelABC):
    """
    A simple MLP classifier head for tile-level predictions.

    Parameters
    ----------
    input_dim:
        The dimensionality of the input features (e.g., ViT token embeddings).
        If omitted, the first linear layer is lazily initialized on the first
        forward pass.
    hidden_dims:
        The dimensionality of the hidden layers in the MLP.
    output_dim:
        The number of output classes for classification.
    dropout:
        The dropout rate to apply between MLP layers.
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
        self: Classifier,
        input_dim: int | None,
        hidden_dims: list[int],
        output_dim: int,
        preproc: Callable[..., Tensor] | None = None,
        *,
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

        if input_dim is not None and (not isinstance(input_dim, int) or input_dim <= 0):
            raise ValueError(
                f"input_dim must be a positive integer or None. Got: {input_dim}"
            )
        if not isinstance(hidden_dims, list) or any(
            not isinstance(hidden_dim, int) or hidden_dim <= 0
            for hidden_dim in hidden_dims
        ):
            raise ValueError("hidden_dims must be a list of positive integers.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(
                f"output_dim must be a positive integer. Got: {output_dim}"
            )
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be a float in the range [0.0, 1.0). Got: {dropout}"
            )

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.dropout = float(dropout)
        self.preproc = preproc or nn.Identity()
        self.mlp = self._build_mlp()

    @staticmethod
    def from_config(config: dict[str, Any]) -> Classifier:
        input_dim = config.get("input_dim", None)
        if input_dim is not None:
            assert (
                isinstance(input_dim, int) and input_dim > 0
            ), "input_dim must be a positive integer when provided"
        hidden_dims = config.get("hidden_dims", None)
        assert hidden_dims is not None, "hidden_dims must be specified in the config"
        assert isinstance(hidden_dims, list) and all(
            isinstance(hd, int) and hd > 0 for hd in hidden_dims
        ), "hidden_dims must be a list of positive integers"
        output_dim = config.get("output_dim", None)
        assert output_dim is not None, "output_dim must be specified in the config"
        assert (
            isinstance(output_dim, int) and output_dim > 0
        ), "output_dim must be a positive integer"

        dropout = config.get("dropout", 0.0)
        assert (
            isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0
        ), "dropout must be a float in the range [0.0, 1.0)"

        preproc = config.get("preproc", None)
        if preproc is not None:
            assert isinstance(preproc, str), "preproc must be a string if specified"
            # For simplicity, we only support "maxpool" as a preproc option here.
            assert (
                preproc == "maxpool"
            ), "Only 'maxpool' is supported as a preproc option"
            preproc_layer = nn.AdaptiveMaxPool2d(1)
        else:
            preproc_layer = None

        # Classifier is a sub-component of TileModel; its optimizer config
        # is owned by the parent and is ignored here even if a stray
        # `optimizer` / `lr_scheduler` key shows up in the partial dict.

        return Classifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            preproc=preproc_layer,
        )

    def forward(  # pylint: disable=arguments-differ
        self: Classifier, x: Tensor
    ) -> Tensor:
        x = self.preproc(x)
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        if self.input_dim is None:
            self.input_dim = int(x.shape[1])
        x = self.mlp(x)
        return x

    def _build_mlp(self: Classifier) -> nn.Sequential:
        """Build the classifier head, lazily inferring the first input width if needed."""
        if self.hidden_dims:
            first_out_dim = self.hidden_dims[0]
        else:
            first_out_dim = self.output_dim

        layers: list[nn.Module] = [
            (
                nn.LazyLinear(first_out_dim)
                if self.input_dim is None
                else nn.Linear(self.input_dim, first_out_dim)
            )
        ]

        if not self.hidden_dims:
            return nn.Sequential(*layers)

        prev_dim = first_out_dim
        layers.append(nn.ReLU(inplace=True))
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))

        for hidden_dim in self.hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)

    def predict_step(  # pylint: disable=arguments-differ
        self: Classifier,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """Run a forward pass during prediction."""
        del batch_idx, dataloader_idx

        assert isinstance(
            batch, Tensor
        ), f"Expected batch to be a Tensor, got {type(batch)}"

        return self.forward(batch)

    def model_step(
        self: Classifier, batch: Any, batch_idx: int, stage: str
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "Classifier is a decoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def configure_optimizers(self: Classifier) -> None:
        """Sub-component — its optimizer is owned by the parent TileModel."""
        raise NotImplementedError(
            "Classifier is a decoder sub-component, not a top-level Lightning "
            "module. Its optimizer is configured by the parent TileModel; "
            "train the TileModel (or another top-level model that wraps "
            "Classifier) instead of this module directly."
        )
