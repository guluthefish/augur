"""Lightning-compatible ViT encoder built on top of timm."""

from __future__ import annotations

from typing import Any, Callable

from dotenv import load_dotenv
import timm
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from VexDR.models.model_abc import ModelABC
from VexDR.models.utils import get_lr_scheduler_from_config, get_optimizer_from_config


class ViTEncoder(ModelABC):
    """ViT encoder exposed as a thin timm wrapper.

    ``model_name`` defines the architecture, so callers usually only need to
    set ``model_name``, ``pretrained``, and maybe ``img_size`` or ``in_chans``.
    Rare timm-specific overrides can still be passed through ``model_kwargs``.

    Parameters
    ----------
    model_name
        The name of a ViT architecture in timm. Defaults to "vit_tiny_patch16_224".
    pretrained
        Whether to load pretrained weights. Defaults to False.
    img_size
        The input image size for the ViT model. Can be an int or a (height, width) tuple.
        If None, the model's default image size will be used. Defaults to None.
    in_chans
        The number of input channels. Defaults to 3.
    drop_rate
        Dropout rate for the model. Defaults to 0.0.
    drop_path_rate
        Drop path rate for the model. Defaults to 0.0.
    attn_drop_rate
        Attention dropout rate for the model. Defaults to 0.0.
    model_kwargs
        Additional keyword arguments for the timm model. Defaults to None.
    optimizer_factory
        Factory function for creating the optimizer. Defaults to None.
    optimizer_kwargs
        Keyword arguments for the optimizer. Defaults to None.
    lr_scheduler_factory
        Factory function for creating the learning rate scheduler. Defaults to None.
    lr_scheduler_kwargs
        Keyword arguments for the learning rate scheduler. Defaults to None.
    lr_scheduler_config
        Configuration dictionary for the learning rate scheduler. Defaults to None.
    """

    def __init__(
        self: ViTEncoder,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        img_size: int | tuple[int, int] | None = None,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
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

        load_dotenv()  # Load environment variables from .env file

        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"model_name must be a non-empty string. Got: {model_name}"
            )
        if not isinstance(in_chans, int) or in_chans <= 0:
            raise ValueError(f"in_chans must be a positive integer. Got: {in_chans}")

        for value, name in (
            (drop_rate, "drop_rate"),
            (drop_path_rate, "drop_path_rate"),
            (attn_drop_rate, "attn_drop_rate"),
        ):
            if not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0). Got: {value}")

        self.model_name = model_name
        self.pretrained = pretrained
        self.img_size = img_size
        self.in_chans = in_chans
        self.drop_rate = float(drop_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.attn_drop_rate = float(attn_drop_rate)
        self.model_kwargs = dict(model_kwargs or {})

        create_kwargs = dict(self.model_kwargs)
        create_kwargs["in_chans"] = in_chans
        create_kwargs["drop_rate"] = self.drop_rate
        create_kwargs["drop_path_rate"] = self.drop_path_rate
        create_kwargs["attn_drop_rate"] = self.attn_drop_rate
        create_kwargs.setdefault("num_classes", 0)
        create_kwargs.setdefault("global_pool", "")
        if img_size is not None:
            create_kwargs["img_size"] = img_size

        self.vit_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            **create_kwargs,
        )
        if not hasattr(self.vit_model, "forward_features"):
            raise TypeError(
                f"timm model '{model_name}' does not expose forward_features()."
            )

        self.embed_dim = getattr(
            self.vit_model,
            "embed_dim",
            getattr(self.vit_model, "num_features", None),
        )
        self.num_prefix_tokens = int(getattr(self.vit_model, "num_prefix_tokens", 0))

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "img_size": img_size,
                "in_chans": in_chans,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "model_kwargs": self.model_kwargs,
                "embed_dim": self.embed_dim,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> ViTEncoder:
        """Construct a ViT encoder from a configuration dictionary."""
        model_name = config.get(
            "model_name", config.get("name", "vit_tiny_patch16_224")
        )
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"model_name must be a non-empty string. Got: {model_name}"
            )

        pretrained = config.get("pretrained", False)
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be a boolean. Got: {pretrained}")

        img_size = config.get("img_size", None)
        if img_size is not None and not isinstance(img_size, (int, tuple)):
            raise ValueError(
                f"img_size must be an int, a (height, width) tuple, or None. Got: {img_size}"
            )

        in_chans = config.get("in_chans", 3)
        if not isinstance(in_chans, int) or in_chans <= 0:
            raise ValueError(f"in_chans must be a positive integer. Got: {in_chans}")

        drop_rate = config.get("drop_rate", 0.0)
        drop_path_rate = config.get("drop_path_rate", 0.0)
        attn_drop_rate = config.get("attn_drop_rate", 0.0)

        model_kwargs = config.get("model_kwargs", {})
        if not isinstance(model_kwargs, dict):
            raise ValueError(
                f"model_kwargs must be a dict of timm overrides. Got: {model_kwargs}"
            )

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", {})
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(
                config.get("lr_scheduler", config.get("scheduler", {}))
            )
        )

        return ViTEncoder(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            model_kwargs=model_kwargs,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(  # pylint: disable=arguments-differ
        self: ViTEncoder, x: Tensor
    ) -> Tensor:
        """Return the token sequence produced by the ViT backbone."""
        return self.vit_model.forward_features(x)  # type: ignore

    def predict_step(  # pylint: disable=arguments-differ
        self: ViTEncoder,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """Run encoder inference on a tensor batch or ``{'image': tensor}`` batch."""
        del batch_idx, dataloader_idx
        return self.forward(self._extract_image_tensor(batch))

    def model_step(
        self: ViTEncoder,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Require task-specific subclasses to define their own loss."""
        del batch, batch_idx, stage
        raise NotImplementedError(
            "ViTEncoder is an encoder-only Lightning module. Override "
            "model_step() in a task-specific subclass to compute a loss."
        )

    def _extract_image_tensor(self: ViTEncoder, batch: Any) -> Tensor:
        """Normalize predict batches to an image tensor."""
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, dict):
            if "image" not in batch:
                raise KeyError("Expected batch dicts to contain an 'image' tensor.")
            image = batch["image"]
            if not isinstance(image, Tensor):
                raise TypeError(
                    "The 'image' entry in the batch dict must be a torch.Tensor."
                )
            return image

        raise TypeError(
            "Expected a torch.Tensor batch or a dict containing an 'image' tensor."
        )
