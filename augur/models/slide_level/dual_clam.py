"""CLAM-inspired slide-level MIL model with subtyping main + mutational-signature pretext heads.

Implements CLAM (Clustering-constrained Attention Multiple instance learning,
Lu et al., 2021) as a Lightning module. The main task is slide-level subtyping
(classification, cross-entropy). Pretext tasks are per-submitter COSMIC
signature exposure heads — one per signature class produced by
:func:`scripts.data_handling.extract_signatures.extract_signature_labels`
(``sbs_regression``, ``dbs_regression``, ``id_regression``, ``cnv_regression``).

The single- and multi-branch backbones (:class:`DualCLAM_SB` /
:class:`DualCLAM_MB`) mirror CLAM-SB and CLAM-MB from the original repo at
https://github.com/mahmoodlab/CLAM, but multi-branch is anchored on the
main subtyping task: one attention branch per subtype class, plus one
extra branch per configured pretext task. For example, with 8 subtype
classes and ``pretext_tasks=["sbs_regression"]`` the MB backbone uses
``num_heads = 8 + 1 = 9``.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from augur.models.model_abc import ModelABC
from augur.models.slide_level.attention import Attention, GatedAttention
from augur.models.tile_level.tile_model import TileModel
from augur.models.utils import (
    get_lr_scheduler_from_config,
    get_optimizer_from_config,
)
from augur.utils.config import load_yaml_config
from augur.utils.metrics import (
    compute_classification_loss,
    compute_distribution_kl_loss,
)

_SUPPORTED_MAIN_TASKS: tuple[str, ...] = ("subtyping",)
_SUPPORTED_PRETEXT_TASKS: tuple[str, ...] = (
    "sbs_regression",
    "dbs_regression",
    "id_regression",
    "cnv_regression",
)


def _component_name(module: nn.Module | None) -> str | None:
    """Return a stable component name for logging and hyperparameters."""
    if module is None:
        return None
    return module.__class__.__name__


def _build_projection(
    enc_dim: int, hidden_dims: list[int], dropout: float
) -> nn.Sequential:
    """Build CLAM's tile-feature projection MLP (``enc_dim -> hidden_dims[-1]``)."""
    layers: list[nn.Module] = []
    prev = enc_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = hidden_dim
    return nn.Sequential(*layers)


def _inst_eval(
    attention: Tensor, h: Tensor, classifier: nn.Module, k_sample: int
) -> Tensor:
    """CLAM in-class clustering: top-k attended as positive, bot-k as negative."""
    k = min(k_sample, attention.shape[0])
    top_p_ids = torch.topk(attention, k).indices
    top_p = h.index_select(0, top_p_ids)
    top_n_ids = torch.topk(-attention, k).indices
    top_n = h.index_select(0, top_n_ids)
    p_targets = torch.ones(k, dtype=torch.long, device=h.device)
    n_targets = torch.zeros(k, dtype=torch.long, device=h.device)
    logits = classifier(torch.cat([top_p, top_n], dim=0))
    targets = torch.cat([p_targets, n_targets], dim=0)
    return F.cross_entropy(logits, targets)


def _inst_eval_out(
    attention: Tensor, h: Tensor, classifier: nn.Module, k_sample: int
) -> Tensor:
    """CLAM out-of-class clustering: top-k attended labeled negative for other classes."""
    k = min(k_sample, attention.shape[0])
    top_p_ids = torch.topk(attention, k).indices
    top_p = h.index_select(0, top_p_ids)
    targets = torch.zeros(k, dtype=torch.long, device=h.device)
    logits = classifier(top_p)
    return F.cross_entropy(logits, targets)


class _DualCLAMBase(nn.Module):
    """Shared backbone for CLAM-SB and CLAM-MB.

    Pipeline: encoder (optional) -> projection MLP -> (gated) attention ->
    one head per task. Each task uses a configurable contiguous slice of
    attention branches (the ``branch_layout``). Instance classifiers for
    CLAM's clustering loss are registered for the main task — one binary
    classifier per main-task class.
    """

    def __init__(
        self,
        encoder: ModelABC | None,
        main_task: str,
        pretext_tasks: list[str] | None,
        *,
        enc_dim: int,
        hidden_dims: list[int],
        output_dims: dict[str, int],
        dropout: float,
        attn_kwargs: dict[str, Any],
        num_main_branches: int,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
        unknown_class_index: int | None = 0,
    ) -> None:
        super().__init__()

        if encoder is not None and not isinstance(encoder, ModelABC):
            raise TypeError("encoder must be an inherited ModelABC or None.")
        if not isinstance(enc_dim, int) or enc_dim <= 0:
            raise ValueError(f"enc_dim must be a positive integer. Got: {enc_dim}")
        if (
            not isinstance(hidden_dims, list)
            or not hidden_dims
            or any(not isinstance(h, int) or h <= 0 for h in hidden_dims)
        ):
            raise ValueError(
                "hidden_dims must be a non-empty list of positive integers."
            )
        if not isinstance(output_dims, dict) or not output_dims:
            raise ValueError("output_dims must be a non-empty dict.")
        if main_task not in output_dims:
            raise ValueError(
                f"output_dims must include an entry for main_task='{main_task}'."
            )
        if not isinstance(output_dims[main_task], int) or output_dims[main_task] <= 1:
            raise ValueError(
                f"output_dims['{main_task}'] must be an integer > 1 for "
                f"a classification main task. Got: {output_dims[main_task]}"
            )
        for pretext in pretext_tasks or []:
            if pretext not in output_dims:
                raise ValueError(
                    f"output_dims must include an entry for pretext task '{pretext}'."
                )
            if not isinstance(output_dims[pretext], int) or output_dims[pretext] <= 0:
                raise ValueError(
                    f"output_dims['{pretext}'] must be a positive integer. "
                    f"Got: {output_dims[pretext]}"
                )
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be a float in [0.0, 1.0). Got: {dropout}")
        if not isinstance(num_main_branches, int) or num_main_branches <= 0:
            raise ValueError(
                "num_main_branches must be a positive integer. "
                f"Got: {num_main_branches}"
            )

        self.has_encoder = encoder is not None
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.freeze_encoder = bool(freeze_encoder) and self.has_encoder
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        if not isinstance(encoder_chunk_size, int):
            raise TypeError("encoder_chunk_size must be an integer.")
        self.encoder_chunk_size = encoder_chunk_size

        self.main_task = main_task
        self.pretext_tasks: list[str] = list(pretext_tasks or [])
        self.enc_dim = int(enc_dim)
        self.hidden_dims = list(hidden_dims)
        self.projection_dim = int(hidden_dims[-1])
        self.num_main_branches = int(num_main_branches)

        # Branch layout. SB (num_main_branches == 1) shares the single branch
        # across all tasks. MB allocates one branch per main-task class plus
        # one extra branch per pretext task.
        if self.num_main_branches == 1:
            self.num_heads = 1
            self.branch_layout: dict[str, tuple[int, int]] = {
                task: (0, 1) for task in [main_task, *self.pretext_tasks]
            }
        else:
            self.num_heads = self.num_main_branches + len(self.pretext_tasks)
            self.branch_layout = {main_task: (0, self.num_main_branches)}
            for i, pretext in enumerate(self.pretext_tasks):
                start = self.num_main_branches + i
                self.branch_layout[pretext] = (start, start + 1)

        self.projection = _build_projection(enc_dim, self.hidden_dims, float(dropout))

        attn_cfg = dict(attn_kwargs or {})
        self.gated = bool(attn_cfg.get("gated", True))
        attn_hidden_dim = int(attn_cfg.get("hidden_dim", 256))
        attn_dropout = float(attn_cfg.get("dropout", 0.0))
        aggregator_cls = GatedAttention if self.gated else Attention
        self.attention_net = aggregator_cls(
            input_dim=self.projection_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout,
        )

        self.heads = nn.ModuleDict()
        for task_name in [main_task, *self.pretext_tasks]:
            start, end = self.branch_layout[task_name]
            head_in_dim = (end - start) * self.projection_dim
            self.heads[task_name] = nn.Linear(head_in_dim, int(output_dims[task_name]))

        # Instance classifiers for CLAM's clustering loss on the main
        # (classification) task. One binary head per main-task class. The
        # `unknown_class_index` slot is replaced with nn.Identity so it has
        # no parameters — _compute_instance_loss skips that index, and a
        # zero-param slot avoids DDP "unused parameters" deadlocks.
        num_main_classes = int(output_dims[main_task])
        self.unknown_class_index = unknown_class_index
        self.instance_classifiers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if unknown_class_index is not None and c == unknown_class_index
                    else nn.Linear(self.projection_dim, 2)
                )
                for c in range(num_main_classes)
            ]
        )

    def train(self: _DualCLAMBase, mode: bool = True) -> _DualCLAMBase:
        """Keep a frozen encoder in eval mode even when the parent is training."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, bag: Tensor | dict[str, Any]) -> dict[str, Any]:
        """Run encoder -> projection -> attention -> heads end-to-end."""
        if isinstance(bag, dict):
            if "image" not in bag:
                raise KeyError(
                    "DualCLAM expects a dict batch with an 'image' entry or a tensor."
                )
            image = bag["image"]
        else:
            image = bag
        if not isinstance(image, Tensor):
            raise TypeError("DualCLAM expected 'image' to be a torch.Tensor.")

        features = self._encode_bag(image)
        projected = self.projection(features)
        aggregated, attention_weights = self.attention_net(projected)

        # Reshape (B, num_heads * L) -> (B, num_heads, L) so we can slice per-task.
        batch_size = aggregated.shape[0]
        per_branch = aggregated.view(batch_size, self.num_heads, self.projection_dim)

        outputs: dict[str, Any] = {}
        for task_name, head in self.heads.items():
            start, end = self.branch_layout[task_name]
            task_input = per_branch[:, start:end, :].flatten(start_dim=1)
            outputs[task_name] = head(task_input)
        outputs["_projected"] = projected
        outputs["_aggregated"] = aggregated
        outputs["_attention_weights"] = attention_weights
        return outputs

    def _encode_bag(self, image: Tensor) -> Tensor:
        """Encode a tile bag to per-instance features of shape ``(B, K, D)``."""
        if image.ndim == 3:
            return image
        if image.ndim != 5:
            raise ValueError(
                "Expected image shape (B, K, 3, H, W) or (B, K, D). "
                f"Got: {tuple(image.shape)}"
            )
        if not self.has_encoder:
            raise ValueError(
                "DualCLAM received raw tiles but no encoder was configured. "
                "Either supply an encoder or pre-encode to (B, K, D)."
            )
        batch_size, num_tiles = image.shape[:2]
        flat = image.flatten(0, 1)
        chunk_size = self.encoder_chunk_size
        # Chunk the encoder forward so layer-1 activations don't blow up GPU
        # memory on whole-slide bags. Frozen encoders can run under no_grad.
        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with ctx:
            if chunk_size <= 0 or flat.shape[0] <= chunk_size:
                features = self._get_last_features(self.encoder(flat))
            else:
                feature_chunks = [
                    self._get_last_features(
                        self.encoder(flat[start : start + chunk_size])
                    )
                    for start in range(0, flat.shape[0], chunk_size)
                ]
                features = torch.cat(feature_chunks, dim=0)
        return features.view(batch_size, num_tiles, -1)

    def _get_last_features(self, outputs: Tensor | Sequence[Tensor]) -> Tensor:
        if not self.has_encoder and isinstance(outputs, Tensor) and outputs.ndim == 2:
            return outputs
        encoder_name = self.encoder.__class__.__name__
        match encoder_name:
            case "ViTEncoder":
                return self._get_last_features_vit(outputs)
            case "ResNetEncoder" | "UNetEncoder":
                assert (
                    isinstance(outputs, Sequence) and len(outputs) == 5
                ), f"Expected 5 feature maps from {encoder_name}."
                _, _, _, _, c4 = outputs
                return c4.flatten(start_dim=2).mean(dim=-1)
            case _:
                if isinstance(outputs, Tensor) and outputs.ndim == 2:
                    return outputs
                raise ValueError(
                    f"DualCLAM has no default feature extraction strategy for "
                    f"'{encoder_name}'. Ensure the encoder returns (N, D)."
                )

    def _get_last_features_vit(self, outputs: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(outputs, Tensor):
            tokens = outputs
        elif (
            isinstance(outputs, Sequence)
            and len(outputs) > 0
            and isinstance(outputs[-1], Tensor)
        ):
            tokens = outputs[-1]
        else:
            raise TypeError(
                "Expected ViT outputs to be a Tensor or non-empty Tensor sequence. "
                f"Got: {type(outputs)!r}"
            )
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected ViT token output (N, seq_len, C). Got: {tuple(tokens.shape)}"
            )
        num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", None)
        assert (
            num_prefix_tokens is not None
        ), "ViTEncoder must expose a num_prefix_tokens attribute."
        if num_prefix_tokens >= 1:
            return tokens[:, 0, :]
        return tokens[:, num_prefix_tokens:, :].mean(dim=1)


class DualCLAM_SB(_DualCLAMBase):
    """Single-branch CLAM backbone.

    A single attention distribution is shared across the main task and every
    pretext task (equivalent to CLAM-SB in the original paper).
    """

    def __init__(
        self: DualCLAM_SB,
        encoder: ModelABC | None = None,
        main_task: str = "subtyping",
        pretext_tasks: list[str] | None = None,
        *,
        enc_dim: int,
        hidden_dims: list[int],
        output_dims: dict[str, int],
        dropout: float = 0.0,
        attn_kwargs: dict[str, Any] | None = None,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
        unknown_class_index: int | None = 0,
    ) -> None:
        super().__init__(
            encoder=encoder,
            main_task=main_task,
            pretext_tasks=pretext_tasks,
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=dropout,
            attn_kwargs=attn_kwargs or {},
            num_main_branches=1,
            freeze_encoder=freeze_encoder,
            encoder_chunk_size=encoder_chunk_size,
            unknown_class_index=unknown_class_index,
        )


class DualCLAM_MB(_DualCLAMBase):
    """Multi-branch CLAM backbone.

    Allocates ``num_main_branches = output_dims[main_task]`` attention branches
    for the main subtyping task — one branch per subtype class — plus one
    extra branch per configured pretext task (each pretext gets its own single
    attention distribution). Total ``num_heads = output_dims[main_task] +
    len(pretext_tasks)``.
    """

    def __init__(
        self: DualCLAM_MB,
        encoder: ModelABC | None = None,
        main_task: str = "subtyping",
        pretext_tasks: list[str] | None = None,
        *,
        enc_dim: int,
        hidden_dims: list[int],
        output_dims: dict[str, int],
        dropout: float = 0.0,
        attn_kwargs: dict[str, Any] | None = None,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
        unknown_class_index: int | None = 0,
    ) -> None:
        if main_task not in output_dims:
            raise ValueError(
                f"output_dims must include '{main_task}' to size the "
                "multi-branch attention."
            )
        num_main_branches = int(output_dims[main_task])
        if num_main_branches < 2:
            raise ValueError(
                "DualCLAM_MB requires output_dims[main_task] >= 2 to allocate "
                f"per-class attention branches. Got: {num_main_branches}."
            )
        super().__init__(
            encoder=encoder,
            main_task=main_task,
            pretext_tasks=pretext_tasks,
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=dropout,
            attn_kwargs=attn_kwargs or {},
            num_main_branches=num_main_branches,
            freeze_encoder=freeze_encoder,
            encoder_chunk_size=encoder_chunk_size,
            unknown_class_index=unknown_class_index,
        )


class DualCLAM(ModelABC):
    """CLAM-based Lightning model for slide-level subtyping + SBS pretexts.

    Batch layout (matches ``TCGASlideDataset``):

    - Main subtyping target: ``batch["target"]`` — ``(B,)`` long.
    - Each pretext SBS target: ``batch[pretext_task]["target"]`` — ``(B, D)``
      float vector.

    Parameters
    ----------
    encoder
        Optional shared tile encoder. If ``None``, inputs are expected to be
        pre-computed feature bags of shape ``(B, K, D)``.
    main_task
        Currently must be ``"subtyping"`` (classification with cross-entropy).
    pretext_tasks
        Optional list of SBS pretext task names from
        :data:`_SUPPORTED_PRETEXT_TASKS`. Each adds a separate decoder head
        and (in MB) a separate attention branch.
    task_weights
        Optional per-task positive weights. Normalized to sum to 1.
    task_kwargs
        Optional mapping from task name to per-task keyword arguments used in
        loss computation. Recognized keys per task:

        - ``unknown_class_index`` (int or None): class index to ignore in the
          subtyping cross-entropy and to skip during instance clustering.
          Defaults to ``0`` for the main subtyping task (matches
          ``TCGASlideDataset.UNKNOWN_SUBTYPE_CLASS``).
    enc_dim
        Tile encoder output dimensionality ``D``.
    hidden_dims
        Projection MLP hidden layer sizes. Last entry defines the CLAM
        projection dim ``L``.
    output_dims
        Mapping from task name to head output dim. Must include ``main_task``
        (with ``> 1`` classes) and every entry of ``pretext_tasks``.
    dropout
        Dropout applied inside the projection MLP.
    attn_kwargs
        Attention-architecture settings:

        - ``multi_branch`` (bool, default False): select :class:`DualCLAM_SB`
          / :class:`DualCLAM_MB` backbone.
        - ``gated`` (bool, default True): gated vs. plain attention.
        - ``hidden_dim`` (int, default 256): attention hidden dim.
        - ``dropout`` (float, default 0.0): attention dropout.
    cluster_kwargs
        CLAM instance-clustering loss settings (applied to the main
        subtyping task):

        - ``k_sample`` (int, default 8): top-k for instance clustering.
        - ``inst_weight`` (float, default 0.0): instance clustering loss
          weight. ``0`` disables instance clustering.
        - ``out_of_class`` (bool, default False): also evaluate the main-task
          instance classifiers of *other* classes against the most-attended
          tiles for the sample's true class (CLAM's "subtyping" auxiliary
          loss in the original paper).
    freeze_encoder
        If ``True`` (default), set ``requires_grad=False`` on every encoder
        parameter and keep the encoder in ``eval()`` mode even when the
        parent is flipped to ``train()``.
    optimizer_factory, optimizer_kwargs, lr_scheduler_factory,
    lr_scheduler_kwargs, lr_scheduler_config
        Optimization configuration forwarded to :class:`ModelABC`.
    """

    def __init__(
        self: DualCLAM,
        encoder: ModelABC | None = None,
        main_task: str = "subtyping",
        pretext_tasks: list[str] | None = None,
        task_weights: dict[str, float] | None = None,
        task_kwargs: dict[str, Any] | None = None,
        *,
        enc_dim: int,
        hidden_dims: list[int],
        output_dims: dict[str, int],
        dropout: float = 0.0,
        attn_kwargs: dict[str, Any] | None = None,
        cluster_kwargs: dict[str, Any] | None = None,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_factory: Callable[..., LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            optimizer_factory,
            optimizer_kwargs,
            lr_scheduler_factory,
            lr_scheduler_kwargs,
            lr_scheduler_config,
        )

        if main_task not in _SUPPORTED_MAIN_TASKS:
            raise ValueError(
                f"main_task must be one of {_SUPPORTED_MAIN_TASKS}. "
                f"Got: {main_task!r}"
            )
        pretext_list = list(pretext_tasks or [])
        for pretext in pretext_list:
            if pretext not in _SUPPORTED_PRETEXT_TASKS:
                raise ValueError(
                    f"pretext task must be one of {_SUPPORTED_PRETEXT_TASKS}. "
                    f"Got: {pretext!r}"
                )

        attn_cfg = dict(attn_kwargs or {})
        multi_branch = bool(attn_cfg.get("multi_branch", False))
        backbone_cls: type[_DualCLAMBase] = DualCLAM_MB if multi_branch else DualCLAM_SB

        # Pull unknown_class_index from task_kwargs once so the backbone's
        # instance_classifiers and runtime _compute_instance_loss agree.
        main_task_kwargs_init = (task_kwargs or {}).get(main_task, {})
        if not isinstance(main_task_kwargs_init, dict):
            raise TypeError(f"task_kwargs['{main_task}'] must be a dict if provided.")
        backbone_unknown_class_index = main_task_kwargs_init.get(
            "unknown_class_index", 0
        )
        if backbone_unknown_class_index is not None and (
            not isinstance(backbone_unknown_class_index, int)
            or backbone_unknown_class_index < 0
        ):
            raise ValueError(
                f"task_kwargs['{main_task}']['unknown_class_index'] must be a "
                f"non-negative int or None. Got: {backbone_unknown_class_index}"
            )

        self.backbone = backbone_cls(
            encoder=encoder,
            main_task=main_task,
            pretext_tasks=pretext_list or None,
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=dropout,
            attn_kwargs=attn_cfg,
            freeze_encoder=freeze_encoder,
            encoder_chunk_size=encoder_chunk_size,
            unknown_class_index=backbone_unknown_class_index,
        )
        self.freeze_encoder = self.backbone.freeze_encoder
        self.encoder_chunk_size = self.backbone.encoder_chunk_size

        self.main_task = main_task
        self.pretext_tasks = pretext_list
        self.output_dims = dict(output_dims)

        all_tasks = [main_task, *pretext_list]
        if not task_weights:
            task_weights = {name: 1.0 for name in all_tasks}
        normalized = {name: float(task_weights.get(name, 1.0)) for name in all_tasks}
        assert all(
            w > 0.0 for w in normalized.values()
        ), "Task weights must be positive."
        total = sum(normalized.values())
        if total > 0:
            normalized = {name: w / total for name, w in normalized.items()}
        self.task_weights = normalized

        if cluster_kwargs is not None and not isinstance(cluster_kwargs, dict):
            raise TypeError("cluster_kwargs must be a dict if provided.")
        cluster_cfg = dict(cluster_kwargs or {})

        k_sample = cluster_cfg.get("k_sample", 8)
        if not isinstance(k_sample, int) or k_sample <= 0:
            raise ValueError(
                f"cluster_kwargs['k_sample'] must be a positive integer. Got: {k_sample}"
            )
        self.k_sample = int(k_sample)

        inst_weight = cluster_cfg.get("inst_weight", 0.0)
        if not isinstance(inst_weight, (int, float)) or inst_weight < 0.0:
            raise ValueError(
                "cluster_kwargs['inst_weight'] must be a non-negative number. "
                f"Got: {inst_weight}"
            )
        self.inst_weight = float(inst_weight)

        out_of_class = cluster_cfg.get("out_of_class", False)
        if not isinstance(out_of_class, bool):
            raise TypeError(
                "cluster_kwargs['out_of_class'] must be a boolean. "
                f"Got: {out_of_class}"
            )
        self.out_of_class = bool(out_of_class)

        if task_kwargs is not None and not isinstance(task_kwargs, dict):
            raise TypeError(
                "task_kwargs must be a dict of task name to per-task options."
            )
        self.task_kwargs: dict[str, Any] = dict(task_kwargs or {})

        self.save_hyperparameters(
            {
                "encoder": _component_name(encoder),
                "backbone": _component_name(self.backbone),
                "main_task": main_task,
                "pretext_tasks": pretext_list,
                "task_weights": self.task_weights,
                "task_kwargs": self.task_kwargs,
                "enc_dim": enc_dim,
                "hidden_dims": list(hidden_dims),
                "output_dims": self.output_dims,
                "dropout": float(dropout),
                "attn_kwargs": attn_cfg,
                "cluster_kwargs": {
                    "k_sample": self.k_sample,
                    "inst_weight": self.inst_weight,
                    "out_of_class": self.out_of_class,
                },
                "freeze_encoder": self.freeze_encoder,
                "encoder_chunk_size": self.encoder_chunk_size,
            }
        )

    @staticmethod
    def from_config(config: dict[str, Any]) -> DualCLAM:
        """Instantiate a DualCLAM from a configuration dictionary.

        Recognized keys mirror the constructor parameters: ``tile_model_config``
        (path or inline dict for a :class:`TileModel` whose encoder is reused),
        ``main_task``, ``pretext_tasks``, ``task_weights``, ``task_kwargs``,
        ``enc_dim``, ``hidden_dims``, ``output_dims``, ``dropout``,
        ``attn_kwargs``, ``cluster_kwargs``, ``freeze_encoder``,
        ``optimizer``, and ``lr_scheduler``.
        """
        if not isinstance(config, dict):
            raise TypeError("DualCLAM config must be provided as a dict.")

        tile_model_spec = config.get("tile_model_config", None)
        if tile_model_spec is not None:
            tile_model_cfg = DualCLAM._resolve_component_config(
                tile_model_spec,
                label="tile_model_config",
            )
            tile_model_params = tile_model_cfg.get("params", tile_model_cfg)
            tile_model = TileModel.from_config(tile_model_params)
            if not isinstance(tile_model, TileModel):
                raise TypeError(
                    "The 'tile_model' component of the config must be a TileModel instance."
                )

            checkpoint_path = tile_model_cfg.get("checkpoint_path", None)
            assert isinstance(
                checkpoint_path, str
            ), "tile_model_config checkpoint_path must be a string."
            assert os.path.isfile(
                checkpoint_path
            ), f"tile_model_config checkpoint_path does not exist: {checkpoint_path}"

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            tile_model.load_state_dict(state_dict, strict=False)

            encoder = tile_model.encoder
        else:
            encoder = None

        main_task = config.get("main_task", "subtyping")
        assert (
            isinstance(main_task, str) and main_task
        ), "main_task must be a non-empty string"

        pretext_tasks = config.get("pretext_tasks", None)
        if pretext_tasks is not None:
            assert isinstance(pretext_tasks, list) and all(
                isinstance(t, str) and t for t in pretext_tasks
            ), "pretext_tasks must be a list of non-empty strings when provided"

        task_weights = config.get("task_weights", None)
        if task_weights is not None and not isinstance(task_weights, dict):
            raise TypeError("task_weights must be a dict if provided.")

        task_kwargs = config.get("task_kwargs", None)
        if task_kwargs is not None and not isinstance(task_kwargs, dict):
            raise TypeError("task_kwargs must be a dict if provided.")

        enc_dim = config.get("enc_dim", None)
        assert enc_dim is not None, "enc_dim must be specified in the config"
        assert (
            isinstance(enc_dim, int) and enc_dim > 0
        ), "enc_dim must be a positive integer"

        hidden_dims = config.get("hidden_dims", None)
        assert hidden_dims is not None, "hidden_dims must be specified in the config"
        assert (
            isinstance(hidden_dims, list)
            and hidden_dims
            and all(isinstance(hd, int) and hd > 0 for hd in hidden_dims)
        ), "hidden_dims must be a non-empty list of positive integers"

        output_dims = config.get("output_dims", None)
        assert output_dims is not None, "output_dims must be specified in the config"
        assert isinstance(output_dims, dict) and all(
            isinstance(name, str) and isinstance(dim, int) and dim > 0
            for name, dim in output_dims.items()
        ), "output_dims must be a dict of task name to positive integer output dim"

        dropout = config.get("dropout", 0.0)
        assert (
            isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0
        ), "dropout must be a float in the range [0.0, 1.0)"

        attn_kwargs = config.get("attn_kwargs", None)
        if attn_kwargs is not None and not isinstance(attn_kwargs, dict):
            raise TypeError("attn_kwargs must be a dict if provided.")

        cluster_kwargs = config.get("cluster_kwargs", None)
        if cluster_kwargs is not None and not isinstance(cluster_kwargs, dict):
            raise TypeError("cluster_kwargs must be a dict if provided.")

        freeze_encoder = config.get("freeze_encoder", True)
        if not isinstance(freeze_encoder, bool):
            raise TypeError("freeze_encoder must be a boolean if provided.")

        encoder_chunk_size = config.get("encoder_chunk_size", 64)
        if not isinstance(encoder_chunk_size, int):
            raise TypeError("encoder_chunk_size must be an int if provided.")

        optimizer_factory, optimizer_kwargs = get_optimizer_from_config(
            config.get("optimizer", None)
        )
        lr_scheduler_factory, lr_scheduler_kwargs, lr_scheduler_config = (
            get_lr_scheduler_from_config(config.get("lr_scheduler", None))
        )

        return DualCLAM(
            encoder=encoder,
            main_task=main_task,
            pretext_tasks=pretext_tasks,
            task_weights=task_weights,
            task_kwargs=task_kwargs,
            enc_dim=enc_dim,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=dropout,
            attn_kwargs=attn_kwargs,
            cluster_kwargs=cluster_kwargs,
            freeze_encoder=freeze_encoder,
            encoder_chunk_size=encoder_chunk_size,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_factory=lr_scheduler_factory,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_config=lr_scheduler_config,
        )

    @staticmethod
    def _resolve_component_config(
        spec: str | Path | dict[str, Any] | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        """Load a component config from an inline dict or YAML path."""
        if spec is None:
            raise ValueError(f"DualCLAM configuration requires '{label}'.")
        if isinstance(spec, (str, Path)):
            return load_yaml_config(spec)
        if isinstance(spec, dict):
            return dict(spec)
        raise TypeError(
            f"DualCLAM configuration field '{label}' must be a dict or path. "
            f"Got: {type(spec)!r}"
        )

    def forward(  # pylint: disable=arguments-differ
        self: DualCLAM, batch_or_image: Tensor | dict[str, Any]
    ) -> dict[str, Any]:
        """Run the backbone end-to-end."""
        return self.backbone(batch_or_image)

    def predict_step(  # pylint: disable=arguments-differ
        self: DualCLAM,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction on a slide-level batch."""
        del batch_idx, dataloader_idx
        return self.forward(batch)

    def model_step(
        self: DualCLAM, batch: Any, batch_idx: int, stage: str
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Compute CLAM's bag-level + optional pretext + instance-clustering losses."""
        del batch_idx, stage

        if not isinstance(batch, dict):
            raise TypeError("DualCLAM.model_step() expects a dict batch.")

        predictions = self.forward(batch)

        main_target = batch.get("target")
        if main_target is None:
            raise KeyError(
                "DualCLAM.model_step() could not find the main-task target at "
                "batch['target']."
            )
        if not isinstance(main_target, Tensor):
            raise TypeError("Main-task target must be a torch.Tensor.")
        main_prediction = predictions[self.main_task]

        main_task_kwargs = self.task_kwargs.get(self.main_task, {})
        unknown_class_index = main_task_kwargs.get("unknown_class_index", 0)

        main_loss = self._compute_main_loss(
            main_prediction, main_target, unknown_class_index=unknown_class_index
        )
        total_loss = self.task_weights[self.main_task] * main_loss
        metrics: dict[str, Tensor] = {f"{self.main_task}_loss": main_loss.detach()}

        # Pretext SBS losses (vector regression / multilabel).
        for pretext in self.pretext_tasks:
            pretext_batch = batch.get(pretext)
            if not isinstance(pretext_batch, dict):
                continue
            pretext_target = pretext_batch.get("target")
            if pretext_target is None:
                continue
            if not isinstance(pretext_target, Tensor):
                raise TypeError(f"Pretext '{pretext}' target must be a torch.Tensor.")

            pretext_logits = predictions[pretext]
            pretext_loss = self._compute_pretext_loss(
                pretext, pretext_logits, pretext_target
            )
            total_loss = total_loss + self.task_weights[pretext] * pretext_loss
            metrics[f"{pretext}_loss"] = pretext_loss.detach()

        # Instance clustering on the main subtyping task.
        if self.inst_weight > 0.0:
            inst_loss = self._compute_instance_loss(
                projected=predictions["_projected"],
                attention_weights=predictions["_attention_weights"],
                main_target=main_target,
                unknown_class_index=unknown_class_index,
            )
            total_loss = total_loss + self.inst_weight * inst_loss
            metrics[f"{self.main_task}_instance_loss"] = inst_loss.detach()

        return total_loss, metrics

    def _compute_main_loss(
        self: DualCLAM,
        prediction: Tensor,
        target: Tensor,
        *,
        unknown_class_index: int | None,
    ) -> Tensor:
        """Cross-entropy for the subtyping main task."""
        match self.main_task:
            case "subtyping":
                return compute_classification_loss(
                    prediction, target, unknown_class_index=unknown_class_index
                )
            case _:
                raise ValueError(
                    f"No default loss defined for main_task '{self.main_task}'."
                )

    def _compute_pretext_loss(
        self: DualCLAM,
        pretext_task: str,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Dispatch the pretext mutational-signature loss based on the task name."""
        match pretext_task:
            case (
                "sbs_regression"
                | "dbs_regression"
                | "id_regression"
                | "cnv_regression"
            ):
                # Targets are per-submitter normalized COSMIC exposure vectors
                # (row-sum 1), so distributional KL is the appropriate loss.
                return compute_distribution_kl_loss(prediction, target)
            case _:
                raise ValueError(
                    f"No default loss defined for pretext task '{pretext_task}'."
                )

    def _compute_instance_loss(
        self: DualCLAM,
        *,
        projected: Tensor,
        attention_weights: Tensor,
        main_target: Tensor,
        unknown_class_index: int | None,
    ) -> Tensor:
        """CLAM instance clustering loss for the main subtyping task."""
        inst_classifiers = self.backbone.instance_classifiers
        assert isinstance(inst_classifiers, nn.ModuleList)
        num_classes = len(inst_classifiers)
        batch_size = projected.shape[0]
        k_sample = self.k_sample
        num_main_branches = self.backbone.num_main_branches
        losses: list[Tensor] = []

        for i in range(batch_size):
            label = int(main_target[i].item())
            if label == unknown_class_index or label < 0 or label >= num_classes:
                continue
            h = projected[i]
            # In SB (num_main_branches == 1) the shared branch is index 0.
            # In MB the i-th main-task class uses the i-th branch.
            branch_idx = 0 if num_main_branches == 1 else label
            attention = attention_weights[i, branch_idx]

            loss_i = _inst_eval(attention, h, inst_classifiers[label], k_sample)
            if self.out_of_class:
                for c in range(num_classes):
                    if c == label or c == unknown_class_index:
                        continue
                    loss_i = loss_i + _inst_eval_out(
                        attention, h, inst_classifiers[c], k_sample
                    )
            losses.append(loss_i)

        if not losses:
            return projected.new_zeros(())
        return torch.stack(losses).mean()
