"""Loss helpers for tile-level tasks."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def _pool_classification_logits(prediction: Tensor, target: Tensor) -> Tensor:
    """Reduce poolable classification outputs to ``(batch, classes)`` logits."""
    if prediction.ndim <= 2:
        return prediction

    if prediction.ndim == 3:
        if target.ndim >= 2 and prediction.shape[-1] == target.shape[-1]:
            return prediction.mean(dim=1)
        if target.ndim >= 2 and prediction.shape[1] == target.shape[-1]:
            return prediction.mean(dim=-1)
        return prediction.mean(dim=1)

    batch_size = prediction.shape[0]
    return prediction.reshape(batch_size, prediction.shape[1], -1).mean(dim=-1)


def _as_class_indices(target: Tensor, *, class_dim: int = 1) -> Tensor:
    """Convert one-hot or channel-first targets to class indices."""
    if target.ndim == 1:
        return target.long()
    if target.ndim == 2 and target.shape[1] == 1:
        return target.squeeze(1).long()
    if target.ndim > class_dim and target.shape[class_dim] > 1:
        return target.argmax(dim=class_dim).long()
    if target.ndim > class_dim and target.shape[class_dim] == 1:
        return target.squeeze(class_dim).long()
    return target.long()


def _match_regression_target_rank(prediction: Tensor, target: Tensor) -> Tensor:
    """Expand or squeeze regression targets to match the prediction rank."""
    if target.ndim == prediction.ndim:
        return target
    if target.ndim == prediction.ndim - 1 and prediction.ndim >= 2:
        return target.unsqueeze(1)
    if target.ndim == prediction.ndim + 1 and target.shape[1] == 1:
        return target.squeeze(1)
    raise ValueError("Could not align regression target rank with prediction rank.")


def compute_semantic_segmentation_loss(
    prediction: Tensor,
    target: Tensor,
    unknown_class_index: int | None = None,
) -> Tensor:
    """Compute semantic segmentation loss from logits and one-hot/index targets."""
    class_target = _as_class_indices(target, class_dim=1)
    if unknown_class_index is None:
        return F.cross_entropy(prediction.float(), class_target)
    if not (class_target != unknown_class_index).any():
        # Every pixel is the ignore class — cross_entropy would divide by 0
        # and return NaN. Contribute 0 to the total loss while keeping the
        # tensor attached to the computation graph.
        return prediction.float().sum() * 0.0
    return F.cross_entropy(
        prediction.float(),
        class_target,
        ignore_index=unknown_class_index,
    )


def compute_regression_loss(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Compute regression loss after aligning target rank to the prediction."""
    aligned_target = _match_regression_target_rank(prediction, target)
    return F.mse_loss(prediction.float(), aligned_target.float())


def compute_classification_loss(
    prediction: Tensor,
    target: Tensor,
    unknown_class_index: int | None = None,
) -> Tensor:
    """Compute classification loss from logits and one-hot/index targets."""
    logits = _pool_classification_logits(prediction, target).float()

    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2:
        raise ValueError(
            "Classification predictions must be shaped like (batch, classes) "
            "or be reducible to that shape."
        )

    if logits.shape[1] == 1:
        if target.ndim == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1, keepdim=True)
        elif target.ndim == 1:
            target = target.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(logits, target.float())

    class_target = _as_class_indices(target, class_dim=1)
    if unknown_class_index is None:
        return F.cross_entropy(logits, class_target)
    return F.cross_entropy(
        logits,
        class_target,
        ignore_index=unknown_class_index,
    )
