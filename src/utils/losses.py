"""Loss configuration and helpers for binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LossConfig:
    """Configuration for binary classification loss.

    Attributes:
        loss_type: Supported loss name.
        pos_weight: Positive-class weighting factor.
        label_smoothing: Label smoothing ratio in ``[0, 1)``.
    """

    loss_type: str = "bce_with_logits"
    pos_weight: float = 1.0
    label_smoothing: float = 0.0


def _smoothed_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply binary label smoothing.

    Args:
        labels: Binary labels in ``{0, 1}``.
        smoothing: Smoothing factor in ``[0, 1)``.

    Returns:
        Smoothed labels in ``[0, 1]``.
    """
    if smoothing <= 0.0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def binary_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_config: LossConfig,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute configured binary classification loss.

    Args:
        logits: Raw logits tensor.
        labels: Binary labels tensor.
        loss_config: Loss hyperparameters.
        reduction: Reduction mode for BCE loss.

    Returns:
        Loss tensor with selected reduction.

    Raises:
        ValueError: If loss configuration is unsupported or invalid.
    """
    loss_type = loss_config.loss_type.lower()
    if loss_type != "bce_with_logits":
        raise ValueError(f"Unsupported loss type: {loss_config.loss_type}")
    if not 0.0 <= loss_config.label_smoothing < 1.0:
        raise ValueError("training_config.loss.label_smoothing must be in [0, 1)")
    if loss_config.pos_weight <= 0.0:
        raise ValueError("training_config.loss.pos_weight must be > 0")

    prepared_logits = logits
    prepared_labels = labels.float()
    if prepared_logits.dim() > 1 and prepared_logits.size(-1) == 1:
        prepared_logits = prepared_logits.squeeze(-1)
    if prepared_labels.dim() > 1 and prepared_labels.size(-1) == 1:
        prepared_labels = prepared_labels.squeeze(-1)

    smoothed_labels = _smoothed_labels(prepared_labels, loss_config.label_smoothing)
    pos_weight = torch.tensor(
        loss_config.pos_weight,
        dtype=prepared_logits.dtype,
        device=prepared_logits.device,
    )
    return functional.binary_cross_entropy_with_logits(
        prepared_logits,
        smoothed_labels,
        pos_weight=pos_weight,
        reduction=reduction,
    )
