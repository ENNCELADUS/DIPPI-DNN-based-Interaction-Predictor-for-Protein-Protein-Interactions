"""Unit tests for configurable loss utilities."""

from __future__ import annotations

import torch

from src.utils.losses import LossConfig, binary_classification_loss


def test_binary_classification_loss_supports_label_smoothing() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
    config = LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.2)
    loss = binary_classification_loss(logits=logits, labels=labels, loss_config=config)
    assert float(loss.item()) > 0.0


def test_binary_classification_loss_supports_pos_weight() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
    base_config = LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0)
    weighted_config = LossConfig(loss_type="bce_with_logits", pos_weight=2.0, label_smoothing=0.0)
    base_loss = binary_classification_loss(logits=logits, labels=labels, loss_config=base_config)
    weighted_loss = binary_classification_loss(
        logits=logits, labels=labels, loss_config=weighted_config
    )
    assert float(weighted_loss.item()) > float(base_loss.item())
