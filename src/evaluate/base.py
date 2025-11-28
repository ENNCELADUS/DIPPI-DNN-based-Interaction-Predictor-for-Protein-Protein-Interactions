"""
Generic Evaluator for DIPPI pipeline.

This module provides a single Evaluator class that handles:
- Single pass over a dataloader
- Loss accumulation
- Metric computation (AUROC, accuracy, precision, recall, F1, MCC, etc.)

It does NOT handle:
- Model state management (model.eval() / torch.no_grad() owned by orchestrator)
- Logging, checkpointing, or file I/O
- Config parsing or device setup
- Early stopping or loop control

Design follows patterns from docs/evaluator.md.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryAveragePrecision,
)


class Evaluator:
    """
    Generic evaluator for binary classification tasks.

    Orchestrator (run.py) must wrap calls in model.eval() and torch.no_grad().
    Evaluator only performs computation and returns a flat metrics dict.
    """

    def __init__(self, metrics_list: List[str], threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            metrics_list: List of metric names to compute, e.g.,
                ["loss", "auroc", "accuracy", "precision", "recall", "f1",
                 "sensitivity", "specificity", "mcc"]
            threshold: Decision boundary for hard predictions (default 0.5).
                Only affects accuracy/precision/recall/F1/sensitivity/specificity.
                AUROC is threshold-free.
        """
        self.metrics_list = metrics_list
        self.threshold = threshold

        # Map metric names to torchmetrics classes
        # Note: sensitivity is an alias for recall
        metric_map = {
            "auroc": (BinaryAUROC, False),  # (class, needs_threshold)
            "auprc": (
                BinaryAveragePrecision,
                False,
            ),  # Area under precision-recall curve
            "accuracy": (BinaryAccuracy, True),
            "precision": (BinaryPrecision, True),
            "recall": (BinaryRecall, True),
            "sensitivity": (BinaryRecall, True),  # Alias
            "f1": (BinaryF1Score, True),
            "specificity": (BinarySpecificity, True),
            "mcc": (BinaryMatthewsCorrCoef, True),
        }

        # Instantiate only the requested metrics
        self.metrics: Dict[str, Any] = {}
        for metric_name in metrics_list:
            if metric_name == "loss":
                continue  # Loss is computed manually
            if metric_name in metric_map:
                metric_class, needs_threshold = metric_map[metric_name]
                if needs_threshold:
                    self.metrics[metric_name] = metric_class(threshold=threshold)
                else:
                    self.metrics[metric_name] = metric_class()
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")

    def evaluate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        logit_bias: float = 0.0,
        threshold_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Run a single evaluation pass over the dataloader.

        Orchestrator must ensure model.eval() and torch.no_grad() are active.

        Args:
            model: The model to evaluate (already in eval mode).
            loader: DataLoader yielding batches.
            device: Device to run on (cuda/cpu).

        Returns:
            Flat dict of metric name -> value, e.g.,
            {"loss": 0.345, "auroc": 0.876, "accuracy": 0.812, ...}
            Only metrics in self.metrics_list are included (plus "loss" if available).
        """
        # Reset and move all metrics to target device
        for metric in self.metrics.values():
            metric.to(device)
            metric.reset()

        # Loss accumulation
        loss_sum = 0.0
        n_total = 0
        threshold = self.threshold if threshold_override is None else threshold_override

        for batch in loader:
            # Move batch tensors to device
            batch = self._move_batch_to_device(batch, device)

            # Extract labels before model forward
            labels = batch["label"]

            # Forward pass (model returns dict with "logits", may include "loss")
            # Don't pass labels to model - it only needs inputs
            model_inputs = {k: v for k, v in batch.items() if k != "label"}
            out = model(model_inputs)

            # Extract outputs
            logits = out["logits"]

            batch_size = labels.size(0)

            # Compute loss if not provided by model
            if "loss" in out:
                loss = out["loss"]
            else:
                # Compute BCE loss for binary classification
                logits_for_loss = self._normalize_logits(logits)
                # Normalize labels shape to match logits: (N, 1) → (N,)
                labels_for_loss = (
                    labels.squeeze(-1)
                    if labels.dim() > 1 and labels.size(-1) == 1
                    else labels
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_for_loss, labels_for_loss.float()
                )

            # Accumulate loss
            loss_sum += loss.item() * batch_size
            n_total += batch_size

            # Normalize logits shape: handle (N, 1) or (N, 2) → (N,)
            logits = self._normalize_logits(logits)
            if logit_bias:
                logits = logits + float(logit_bias)

            # Convert logits to probabilities for metrics
            probs = torch.sigmoid(logits)

            # Normalize labels shape for metrics: (N, 1) → (N,)
            labels_for_metrics = (
                labels.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1
                else labels
            )
            labels_for_metrics = labels_for_metrics.long()

            # Update all metrics using device-native tensors
            for metric_name, metric in self.metrics.items():
                if metric_name in ["auroc", "auprc"]:
                    # AUROC and AUPRC use probabilities with integer labels
                    metric.update(probs, labels_for_metrics)
                else:
                    # Other metrics use binary predictions
                    preds = (probs > threshold).long()
                    metric.update(preds, labels_for_metrics)

        # Compute final results
        results: Dict[str, float] = {}

        # Add loss
        if "loss" in self.metrics_list:
            results["loss"] = loss_sum / n_total if n_total > 0 else 0.0

        # Compute and add all other metrics
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.compute().detach().cpu().item()

        return results

    def _move_batch_to_device(
        self, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """Move tensor values in batch dict to device."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Normalize logits to shape (N,) for binary classification.

        Handles:
        - (N, 2): extract positive class logit [:, 1]
        - (N, 1): squeeze to (N,)
        - (N,): already correct
        """
        if logits.dim() == 2:
            if logits.size(1) == 2:
                # Two-class logits: take positive class
                return logits[:, 1]
            elif logits.size(1) == 1:
                # Single logit: squeeze
                return logits.squeeze(-1)
        return logits
