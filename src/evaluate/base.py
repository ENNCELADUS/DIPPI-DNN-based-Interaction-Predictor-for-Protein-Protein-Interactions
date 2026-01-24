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

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

# Number of bins used for AUROC/AUPRC when no explicit config is provided.
# Using thresholds bounds memory so evaluation on large splits does not OOM.
DEFAULT_CURVE_THRESHOLDS = 512


class Evaluator:
    """
    Generic evaluator for binary classification tasks.

    Orchestrator (run.py) must wrap calls in model.eval() and torch.no_grad().
    Evaluator only performs computation and returns a flat metrics dict.
    """

    DEFAULT_CURVE_THRESHOLDS = DEFAULT_CURVE_THRESHOLDS

    def __init__(
        self,
        metrics_list: List[str],
        threshold: float = 0.5,
        curve_thresholds: Optional[int] = DEFAULT_CURVE_THRESHOLDS,
    ):
        """
        Initialize evaluator.

        Args:
            metrics_list: List of metric names to compute, e.g.,
                ["loss", "auroc", "accuracy", "precision", "recall", "f1",
                 "sensitivity", "specificity", "mcc"]
            threshold: Decision boundary for hard predictions (default 0.5).
                Only affects accuracy/precision/recall/F1/sensitivity/specificity.
                AUROC/AUPRC ignore this threshold.
            curve_thresholds: Number of thresholds to use when computing AUROC/AUPRC.
                Setting this bounds memory (O(num_thresholds) instead of O(num_samples)).
                Set to None to fall back to exact accumulation (higher memory).
        """
        self.metrics_list = metrics_list
        self.threshold = threshold

        if curve_thresholds is not None:
            if curve_thresholds < 2:
                raise ValueError("curve_thresholds must be >= 2 when provided")
            self.curve_thresholds: Optional[int] = int(curve_thresholds)
        else:
            self.curve_thresholds = None

        # Map metric names to torchmetrics classes and whether they support binned curves
        # Note: sensitivity is an alias for recall
        metric_map = {
            "auroc": (
                BinaryAUROC,
                False,
                True,
            ),  # (class, needs_threshold, supports_curve_bins)
            "auprc": (
                BinaryAveragePrecision,
                False,
                True,
            ),  # Area under precision-recall curve
            "accuracy": (BinaryAccuracy, True, False),
            "precision": (BinaryPrecision, True, False),
            "recall": (BinaryRecall, True, False),
            "sensitivity": (BinaryRecall, True, False),  # Alias
            "f1": (BinaryF1Score, True, False),
            "specificity": (BinarySpecificity, True, False),
            "mcc": (BinaryMatthewsCorrCoef, True, False),
        }

        # Instantiate only the requested metrics
        self.metrics: Dict[str, Any] = {}
        for metric_name in metrics_list:
            if metric_name == "loss":
                continue  # Loss is computed manually
            if metric_name not in metric_map:
                raise ValueError(f"Unsupported metric: {metric_name}")

            metric_class, needs_threshold, supports_bins = metric_map[metric_name]
            metric_kwargs: Dict[str, Any] = {}
            if needs_threshold:
                metric_kwargs["threshold"] = threshold
            if supports_bins and self.curve_thresholds:
                metric_kwargs["thresholds"] = self.curve_thresholds

            # Older torchmetrics versions may not accept thresholds; fall back gracefully
            try:
                self.metrics[metric_name] = metric_class(**metric_kwargs)
            except TypeError as exc:
                if "thresholds" in metric_kwargs:
                    metric_kwargs.pop("thresholds", None)
                    logging.warning(
                        "Metric %s does not support 'thresholds' on this torchmetrics "
                        "version; falling back to exact accumulation (higher memory). "
                        "Error: %s",
                        metric_name,
                        exc,
                    )
                    self.metrics[metric_name] = metric_class(**metric_kwargs)
                else:
                    raise

    def evaluate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run a single evaluation pass over the dataloader, yielding per-batch progress.

        Orchestrator must ensure model.eval() and torch.no_grad() are active.

        Args:
            model: The model to evaluate (already in eval mode).
            loader: DataLoader yielding batches.
            device: Device to run on (cuda/cpu).
        Yields:
            Per-batch dict with keys: batch_idx, batch_size, loss
            Final dict with aggregated metrics and _evaluation_end=True

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
        threshold = self.threshold

        for batch_idx, batch in enumerate(loader):
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
            batch_loss = loss.item()
            loss_sum += batch_loss * batch_size
            n_total += batch_size

            # Normalize logits shape: handle (N, 1) or (N, 2) → (N,)
            logits = self._normalize_logits(logits)
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

            # Yield per-batch metrics for pipeline logging
            yield {
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "loss": batch_loss,
            }

        # Compute final results
        results: Dict[str, float] = {}

        # Add loss
        if "loss" in self.metrics_list:
            results["loss"] = loss_sum / n_total if n_total > 0 else 0.0

        # Compute and add all other metrics
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.compute().detach().cpu().item()

        # Clear metric state to release memory before the next split/run
        for metric in self.metrics.values():
            metric.reset()

        # Yield final aggregated metrics with sentinel
        results["_evaluation_end"] = True
        yield results
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
