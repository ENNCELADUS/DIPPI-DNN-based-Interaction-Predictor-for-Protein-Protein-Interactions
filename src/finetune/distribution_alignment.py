"""
Distribution alignment helpers for finetune stage.

Implements a lightweight version of DisAlign for the binary interaction task.
The aligner collects validation logits, applies a shift so the mean probability
matches a target prior, then searches for the best threshold using a chosen
metric (default F1).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


class DistributionAligner:
    """Distribution Alignment for binary classification under prior shift."""

    _RANK_METRICS = {"auprc", "auroc"}
    _VALID_METRICS = {"f1", "balanced_accuracy", "mcc", *_RANK_METRICS}

    def __init__(
        self,
        target_prior: float = 0.5,
        search_metric: str = "f1",
        search_steps: int = 100,
    ) -> None:
        """
        Args:
            target_prior: Target positive prior for calibration (0 < prior < 1).
            search_metric: Metric used to select the best threshold
                           ("f1", "balanced_accuracy", "mcc", "auprc", "auroc").
            search_steps: Number of thresholds uniformly sampled in [0, 1].
        """
        if not 0 < target_prior < 1:
            raise ValueError("target_prior must be within (0, 1)")
        metric = search_metric.lower()
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"search_metric must be one of {sorted(self._VALID_METRICS)}, got {search_metric}"
            )
        if search_steps < 2:
            raise ValueError("search_steps must be >= 2 to evaluate thresholds")

        self.target_prior = float(target_prior)
        self.search_metric = metric
        self.search_steps = search_steps

    def calibrate_and_search(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate predictions to the target prior and search for an optimal threshold.

        Args:
            model: Model already set to eval mode.
            val_loader: Validation DataLoader (no sampling).
            device: Device for inference.

        Returns:
            Dictionary with bias, threshold, loss, priors, and metrics.
        """
        logits, labels, avg_loss = self._collect_logits_and_loss(
            model, val_loader, device, amp_dtype
        )

        if logits.numel() == 0:
            raise ValueError("Validation loader produced no samples for DA calibration")

        probabilities = torch.sigmoid(logits)
        predicted_prior = probabilities.mean().item()

        bias = self._compute_bias(logits, predicted_prior)
        calibrated_logits = logits + bias
        calibrated_probs = torch.sigmoid(calibrated_logits)
        calibrated_prior = calibrated_probs.mean().item()

        threshold = self._search_threshold(calibrated_probs, labels)
        metrics = self._compute_metrics(calibrated_probs, labels, threshold)

        return {
            "bias": bias,
            "threshold": threshold,
            "loss": avg_loss,
            "predicted_prior": predicted_prior,
            "calibrated_prior": calibrated_prior,
            "metrics": metrics,
        }

    def evaluate_with_bias(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        bias: float,
        threshold: float,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Apply stored bias/threshold to loader and compute metrics."""
        logits, labels, avg_loss = self._collect_logits_and_loss(
            model, loader, device, amp_dtype
        )
        calibrated_logits = logits + bias
        calibrated_probs = torch.sigmoid(calibrated_logits)
        metrics = self._compute_metrics(calibrated_probs, labels, threshold)
        return {"loss": avg_loss, "metrics": metrics}

    def _collect_logits_and_loss(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run inference and accumulate logits, labels, and average loss."""
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        loss_sum = 0.0
        n_total = 0

        for batch in loader:
            batch = self._move_batch_to_device(batch, device)
            labels = self._extract_labels(batch)
            if labels is None:
                raise ValueError("Validation batch missing 'label' or 'labels' key")
            model_inputs = {
                k: v for k, v in batch.items() if k not in {"label", "labels"}
            }

            context = (
                torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )

            with context:
                outputs = model(model_inputs)
            logits = self._normalize_logits(outputs["logits"])

            if "loss" in outputs:
                loss_tensor = outputs["loss"]
            else:
                loss_tensor = F.binary_cross_entropy_with_logits(logits, labels.float())

            batch_size = labels.size(0)
            loss_sum += loss_tensor.item() * batch_size
            n_total += batch_size

            # Ensure tensors are on CPU with float32 dtype for downstream numpy conversion
            all_logits.append(logits.detach().to(dtype=torch.float32, device="cpu"))
            all_labels.append(labels.detach().to(device="cpu"))

        stacked_logits = (
            torch.cat(all_logits, dim=0) if all_logits else torch.tensor([])
        )
        stacked_labels = (
            torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        )
        avg_loss = loss_sum / n_total if n_total > 0 else 0.0

        return stacked_logits, stacked_labels, avg_loss

    def _search_threshold(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Search threshold that maximizes the configured metric."""
        probs_np = probs.detach().to(dtype=torch.float32, device="cpu").numpy()
        labels_np = labels.detach().to(device="cpu").numpy()

        # Metrics like AUROC/AUPRC are threshold-independent; keep default threshold.
        if self.search_metric in self._RANK_METRICS:
            return 0.5

        thresholds = np.linspace(0.0, 1.0, num=self.search_steps, dtype=np.float32)
        best_metric = float("-inf")
        best_threshold = 0.5

        for thr in thresholds:
            preds = (probs_np >= thr).astype(int)
            metric_value = self._compute_search_metric(preds, labels_np, probs_np)
            if metric_value > best_metric or (
                math.isclose(metric_value, best_metric) and thr < best_threshold
            ):
                best_metric = metric_value
                best_threshold = float(thr)

        return best_threshold

    def _compute_metrics(
        self, probs: torch.Tensor, labels: torch.Tensor, threshold: float
    ) -> Dict[str, float]:
        """Compute evaluation metrics given calibrated probabilities and threshold."""
        probs_np = probs.detach().to(dtype=torch.float32, device="cpu").numpy()
        labels_np = labels.detach().to(device="cpu").numpy()
        preds_np = (probs_np >= threshold).astype(int)

        metrics = {
            "f1": f1_score(labels_np, preds_np, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(labels_np, preds_np),
            "mcc": matthews_corrcoef(labels_np, preds_np),
            "precision": precision_score(labels_np, preds_np, zero_division=0),
            "recall": recall_score(labels_np, preds_np, zero_division=0),
            "auroc": self._safe_auc(labels_np, probs_np),
            "auprc": self._safe_average_precision(labels_np, probs_np),
        }
        return metrics

    def _compute_search_metric(
        self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray
    ) -> float:
        """Return metric value used in threshold search."""
        if self.search_metric == "f1":
            return f1_score(labels, preds, zero_division=0)
        if self.search_metric == "balanced_accuracy":
            return balanced_accuracy_score(labels, preds)
        if self.search_metric == "auprc":
            return self._safe_average_precision(labels, probs)
        if self.search_metric == "auroc":
            return self._safe_auc(labels, probs)
        return matthews_corrcoef(labels, preds)

    def _compute_bias(self, logits: torch.Tensor, predicted_prior: float) -> float:
        """Binary search bias such that mean sigmoid(logits + bias) ~= target."""
        target = self.target_prior
        if math.isclose(predicted_prior, target, rel_tol=1e-6, abs_tol=1e-6):
            return 0.0

        low = -20.0
        high = 20.0
        for _ in range(60):
            mid = (low + high) / 2.0
            mean_prob = torch.sigmoid(logits + mid).mean().item()
            if mean_prob < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    @staticmethod
    def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute AUROC with graceful fallback when one class present."""
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return float("nan")

    @staticmethod
    def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute AUPRC with graceful fallback when one class present."""
        if len(np.unique(labels)) < 2:
            return float("nan")
        try:
            return float(average_precision_score(labels, scores))
        except ValueError:
            return float("nan")

    @staticmethod
    def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        """Normalize logits to shape (N,) regardless of input layout."""
        if logits.dim() == 2 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        if logits.dim() == 2 and logits.size(-1) == 2:
            return logits[:, 1]
        return logits.view(-1)

    @staticmethod
    def _move_batch_to_device(
        batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """Move tensor values in batch dict to the specified device."""
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _extract_labels(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract labels regardless of whether batch uses 'label' or 'labels' key."""
        labels = batch.get("label")
        if labels is None:
            labels = batch.get("labels")
        if labels is None:
            return None
        if labels.dim() > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        return labels.long()


__all__ = ["DistributionAligner"]
