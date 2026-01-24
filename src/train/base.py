"""Lightweight Trainer used by pretrain/finetune stages.

This implementation keeps the surface area minimal:
- Build optimizer/scheduler from simple config dicts or callables.
- Optional AMP support (fp16/bf16).
- Support for LLRD param groups via substring pattern matching.
- One-epoch training loop that returns average loss and current LR.

It intentionally avoids orchestration concerns (logging, checkpointing, etc.).
Those are owned by the stage runners in src/stages/.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "BaseTrainer"]


class Trainer:
    """Minimal trainer used by run_pretrain/run_finetune."""

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        amp_cfg: Optional[Dict[str, Any]] = None,
        strategy: Any = None,
        max_norm: Optional[float] = None,
        loss_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer_cfg = copy.deepcopy(optimizer_cfg)
        self.scheduler_cfg = copy.deepcopy(scheduler_cfg)
        self.strategy = strategy
        self.max_norm = max_norm
        self.loss_cfg = copy.deepcopy(loss_cfg) if loss_cfg else None

        amp_cfg = amp_cfg or {}
        self.use_amp = bool(amp_cfg.get("enabled", False))
        self.amp_dtype_str = str(amp_cfg.get("dtype", "bf16")).lower()
        self.amp_dtype = (
            torch.float16
            if self.amp_dtype_str in {"fp16", "float16", "half"}
            else torch.bfloat16
        )
        # On non-CUDA devices, keep AMP enabled but prefer bf16 over fp16.
        if self.use_amp and device.type != "cuda":
            if self.amp_dtype == torch.float16:
                logger.warning(
                    "AMP fp16 requested on non-CUDA device; switching to bf16."
                )
                self.amp_dtype = torch.bfloat16
        if self.use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler(self.device.type)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        self.optimizer = self._build_optimizer()
        self.scheduler, self._scheduler_step_per_batch = self._build_scheduler()

    # ------------------------------------------------------------------ #
    # Optimizer / Scheduler builders
    # ------------------------------------------------------------------ #
    def _trainable_params(self) -> List[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _build_param_groups(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_lr = cfg.get("lr", 1e-3)
        base_wd = cfg.get("weight_decay", 0.0)
        param_groups_cfg = cfg.get("param_groups") or []

        named_params = self._trainable_params()
        if not named_params:
            raise ValueError("No trainable parameters found in model.")

        # No custom groups: single group with all params
        if not param_groups_cfg:
            return [
                {
                    "params": [p for _, p in named_params],
                    "lr": base_lr,
                    "weight_decay": base_wd,
                }
            ]

        assigned: set[int] = set()
        groups: List[Dict[str, Any]] = []

        for group_cfg in param_groups_cfg:
            pattern = group_cfg.get("pattern")
            if not pattern:
                continue
            group_params = [
                p
                for name, p in named_params
                if pattern in name and id(p) not in assigned
            ]
            for p in group_params:
                assigned.add(id(p))
            if not group_params:
                continue
            groups.append(
                {
                    "params": group_params,
                    "lr": group_cfg.get("lr", base_lr),
                    "weight_decay": group_cfg.get("weight_decay", base_wd),
                }
            )

        # Add remaining params to a base group
        remaining = [p for _, p in named_params if id(p) not in assigned]
        if remaining:
            groups.append({"params": remaining, "lr": base_lr, "weight_decay": base_wd})

        # Validate coverage: each trainable param should appear exactly once
        total_group_params = sum(p.numel() for g in groups for p in g["params"])
        total_model_params = sum(p.numel() for _, p in named_params)
        if total_group_params != total_model_params:
            raise ValueError("Parameter grouping failed: mismatch in parameter counts.")

        return groups

    def _build_optimizer(self) -> Optimizer:
        cfg = self.optimizer_cfg
        if callable(cfg):
            optim = cfg()
            if not isinstance(optim, Optimizer):
                raise ValueError(
                    "Callable optimizer_cfg must return a torch.optim.Optimizer."
                )
            return optim

        if not isinstance(cfg, dict):
            raise ValueError("optimizer_cfg must be a dict or callable.")

        optim_type = str(cfg.get("type", "adamw")).lower()
        lr = cfg.get("lr", 1e-3)
        weight_decay = cfg.get("weight_decay", 0.0)
        beta_vals = None
        if "beta1" in cfg or "beta2" in cfg:
            beta_vals = (cfg.get("beta1", 0.9), cfg.get("beta2", 0.999))

        param_groups = self._build_param_groups(cfg)

        common_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
        if beta_vals is not None:
            common_kwargs["betas"] = beta_vals

        if optim_type == "adamw":
            return torch.optim.AdamW(param_groups, **common_kwargs)
        if optim_type == "adam":
            return torch.optim.Adam(param_groups, **common_kwargs)
        if optim_type == "sgd":
            momentum = cfg.get("momentum", 0.0)
            return torch.optim.SGD(
                param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum
            )

        raise ValueError(f"Unknown optimizer type: {optim_type}")

    def _build_warmup_cosine_scheduler(
        self, optimizer: Optimizer, cfg: Dict[str, Any]
    ) -> LambdaLR:
        total_steps = int(cfg.get("num_training_steps", 0))
        warmup_steps = int(cfg.get("num_warmup_steps", 0))
        num_cycles = float(cfg.get("num_cycles", 0.5))
        if total_steps <= 0:
            raise ValueError("warmup_cosine scheduler requires num_training_steps > 0.")

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
            )

        return LambdaLR(optimizer, lr_lambda)

    def _build_scheduler(
        self,
    ) -> tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
        cfg = self.scheduler_cfg
        if cfg is None:
            return None, False
        if callable(cfg):
            scheduler = cfg(self.optimizer)
            return scheduler, False
        if not isinstance(cfg, dict):
            raise ValueError("scheduler_cfg must be a dict, callable, or None.")

        sched_type = str(cfg.get("type", "")).lower()
        step_per_batch = False

        if sched_type in {"onecycle", "onecyclelr"}:
            if "total_steps" in cfg:
                total_steps = cfg["total_steps"]
                scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=cfg.get("max_lr", 1e-3),
                    total_steps=total_steps,
                    **{
                        k: v
                        for k, v in cfg.items()
                        if k not in {"type", "max_lr", "total_steps"}
                    },
                )
            elif "steps_per_epoch" in cfg and "epochs" in cfg:
                scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=cfg.get("max_lr", 1e-3),
                    steps_per_epoch=cfg["steps_per_epoch"],
                    epochs=cfg["epochs"],
                    **{
                        k: v
                        for k, v in cfg.items()
                        if k not in {"type", "max_lr", "steps_per_epoch", "epochs"}
                    },
                )
            else:
                raise ValueError(
                    "OneCycleLR requires total_steps or (steps_per_epoch and epochs)."
                )
            step_per_batch = True
        elif sched_type == "cosineannealinglr":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.get("T_max", 10))
        elif sched_type == "steplr":
            scheduler = StepLR(
                self.optimizer,
                step_size=cfg.get("step_size", 10),
                gamma=cfg.get("gamma", 0.1),
            )
        elif sched_type == "multisteplr":
            scheduler = MultiStepLR(
                self.optimizer,
                milestones=cfg.get("milestones", []),
                gamma=cfg.get("gamma", 0.1),
            )
        elif sched_type in {"warmup_cosine", "cosine_warmup"}:
            scheduler = self._build_warmup_cosine_scheduler(self.optimizer, cfg)
            step_per_batch = True
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        return scheduler, step_per_batch

    def rebuild_optimizer_and_scheduler(self) -> None:
        """Rebuild optimizer and scheduler (used by fine-tune strategies)."""
        self.optimizer = self._build_optimizer()
        self.scheduler, self._scheduler_step_per_batch = self._build_scheduler()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train_one_epoch(self, loader: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train for one epoch and return aggregated metrics.

        Returns:
            Dict with keys: loss (avg), lr
        """
        final_metrics = None
        for batch_metrics in self.train_one_epoch_iter(loader):
            if batch_metrics.get("_epoch_end", False):
                final_metrics = batch_metrics
                break

        if final_metrics is None:
            raise RuntimeError("Trainer did not yield epoch summary")

        final_metrics = dict(final_metrics)
        final_metrics.pop("_epoch_end", None)
        return final_metrics

    def train_one_epoch_iter(
        self, loader: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        """
        Train for one epoch, yielding per-batch metrics.

        Yields:
            Per-batch dict with keys: batch_idx, loss, lr, batch_size
            Final dict with keys: loss (avg), lr, _epoch_end=True
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(loader):
            if self._is_ohem_batch(batch):
                batch = self._prepare_ohem_batch(batch)
            else:
                batch = self._move_batch_to_device(batch)

            if self.use_amp:
                with self._autocast():
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)

            if loss is None:
                raise ValueError(
                    "Model must return a 'loss' tensor or Trainer must be configured with loss_cfg."
                )

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
                self.optimizer.step()

            if self.scheduler is not None and self._scheduler_step_per_batch:
                self.scheduler.step()

            batch_loss = float(loss.detach().item())
            total_loss += batch_loss
            total_batches += 1

            # Yield per-batch metrics for pipeline logging
            batch_size = batch.get("label", next(iter(batch.values()))).size(0)
            yield {
                "batch_idx": batch_idx,
                "loss": batch_loss,
                "lr": self._current_lr(),
                "batch_size": batch_size,
            }

        if self.scheduler is not None and not self._scheduler_step_per_batch:
            self.scheduler.step()

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        current_lr = self._current_lr()

        # Yield final aggregated metrics with sentinel
        final_metrics = {"loss": avg_loss, "lr": current_lr, "_epoch_end": True}
        yield final_metrics

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _autocast(self):
        # Prefer the unified torch.amp.autocast API; fallback keeps older torch versions working.
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            )
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _compute_loss(
        self, outputs: Any, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif self.loss_cfg:
            loss = self._compute_loss_from_cfg(outputs, batch)
        else:
            loss = None

        if loss is not None and not isinstance(loss, torch.Tensor):
            raise ValueError("Loss must be a torch.Tensor.")
        return loss

    def _compute_loss_from_cfg(
        self, outputs: Any, batch: Dict[str, Any]
    ) -> torch.Tensor:
        if not isinstance(outputs, dict) or "logits" not in outputs:
            raise ValueError(
                "Configured loss requires model outputs to include 'logits'."
            )
        if "label" not in batch:
            raise ValueError("Configured loss requires 'label' in batch.")

        logits = outputs["logits"]
        labels = batch["label"].float()
        loss = self._compute_bce_loss(logits, labels, reduction="mean")

        l1_lambda = float(self.loss_cfg.get("l1_lambda", 0.0))
        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + l1_lambda * l1_penalty

        return loss

    def _compute_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        labels = self._normalize_labels(labels)
        logits = self._normalize_logits(logits)

        smoothing = (
            float(self.loss_cfg.get("label_smoothing", 0.0)) if self.loss_cfg else 0.0
        )
        if smoothing > 0:
            labels = labels * (1.0 - smoothing) + 0.5 * smoothing

        pos_weight = self.loss_cfg.get("pos_weight") if self.loss_cfg else None
        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([float(pos_weight)], device=logits.device)

        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight_tensor,
            reduction=reduction,
        )

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            if logits.size(1) == 2:
                return logits[:, 1]
            if logits.size(1) == 1:
                return logits.squeeze(1)
        return logits

    def _normalize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        if labels.dim() > 1 and labels.size(-1) == 1:
            return labels.squeeze(-1)
        return labels

    def _is_ohem_batch(self, batch: Any) -> bool:
        return isinstance(batch, dict) and bool(batch.get("_ohem", False))

    def _prepare_ohem_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pos_batch = batch.get("pos")
        neg_candidates = batch.get("neg_candidates")
        neg_default = batch.get("neg_default")
        hard_count = int(batch.get("hard_count", 0))

        hard_neg_batch = None
        if neg_candidates is not None and hard_count > 0:
            was_training = self.model.training
            self.model.eval()
            neg_candidates = self._move_batch_to_device(neg_candidates)

            with torch.no_grad():
                with self._amp_context():
                    outputs = self.model(neg_candidates)
                    if not isinstance(outputs, dict) or "logits" not in outputs:
                        raise ValueError(
                            "OHEM mining requires model outputs to include 'logits'."
                        )
                    logits = outputs["logits"]
                    labels = neg_candidates["label"]
                    losses = self._compute_bce_loss(
                        logits, labels, reduction="none"
                    ).view(-1)

            if was_training:
                self.model.train()

            k = min(hard_count, int(losses.numel()))
            if k > 0:
                _, topk_idx = torch.topk(losses, k=k, largest=True)
                hard_neg_batch = self._index_batch(neg_candidates, topk_idx)

        batches = []
        if pos_batch is not None:
            batches.append(self._move_batch_to_device(pos_batch))
        if hard_neg_batch is not None:
            batches.append(hard_neg_batch)
        if neg_default is not None:
            batches.append(self._move_batch_to_device(neg_default))

        if not batches:
            raise RuntimeError("OHEM batch produced no samples for training.")

        return self._concat_batches(batches)

    def _index_batch(
        self, batch: Dict[str, Any], indices: torch.Tensor
    ) -> Dict[str, Any]:
        if indices.numel() == 0:
            return {}
        return {
            k: v.index_select(0, indices) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _concat_batches(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batches:
            return {}
        keys = [k for k in batches[0].keys() if isinstance(batches[0][k], torch.Tensor)]
        merged: Dict[str, Any] = {}
        for key in keys:
            merged[key] = torch.cat([b[key] for b in batches if key in b], dim=0)
        return merged

    def _amp_context(self):
        if self.use_amp:
            return self._autocast()
        return contextlib.nullcontext()

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

    def _current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))


# Backward compatibility: allow imports of BaseTrainer to reference Trainer
BaseTrainer = Trainer
