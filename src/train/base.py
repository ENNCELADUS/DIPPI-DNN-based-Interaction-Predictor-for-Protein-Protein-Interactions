"""Generic Trainer implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from src.utils.losses import LossConfig, binary_classification_loss
from src.utils.ohem_sample_strategy import OHEMSampleStrategy


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer parameters.

    Attributes:
        optimizer_type: Optimizer name.
        lr: Base learning rate.
        beta1: Beta1 for Adam-like optimizers.
        beta2: Beta2 for Adam-like optimizers.
        eps: Numerical stability epsilon.
        weight_decay: Weight decay coefficient.
    """

    optimizer_type: str
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler parameters.

    Attributes:
        scheduler_type: Scheduler name.
        max_lr: Max learning rate for schedule.
        pct_start: OneCycle warmup fraction.
        div_factor: Initial LR divisor.
        final_div_factor: Final LR divisor.
        anneal_strategy: OneCycle annealing strategy.
    """

    scheduler_type: str
    max_lr: float = 1e-3
    pct_start: float = 0.2
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    anneal_strategy: Literal["cos", "linear"] = "cos"


class Trainer:
    """Mechanics-only training class for one epoch execution.

    This class handles forward/backward passes, optimizer/scheduler stepping,
    and optional AMP/OHEM behavior. It deliberately excludes orchestration
    concerns like checkpointing and stage transitions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
        loss_config: LossConfig,
        use_amp: bool,
        total_epochs: int,
        steps_per_epoch: int,
        ohem_strategy: OHEMSampleStrategy | None = None,
        logger: logging.Logger | None = None,
        heartbeat_every_n_steps: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.loss_config = loss_config
        self.use_amp = use_amp and device.type == "cuda"
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.ohem_strategy = ohem_strategy
        self.logger = logger
        self.heartbeat_every_n_steps = max(0, int(heartbeat_every_n_steps))
        self.scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            "cuda",
            enabled=self.use_amp,
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _build_optimizer(self) -> Optimizer:
        optimizer_type = self.optimizer_config.optimizer_type.lower()
        params = self._trainable_parameters()
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                params=params,
                lr=self.optimizer_config.lr,
                betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
                eps=self.optimizer_config.eps,
                weight_decay=self.optimizer_config.weight_decay,
            )
        if optimizer_type == "sgd":
            return torch.optim.SGD(
                params=params,
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_config.optimizer_type}")

    def _build_scheduler(self) -> LRScheduler | None:
        scheduler_type = self.scheduler_config.scheduler_type.lower()
        if scheduler_type in {"none", "disabled"}:
            return None
        if scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.scheduler_config.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.total_epochs,
                pct_start=self.scheduler_config.pct_start,
                div_factor=self.scheduler_config.div_factor,
                final_div_factor=self.scheduler_config.final_div_factor,
                anneal_strategy=self.scheduler_config.anneal_strategy,
            )
        raise ValueError(f"Unsupported scheduler type: {self.scheduler_config.scheduler_type}")

    def rebuild_optimizer_and_scheduler(self) -> None:
        """Rebuild optimizer and scheduler after trainable params change."""
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _forward_model(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        try:
            output = self.model(**batch)
        except TypeError:
            output = self.model(batch=batch)
        if not isinstance(output, dict):
            raise ValueError("Model forward output must be a dictionary")
        return output

    def _select_loss(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch_index: int,
    ) -> torch.Tensor:
        logits = output["logits"]
        labels = batch["label"].float()
        loss = binary_classification_loss(
            logits=logits,
            labels=labels,
            loss_config=self.loss_config,
            reduction="mean",
        )

        if self.ohem_strategy is None:
            return loss
        if epoch_index < self.ohem_strategy.warmup_epochs:
            return loss

        mining_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=0.0,
        )
        with torch.no_grad():
            mining_loss = binary_classification_loss(
                logits=logits,
                labels=labels,
                loss_config=mining_loss_config,
                reduction="none",
            )
            selected_indices = self.ohem_strategy.select(
                losses=mining_loss,
                epoch_index=epoch_index,
                protein_a_ids=batch.get("protein_a_id"),
                protein_b_ids=batch.get("protein_b_id"),
            )

        ohem_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=self.loss_config.label_smoothing,
        )
        per_sample = binary_classification_loss(
            logits=logits,
            labels=labels,
            loss_config=ohem_loss_config,
            reduction="none",
        )
        return per_sample[selected_indices].mean()

    def train_one_epoch(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        epoch_index: int = 0,
    ) -> dict[str, float]:
        """Run one full training epoch.

        Args:
            train_loader: Training data loader.
            epoch_index: Zero-based epoch index used for heartbeat log messages.

        Returns:
            Aggregate epoch metrics, including average loss and learning rate.
        """
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        total_steps = max(1, len(train_loader))

        for batch in train_loader:
            batch_count += 1
            prepared_batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self._forward_model(prepared_batch)
                loss = self._select_loss(
                    output=output,
                    batch=prepared_batch,
                    epoch_index=epoch_index,
                )

            if self.use_amp:
                scaled_loss = self.scaler.scale(loss)
                torch.autograd.backward(scaled_loss)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.autograd.backward(loss)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += float(loss.detach().item())
            if self._should_log_heartbeat(step=batch_count, total_steps=total_steps):
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                if self.logger is not None:
                    self.logger.info(
                        "Epoch %d | Step %d/%d | Loss %.4f | LR %.4e",
                        epoch_index + 1,
                        batch_count,
                        total_steps,
                        running_loss / batch_count,
                        current_lr,
                    )

        average_loss = running_loss / max(1, batch_count)
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        return {"loss": average_loss, "lr": current_lr}

    def _should_log_heartbeat(self, step: int, total_steps: int) -> bool:
        """Return whether heartbeat logs should be emitted for this step."""
        if self.logger is None:
            return False
        if step == 1 or step == total_steps:
            return True
        return self.heartbeat_every_n_steps > 0 and step % self.heartbeat_every_n_steps == 0
