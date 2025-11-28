"""
Strategy classes for fine-tune policies in DIPPI pipeline.

This module provides pluggable fine-tune helpers (callbacks) that decide
when/how to freeze/unfreeze modules, rebuild optimizer/scheduler, apply LLRD, etc.

Strategies implement lifecycle hooks (on_train_begin, on_epoch_begin/end) and
manipulate trainer state (model requires_grad, optimizer_cfg, scheduler_cfg) but
do NOT perform logging, metrics, checkpointing, or config parsing.

Design follows patterns from docs/trainer.md.
"""

from typing import Any, Dict, List

import copy
import logging

import torch.nn as nn

from src.utils.distributed import is_main_process


class BaseStrategy:
    """
    Base strategy with no-op lifecycle hooks.

    Subclasses override hooks to implement fine-tune policies like
    staged unfreezing, LLRD, or custom training dynamics.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """
        Called once before training starts.

        Args:
            trainer: Trainer instance with access to model, optimizer_cfg, etc.
        """
        pass

    def on_epoch_begin(self, trainer: Any, epoch_idx: int) -> None:
        """
        Called at the start of each epoch.

        Args:
            trainer: Trainer instance
            epoch_idx: Current epoch index (0-based)
        """
        pass

    def on_epoch_end(self, trainer: Any, epoch_idx: int) -> None:
        """
        Called at the end of each epoch.

        Args:
            trainer: Trainer instance
            epoch_idx: Current epoch index (0-based)
        """
        pass


class StagedUnfreeze(BaseStrategy):
    """
    Strategy for staged unfreezing of model parameters during fine-tuning.

    This strategy applies freeze/unfreeze patterns at specified epoch boundaries
    and rebuilds optimizer/scheduler when parameter groups change.

    Schedule format (list of dicts):
        [
            {
                "at_epoch": 0,              # Apply at start of this epoch (0-indexed)
                "freeze": ["layer1.*"],     # Optional: patterns to freeze
                "unfreeze": ["head.*"],     # Optional: patterns to unfreeze
                "optimizer_cfg": {...},     # Optional: new optimizer config
                "scheduler_cfg": {...},     # Optional: new scheduler config
            },
            {
                "at_epoch": 3,
                "unfreeze": ["encoder.*"],
                "optimizer_cfg": {"type": "adamw", "lr": 1e-5, ...},
            },
            ...
        ]

    Pattern matching uses simple substring matching: a parameter name like
    "encoder.layer.0.weight" matches pattern "encoder." or "layer.0".

    When both freeze and unfreeze are present, freeze patterns are applied first,
    then unfreeze patterns, allowing selective unfreezing after broad freezing.
    """

    def __init__(self, schedule: List[Dict[str, Any]]):
        """
        Initialize staged unfreeze strategy.

        Args:
            schedule: List of schedule entries (dicts) with keys:
                - at_epoch (int, required): Epoch index to apply changes
                - freeze (list[str], optional): Patterns to freeze
                - unfreeze (list[str], optional): Patterns to unfreeze
                - optimizer_cfg (dict, optional): New optimizer config
                - scheduler_cfg (dict, optional): New scheduler config

        Raises:
            ValueError: If schedule is empty or malformed
        """
        if not isinstance(schedule, list) or len(schedule) == 0:
            raise ValueError("Schedule must be a non-empty list of dicts")

        for i, entry in enumerate(schedule):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Schedule entry {i} must be a dict, got {type(entry)}"
                )
            if "at_epoch" not in entry:
                raise ValueError(f"Schedule entry {i} missing required 'at_epoch' key")
            if not isinstance(entry["at_epoch"], int):
                raise ValueError(
                    f"Schedule entry {i} 'at_epoch' must be int, got {type(entry['at_epoch'])}"
                )

        self.schedule = schedule
        self._applied_indices: set[int] = set()

    def on_train_begin(self, trainer: Any) -> None:
        """
        Apply any schedule entries targeting epoch 0 before the first batch.
        """
        self._apply_schedule_entries(trainer, epoch_idx=0, hook="on_train_begin")

    def on_epoch_begin(self, trainer: Any, epoch_idx: int) -> None:
        """
        Apply schedule entries at the start of an epoch (before forward passes).
        """
        self._apply_schedule_entries(
            trainer, epoch_idx=epoch_idx, hook="on_epoch_begin"
        )

    def on_epoch_end(self, trainer: Any, epoch_idx: int) -> None:
        """
        Retained for API compatibility; updates are handled at epoch start.
        """
        return

    def _apply_schedule_entries(self, trainer: Any, epoch_idx: int, hook: str) -> None:
        """
        Apply matching schedule entries for the given epoch index.
        """
        for entry_idx, entry in enumerate(self.schedule):
            if entry_idx in self._applied_indices:
                continue
            if entry["at_epoch"] != epoch_idx:
                continue
            self._apply_entry(trainer, entry, entry_idx, epoch_idx, hook)

    def _apply_entry(
        self,
        trainer: Any,
        entry: Dict[str, Any],
        entry_idx: int,
        epoch_idx: int,
        hook: str,
    ) -> None:
        """
        Apply a single schedule entry and rebuild optimizer/scheduler if needed.
        """
        freeze_patterns = entry.get("freeze", [])
        unfreeze_patterns = entry.get("unfreeze", [])

        has_param_updates = bool(freeze_patterns or unfreeze_patterns)
        if has_param_updates:
            self._apply_freeze_plan(trainer.model, freeze_patterns, unfreeze_patterns)

        needs_rebuild = has_param_updates

        if "optimizer_cfg" in entry:
            trainer.optimizer_cfg = copy.deepcopy(entry["optimizer_cfg"])
            needs_rebuild = True

        if "scheduler_cfg" in entry:
            trainer.scheduler_cfg = copy.deepcopy(entry["scheduler_cfg"])
            needs_rebuild = True

        if needs_rebuild:
            trainer.rebuild_optimizer_and_scheduler()

        self._applied_indices.add(entry_idx)
        self._log_trainable_parameters(trainer, epoch_idx, hook)

    def _apply_freeze_plan(
        self,
        model: nn.Module,
        freeze_patterns: List[str],
        unfreeze_patterns: List[str],
    ) -> None:
        """
        Apply freeze/unfreeze patterns to model parameters.

        Patterns are matched using substring search: parameter name
        "encoder.layer.0.weight" matches pattern "encoder." or "layer.0".

        Freeze patterns are applied first, then unfreeze patterns.

        Args:
            model: PyTorch model to modify
            freeze_patterns: List of patterns to freeze (set requires_grad=False)
            unfreeze_patterns: List of patterns to unfreeze (set requires_grad=True)
        """
        # Apply freeze patterns first
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False

        # Then apply unfreeze patterns (can override freeze)
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in unfreeze_patterns):
                param.requires_grad = True

    def _log_trainable_parameters(
        self, trainer: Any, epoch_idx: int, hook: str
    ) -> None:
        """
        Log the number of trainable parameters after applying a schedule entry.
        """
        if not is_main_process():
            return

        num_trainable = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        logger = logging.getLogger(__name__)
        logger.info(
            "StagedUnfreeze applied at epoch %d via %s; trainable parameters: %s",
            epoch_idx,
            hook,
            f"{num_trainable:,}",
        )
