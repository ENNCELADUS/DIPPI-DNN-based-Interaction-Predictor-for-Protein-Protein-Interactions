"""Training strategy callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from src.train.base import Trainer


class TrainingStrategy:
    """Base callback interface for training policies."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before training loop.

        Args:
            trainer: Trainer instance being orchestrated.
        """

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            trainer: Trainer instance being orchestrated.
            epoch: Zero-based epoch index.
        """

    def on_epoch_end(self, trainer: Trainer, epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: Trainer instance being orchestrated.
            epoch: Zero-based epoch index.
        """


class NoOpStrategy(TrainingStrategy):
    """Default strategy with no policy changes."""


def _set_trainable_by_prefix(
    model: nn.Module,
    trainable_prefixes: tuple[str, ...],
) -> None:
    """Set parameter ``requires_grad`` by prefix rules.

    Args:
        model: Target model.
        trainable_prefixes: Name prefixes that should remain trainable.
    """

    def _matches(name: str) -> bool:
        return any(
            name.startswith(prefix) or name.startswith(f"module.{prefix}")
            for prefix in trainable_prefixes
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = _matches(name)


@dataclass
class StagedUnfreezeStrategy(TrainingStrategy):
    """Freeze most layers, then unfreeze all at a configured epoch."""

    unfreeze_epoch: int = 1
    initial_trainable_prefixes: tuple[str, ...] = ("output_head",)
    _has_unfroze: bool = False

    def on_train_begin(self, trainer: Trainer) -> None:
        """Freeze to initial trainable prefixes before training.

        Args:
            trainer: Trainer instance being orchestrated.
        """
        _set_trainable_by_prefix(
            model=trainer.model,
            trainable_prefixes=self.initial_trainable_prefixes,
        )
        trainer.rebuild_optimizer_and_scheduler()

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Unfreeze all layers at or after ``unfreeze_epoch``.

        Args:
            trainer: Trainer instance being orchestrated.
            epoch: Zero-based epoch index.
        """
        if self._has_unfroze:
            return
        if epoch >= self.unfreeze_epoch:
            for parameter in trainer.model.parameters():
                parameter.requires_grad = True
            trainer.rebuild_optimizer_and_scheduler()
            self._has_unfroze = True
