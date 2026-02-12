"""Early stopping helper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track metric improvements and trigger early stop.

    Attributes:
        patience: Number of non-improving epochs tolerated.
        mode: Optimization direction (`max` or `min`).
        best_value: Best observed metric value.
        no_improve_epochs: Count of consecutive non-improving epochs.
    """

    patience: int
    mode: str = "max"
    best_value: float | None = None
    no_improve_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Update with a metric value.

        Args:
            value: Current epoch metric value.

        Returns:
            Tuple of ``(improved, should_stop)``.

        Raises:
            ValueError: If ``mode`` is invalid.
        """
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

        improved = False
        if (
            self.best_value is None
            or self.mode == "max"
            and value > self.best_value
            or self.mode == "min"
            and value < self.best_value
        ):
            improved = True

        if improved:
            self.best_value = value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

        should_stop = self.no_improve_epochs >= self.patience
        return improved, should_stop
