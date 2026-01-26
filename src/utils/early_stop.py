"""Early stopping utility for training loops.

Role boundary (MVP):
  - This module provides a pure, stateless decision function.
  - run.py owns the training loop, builds the metric history from
    training_step.csv, and acts on the boolean result (log/stop/checkpoint).
  - No I/O, no state tracking, no logging happens here.

References:
  - Lightning AI EarlyStopping callback semantics
  - Keras EarlyStopping behavior
"""

from __future__ import annotations

from typing import Sequence


def check_early_stop(
    metrics: Sequence[float],
    patience: int,
    monitor: str,
    mode: str = "min",
    min_delta: float = 0.0,
) -> bool:
    """Return True if training should stop based on metric history.

    Contract (MVP):
      - `metrics` is the history of the monitored metric across epochs
        (e.g., values parsed by run.py from training_step.csv).
      - If the latest value has not improved over the best previous value
        by MORE than `min_delta` for `patience` consecutive checks, return True.
      - `mode="min"` means lower is better; `mode="max"` means higher is better.
      - A tie (equal value) is NOT an improvement when min_delta=0.0.

    Args:
        metrics: History of monitored metric values (one per epoch/validation).
                 Assumed to be sanitized (finite floats) by the orchestrator.
        patience: Number of consecutive non-improvements before stopping.
                  patience=0 stops on first non-improvement.
        monitor: Name of the metric being monitored (for clarity; used in error messages).
        mode: "min" for metrics where lower is better (e.g., loss),
              "max" for metrics where higher is better (e.g., accuracy, AUROC).
        min_delta: Minimum absolute change to qualify as an improvement.
                   Improvement requires change > min_delta (strict inequality).

    Returns:
        True if training should stop now, False otherwise.

    Raises:
        ValueError: If mode is not "min" or "max".

    Examples:
        >>> # Loss decreasing, then plateaus
        >>> check_early_stop([0.5, 0.4, 0.38, 0.37, 0.37, 0.38], patience=3, monitor="loss", mode="min")
        False  # Only 2 consecutive non-improvements at the end

        >>> # Loss plateaus for exactly patience epochs
        >>> check_early_stop([0.5, 0.4, 0.38, 0.37, 0.37, 0.37, 0.37], patience=3, monitor="loss", mode="min")
        True  # 3 consecutive non-improvements (epochs 4, 5, 6)

        >>> # Accuracy increasing, then plateaus
        >>> check_early_stop([0.7, 0.75, 0.78, 0.78, 0.77], patience=2, monitor="acc", mode="max")
        True  # 2 consecutive non-improvements (epochs 3, 4)
    """
    # Validate mode
    if mode not in {"min", "max"}:
        raise ValueError(
            f"mode must be 'min' or 'max' for monitor '{monitor}', got '{mode}'"
        )

    # Edge case: empty metrics
    if not metrics:
        return False

    # Initialize with first metric as baseline
    best_so_far = metrics[0]
    consecutive_no_improvement = 0

    # Scan through remaining metrics
    for metric in metrics[1:]:
        # Check if current metric improves over best_so_far
        # Use strict inequality: tie is NOT an improvement
        if mode == "min":
            improved = (best_so_far - metric) > min_delta
        else:  # mode == "max"
            improved = (metric - best_so_far) > min_delta

        if improved:
            # Update best and reset counter
            best_so_far = metric
            consecutive_no_improvement = 0
        else:
            # No improvement: increment counter
            consecutive_no_improvement += 1
            # Check if we've hit patience
            if consecutive_no_improvement >= patience:
                return True

    # If we finish the loop without hitting patience, don't stop
    return False
