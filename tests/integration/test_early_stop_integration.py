"""
Integration tests for early stopping in training loops.

Tests the integration of check_early_stop with realistic training scenarios:
- Simulated training loop with metric tracking
- Integration with run.py pattern (metric history building)
- Edge cases in multi-epoch training
"""

import pytest

from src.utils.early_stop import check_early_stop


class TestEarlyStopTrainingLoopIntegration:
    """Test early stopping in simulated training loop scenarios."""

    def test_simple_training_loop_stops_correctly(self):
        """Simulate a basic training loop that should trigger early stopping."""
        # Simulate validation metrics over epochs
        patience = 3
        monitor_metric = "val_loss"
        mode = "min"
        metric_history = []

        # Simulated epoch validation results
        val_losses = [0.5, 0.4, 0.35, 0.33, 0.33, 0.33, 0.33, 0.32]

        stopped_at_epoch = None
        for epoch, val_loss in enumerate(val_losses):
            metric_history.append(val_loss)

            # Check early stopping (as run.py does)
            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=mode,
            ):
                stopped_at_epoch = epoch
                break

        # Should stop at epoch 6 (0-indexed), after 3 consecutive plateaus at 0.33
        assert stopped_at_epoch == 6

    def test_training_loop_with_late_improvement_no_stop(self):
        """Training improves late, should not stop prematurely."""
        patience = 3
        mode = "max"
        metric_history = []

        # Accuracy: plateaus then improves
        val_accs = [0.7, 0.75, 0.76, 0.76, 0.76, 0.82, 0.85]

        stopped_at_epoch = None
        for epoch, val_acc in enumerate(val_accs):
            metric_history.append(val_acc)

            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor="val_acc",
                mode=mode,
            ):
                stopped_at_epoch = epoch
                break

        # Should not stop (improvement at epoch 5 resets counter)
        assert stopped_at_epoch is None

    def test_training_loop_with_patience_zero(self):
        """Patience=0 should stop immediately on first plateau."""
        patience = 0
        mode = "min"
        metric_history = []

        val_losses = [0.5, 0.4, 0.4, 0.3]  # Plateaus at second 0.4

        stopped_at_epoch = None
        for epoch, val_loss in enumerate(val_losses):
            metric_history.append(val_loss)

            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor="val_loss",
                mode=mode,
            ):
                stopped_at_epoch = epoch
                break

        # Should stop at epoch 2 (0-indexed) when second 0.4 appears
        assert stopped_at_epoch == 2

    def test_full_training_no_early_stop(self):
        """Training that never triggers early stopping."""
        patience = 5
        mode = "min"
        metric_history = []

        # Continuous improvement
        val_losses = [0.5, 0.45, 0.4, 0.36, 0.33, 0.31, 0.29, 0.27]

        stopped_at_epoch = None
        for epoch, val_loss in enumerate(val_losses):
            metric_history.append(val_loss)

            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor="val_loss",
                mode=mode,
            ):
                stopped_at_epoch = epoch
                break

        # Should complete all epochs without stopping
        assert stopped_at_epoch is None
        assert len(metric_history) == len(val_losses)


class TestEarlyStopRunPyPattern:
    """Test the exact pattern used in run.py."""

    def test_pretrain_pattern_min_mode(self):
        """Test pretrain loop pattern with loss monitoring (min mode)."""
        # This mimics the exact pattern in run.py's run_pretrain
        pretrain_cfg_mock = {
            "monitor_metric": "val_loss",
            "early_stopping_patience": 3,
        }

        monitor_metric = pretrain_cfg_mock["monitor_metric"]
        patience = pretrain_cfg_mock["early_stopping_patience"]
        monitor_mode = "min" if "loss" in monitor_metric else "max"
        metric_history = []

        # Simulate epochs with validation
        epochs_val_metrics = [
            {"val_loss": 0.8, "val_auroc": 0.7},
            {"val_loss": 0.6, "val_auroc": 0.75},
            {"val_loss": 0.5, "val_auroc": 0.78},
            {"val_loss": 0.5, "val_auroc": 0.78},
            {"val_loss": 0.5, "val_auroc": 0.78},
            {"val_loss": 0.5, "val_auroc": 0.78},
            {"val_loss": 0.49, "val_auroc": 0.79},  # Won't reach here
        ]

        stopped = False
        completed_epochs = 0

        for epoch, val_metrics in enumerate(epochs_val_metrics):
            # Extract monitored metric
            current_metric = val_metrics.get(monitor_metric)

            # Append to history
            metric_history.append(current_metric)

            # Check early stopping
            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            ):
                stopped = True
                completed_epochs = epoch + 1
                break

        assert stopped is True
        assert (
            completed_epochs == 6
        )  # Stops at epoch 5 (0-indexed = epoch 5, 1-indexed = 6)

    def test_finetune_pattern_max_mode(self):
        """Test finetune loop pattern with AUROC monitoring (max mode)."""
        # This mimics the exact pattern in run.py's run_finetune
        finetune_cfg_mock = {
            "monitor_metric": "val_auroc",
            "early_stopping_patience": 2,
        }

        monitor_metric = finetune_cfg_mock["monitor_metric"]
        patience = finetune_cfg_mock["early_stopping_patience"]
        monitor_mode = "min" if "loss" in monitor_metric else "max"
        metric_history = []

        # Simulate epochs with validation
        epochs_val_metrics = [
            {"val_loss": 0.4, "val_auroc": 0.82},
            {"val_loss": 0.38, "val_auroc": 0.85},
            {"val_loss": 0.36, "val_auroc": 0.87},
            {"val_loss": 0.35, "val_auroc": 0.87},
            {"val_loss": 0.34, "val_auroc": 0.86},
            {"val_loss": 0.33, "val_auroc": 0.88},  # Won't reach here
        ]

        stopped = False
        completed_epochs = 0

        for epoch, val_metrics in enumerate(epochs_val_metrics):
            # Extract monitored metric
            current_metric = val_metrics.get(monitor_metric)

            # Append to history
            metric_history.append(current_metric)

            # Check early stopping
            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            ):
                stopped = True
                completed_epochs = epoch + 1
                break

        assert stopped is True
        assert completed_epochs == 5  # Stops at epoch 4 (0-indexed)

    def test_monitor_metric_fallback_to_loss(self):
        """Test fallback when monitor_metric not in val_metrics."""
        # In run.py: current_metric = val_metrics.get(monitor_metric, val_metrics.get("loss"))
        # When monitor_metric key doesn't exist, fallback to "loss" key
        monitor_metric = "val_custom_metric"
        patience = 2
        monitor_mode = "min"  # min mode for loss
        metric_history = []

        epochs_val_metrics = [
            {"loss": 0.5},  # No custom metric key, falls back to loss
            {"loss": 0.4},
            {"loss": 0.4},
            {"loss": 0.4},
        ]

        stopped = False

        for epoch, val_metrics in enumerate(epochs_val_metrics):
            # Mimic run.py's fallback logic: get monitor_metric if present, else get loss
            current_metric = val_metrics.get(monitor_metric, val_metrics.get("loss"))
            metric_history.append(current_metric)

            # In this case we're monitoring loss in min mode
            if check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            ):
                stopped = True
                break

        # With min mode on loss values [0.5, 0.4, 0.4, 0.4], it sees:
        # 0.5 → 0.4 (improvement), 0.4 → 0.4 (no improvement), 0.4 → 0.4 (no improvement)
        # = 2 consecutive non-improvements, triggers stop
        assert stopped is True


class TestEarlyStopMultipleMonitorMetrics:
    """Test scenarios with different monitor metrics."""

    def test_loss_vs_auroc_different_stopping_points(self):
        """Same val metrics, different monitor choices lead to different stops."""
        patience = 3

        # Scenario: loss improves, auroc plateaus
        epochs_val_metrics = [
            {"val_loss": 0.6, "val_auroc": 0.80},
            {"val_loss": 0.5, "val_auroc": 0.82},
            {"val_loss": 0.45, "val_auroc": 0.83},
            {"val_loss": 0.42, "val_auroc": 0.83},
            {"val_loss": 0.40, "val_auroc": 0.83},
            {"val_loss": 0.38, "val_auroc": 0.83},
            {"val_loss": 0.36, "val_auroc": 0.84},
        ]

        # Test 1: Monitor loss (should not stop - loss keeps improving)
        metric_history_loss = []
        stopped_loss = False
        for val_metrics in epochs_val_metrics:
            metric_history_loss.append(val_metrics["val_loss"])
            if check_early_stop(metric_history_loss, patience, "val_loss", mode="min"):
                stopped_loss = True
                break
        assert stopped_loss is False

        # Test 2: Monitor AUROC (should stop - AUROC plateaus)
        metric_history_auroc = []
        stopped_auroc = False
        stopped_at_epoch_auroc = None
        for epoch, val_metrics in enumerate(epochs_val_metrics):
            metric_history_auroc.append(val_metrics["val_auroc"])
            if check_early_stop(
                metric_history_auroc, patience, "val_auroc", mode="max"
            ):
                stopped_auroc = True
                stopped_at_epoch_auroc = epoch
                break
        assert stopped_auroc is True
        assert stopped_at_epoch_auroc == 5  # Stops at epoch 5 (0-indexed)


class TestEarlyStopEdgeCasesInTraining:
    """Test edge cases that might occur during actual training."""

    def test_very_short_training_no_stop(self):
        """Training with fewer epochs than patience."""
        patience = 10
        metric_history = []

        # Only 3 epochs of training
        val_losses = [0.5, 0.5, 0.5]

        stopped = False
        for val_loss in val_losses:
            metric_history.append(val_loss)
            if check_early_stop(metric_history, patience, "val_loss", mode="min"):
                stopped = True
                break

        # Should not stop (need patience consecutive non-improvements)
        assert stopped is False

    def test_single_epoch_training_no_stop(self):
        """Single epoch should never trigger stop."""
        patience = 1
        metric_history = [0.5]

        result = check_early_stop(metric_history, patience, "val_loss", mode="min")
        assert result is False

    def test_exact_patience_epochs_of_plateau_from_start(self):
        """Plateau from the very beginning should trigger stop."""
        patience = 3
        metric_history = []

        # All same value from start
        val_losses = [0.5, 0.5, 0.5, 0.5]

        stopped = False
        stopped_at_epoch = None
        for epoch, val_loss in enumerate(val_losses):
            metric_history.append(val_loss)
            if check_early_stop(metric_history, patience, "val_loss", mode="min"):
                stopped = True
                stopped_at_epoch = epoch
                break

        assert stopped is True
        assert stopped_at_epoch == 3  # After 3 consecutive non-improvements

    def test_alternating_improvement_degradation(self):
        """Metrics that alternate between improving and degrading."""
        patience = 2
        metric_history = []

        # Alternating: improve, degrade, improve, degrade
        val_losses = [0.5, 0.4, 0.45, 0.35, 0.38, 0.3]

        stopped = False
        for val_loss in val_losses:
            metric_history.append(val_loss)
            if check_early_stop(metric_history, patience, "val_loss", mode="min"):
                stopped = True
                break

        # Should not stop (improvements reset counter each time)
        assert stopped is False
