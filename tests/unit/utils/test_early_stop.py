"""
Unit tests for src.utils.early_stop module.

Tests the pure stateless check_early_stop function with various scenarios:
- Basic min/max mode behavior
- Patience thresholds
- min_delta improvements
- Edge cases (empty, single value, patience=0)
- Error handling (invalid mode)
"""

import pytest

from src.utils.early_stop import check_early_stop


class TestCheckEarlyStopBasic:
    """Test basic early stopping behavior for min and max modes."""

    def test_min_mode_no_stop_improving(self):
        """Loss continuously improving, should not stop."""
        metrics = [0.5, 0.4, 0.3, 0.2, 0.1]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min"
        )
        assert result is False

    def test_min_mode_stops_after_patience(self):
        """Loss plateaus for exactly patience epochs, should stop."""
        metrics = [0.5, 0.4, 0.3, 0.3, 0.3, 0.3]  # 3 consecutive no-improvements
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min"
        )
        assert result is True

    def test_min_mode_stops_at_patience_boundary(self):
        """Loss plateaus for exactly patience-1, should not stop."""
        metrics = [0.5, 0.4, 0.3, 0.3, 0.3]  # Only 2 consecutive no-improvements
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min"
        )
        assert result is False

    def test_max_mode_no_stop_improving(self):
        """Accuracy continuously improving, should not stop."""
        metrics = [0.7, 0.75, 0.8, 0.85, 0.9]
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="accuracy", mode="max"
        )
        assert result is False

    def test_max_mode_stops_after_patience(self):
        """Accuracy plateaus for patience epochs, should stop."""
        metrics = [0.7, 0.75, 0.8, 0.8, 0.79]  # 2 consecutive no-improvements
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="accuracy", mode="max"
        )
        assert result is True

    def test_max_mode_stops_at_patience_boundary(self):
        """Accuracy plateaus for exactly patience-1, should not stop."""
        metrics = [0.7, 0.75, 0.8, 0.8]  # Only 1 no-improvement
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="accuracy", mode="max"
        )
        assert result is False


class TestCheckEarlyStopPatience:
    """Test patience parameter behavior."""

    def test_patience_zero_stops_on_first_non_improvement(self):
        """With patience=0, should stop on first non-improvement."""
        metrics = [0.5, 0.4, 0.4]  # Second 0.4 is no improvement
        result = check_early_stop(
            metrics=metrics, patience=0, monitor="loss", mode="min"
        )
        assert result is True

    def test_patience_zero_no_stop_if_always_improving(self):
        """With patience=0, continuous improvement should not stop."""
        metrics = [0.5, 0.4, 0.3]
        result = check_early_stop(
            metrics=metrics, patience=0, monitor="loss", mode="min"
        )
        assert result is False

    def test_large_patience_no_premature_stop(self):
        """Large patience should not stop prematurely."""
        metrics = [0.5, 0.4, 0.4, 0.4, 0.4]  # 3 consecutive no-improvements
        result = check_early_stop(
            metrics=metrics, patience=5, monitor="loss", mode="min"
        )
        assert result is False

    def test_improvement_resets_counter(self):
        """Improvement should reset the consecutive no-improvement counter."""
        # 0.5 → 0.4 (improve), 0.4 → 0.4 (no), 0.4 → 0.4 (no), 0.4 → 0.3 (improve, reset)
        # 0.3 → 0.3 (no), 0.3 → 0.3 (no) = only 2 consecutive at end
        metrics = [0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min"
        )
        assert result is False


class TestCheckEarlyStopMinDelta:
    """Test min_delta threshold for improvement detection."""

    def test_min_delta_zero_tie_is_no_improvement_min_mode(self):
        """Equal value should count as no improvement (strict inequality)."""
        metrics = [0.5, 0.4, 0.4, 0.4, 0.4]  # Tied at 0.4 for 3 epochs
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min", min_delta=0.0
        )
        assert result is True

    def test_min_delta_zero_tie_is_no_improvement_max_mode(self):
        """Equal value should count as no improvement (strict inequality)."""
        metrics = [0.7, 0.8, 0.8, 0.8]  # Tied at 0.8 for 2 epochs
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="acc", mode="max", min_delta=0.0
        )
        assert result is True

    def test_min_delta_positive_requires_larger_improvement_min_mode(self):
        """Small changes below min_delta should not count as improvement."""
        # 0.5 → 0.495 (change=0.005 < 0.01, no improvement)
        metrics = [0.5, 0.495, 0.494, 0.493]
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="loss", mode="min", min_delta=0.01
        )
        assert result is True

    def test_min_delta_positive_allows_sufficient_improvement_min_mode(self):
        """Changes above min_delta should count as improvement."""
        # 0.5 → 0.48 (change=0.02 > 0.01, improvement)
        metrics = [0.5, 0.48, 0.47, 0.46]
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="loss", mode="min", min_delta=0.01
        )
        assert result is False

    def test_min_delta_positive_max_mode(self):
        """min_delta works correctly in max mode."""
        # 0.7 → 0.705 (change=0.005 < 0.01, no improvement)
        # 0.705 → 0.706 (change=0.001 < 0.01, no improvement)
        metrics = [0.7, 0.705, 0.706]
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="acc", mode="max", min_delta=0.01
        )
        assert result is True


class TestCheckEarlyStopEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_metrics_returns_false(self):
        """Empty metrics list should return False."""
        result = check_early_stop(metrics=[], patience=3, monitor="loss", mode="min")
        assert result is False

    def test_single_metric_returns_false(self):
        """Single metric (baseline) should not trigger stop."""
        result = check_early_stop(metrics=[0.5], patience=3, monitor="loss", mode="min")
        assert result is False

    def test_two_metrics_no_stop_if_patience_greater_than_one(self):
        """Two metrics with no improvement, but patience > 1, should not stop."""
        metrics = [0.5, 0.5]  # 1 no-improvement
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="loss", mode="min"
        )
        assert result is False

    def test_metrics_length_equals_patience_plus_one_boundary(self):
        """Exactly enough history to trigger stop at boundary."""
        # 0.5 (baseline), 0.5 (no), 0.5 (no), 0.5 (no) = 3 consecutive no-improvements
        metrics = [0.5, 0.5, 0.5, 0.5]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="loss", mode="min"
        )
        assert result is True


class TestCheckEarlyStopErrorHandling:
    """Test error handling and validation."""

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            check_early_stop(
                metrics=[0.5, 0.4], patience=3, monitor="loss", mode="invalid"
            )

    def test_mode_case_sensitive(self):
        """Mode is case-sensitive, uppercase should fail."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            check_early_stop(metrics=[0.5, 0.4], patience=3, monitor="loss", mode="MIN")


class TestCheckEarlyStopRealWorldScenarios:
    """Test realistic training scenarios."""

    def test_training_loss_decreasing_then_plateau(self):
        """Common scenario: loss improves then plateaus."""
        # Loss decreases, then plateaus at 0.37
        metrics = [0.5, 0.45, 0.4, 0.38, 0.37, 0.37, 0.37, 0.37]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="val_loss", mode="min"
        )
        assert result is True

    def test_accuracy_increasing_with_noise(self):
        """Accuracy increases with some noise, shouldn't stop early."""
        # Accuracy generally increases despite small fluctuations
        metrics = [0.7, 0.75, 0.74, 0.78, 0.82, 0.81, 0.85]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="val_acc", mode="max"
        )
        assert result is False

    def test_validation_loss_degrading(self):
        """Loss starts increasing (overfitting), should stop."""
        # Loss increases for patience epochs
        metrics = [0.3, 0.28, 0.27, 0.27, 0.28, 0.29]
        result = check_early_stop(
            metrics=metrics, patience=3, monitor="val_loss", mode="min"
        )
        assert result is True

    def test_auroc_with_small_improvements(self):
        """AUROC with very small improvements near convergence."""
        # Small improvements < 0.001 should count as no improvement with appropriate min_delta
        metrics = [0.850, 0.851, 0.8515, 0.8518, 0.852]
        result = check_early_stop(
            metrics=metrics, patience=2, monitor="auroc", mode="max", min_delta=0.005
        )
        # Only the first step (0.850 → 0.851 = 0.001 < 0.005) doesn't count as improvement
        # Rest are also small, so multiple non-improvements
        assert result is True


class TestCheckEarlyStopDocstringExamples:
    """Verify examples from the docstring work as documented."""

    def test_docstring_example_1(self):
        """Loss decreasing then plateaus - only 2 consecutive no-improvements."""
        result = check_early_stop(
            [0.5, 0.4, 0.38, 0.37, 0.37, 0.38], patience=3, monitor="loss", mode="min"
        )
        assert result is False

    def test_docstring_example_2(self):
        """Loss plateaus for exactly patience epochs."""
        result = check_early_stop(
            [0.5, 0.4, 0.38, 0.37, 0.37, 0.37, 0.37],
            patience=3,
            monitor="loss",
            mode="min",
        )
        assert result is True

    def test_docstring_example_3(self):
        """Accuracy increasing then plateaus."""
        result = check_early_stop(
            [0.7, 0.75, 0.78, 0.78, 0.77], patience=2, monitor="acc", mode="max"
        )
        assert result is True
