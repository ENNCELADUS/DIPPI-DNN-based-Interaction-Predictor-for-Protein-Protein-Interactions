"""
Unit tests for src/utils/logging.py

Tests CSV logging utilities (append_row) for correct header handling,
directory creation, and column ordering.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from src.utils.logging import append_row


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create a temporary directory for CSV files."""
    return tmp_path / "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: append_row()
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendRowBasic:
    """Test basic append_row functionality."""

    def test_create_new_csv_with_header(self, temp_csv_dir):
        """Test creating a new CSV file with header."""
        csv_path = temp_csv_dir / "test.csv"
        row = {"Epoch": 0, "Loss": 0.5, "Accuracy": 0.85}
        columns = ["Epoch", "Loss", "Accuracy"]

        append_row(csv_path, row, columns)

        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_row = next(reader)

        assert header == columns
        assert data_row == ["0", "0.5", "0.85"]

    def test_append_to_existing_csv(self, temp_csv_dir):
        """Test appending to an existing CSV file."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["Epoch", "Loss", "Accuracy"]

        # First row
        append_row(csv_path, {"Epoch": 0, "Loss": 0.5, "Accuracy": 0.85}, columns)
        # Second row
        append_row(csv_path, {"Epoch": 1, "Loss": 0.4, "Accuracy": 0.88}, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # Header + 2 data rows
        assert rows[0] == columns
        assert rows[1] == ["0", "0.5", "0.85"]
        assert rows[2] == ["1", "0.4", "0.88"]

    def test_parent_directory_creation(self, temp_csv_dir):
        """Test that parent directories are created automatically."""
        csv_path = temp_csv_dir / "nested" / "deep" / "test.csv"
        row = {"A": 1, "B": 2}

        append_row(csv_path, row, ["A", "B"])

        assert csv_path.exists()
        assert csv_path.parent.exists()


class TestAppendRowColumnHandling:
    """Test column ordering and fieldnames logic."""

    def test_explicit_columns_parameter(self, temp_csv_dir):
        """Test using explicit columns parameter."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["C", "B", "A"]  # Non-alphabetical order
        row = {"A": 1, "B": 2, "C": 3}

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_row = next(reader)

        assert header == ["C", "B", "A"]
        assert data_row == ["3", "2", "1"]  # Values follow column order

    def test_existing_header_preserved(self, temp_csv_dir):
        """Test that existing header order is preserved when columns=None."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["Z", "Y", "X"]

        # First write with explicit columns
        append_row(csv_path, {"X": 1, "Y": 2, "Z": 3}, columns)

        # Second write without columns parameter (should use existing header)
        append_row(csv_path, {"X": 10, "Y": 20, "Z": 30}, columns=None)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["Z", "Y", "X"]  # Original order preserved
        assert rows[1] == ["3", "2", "1"]
        assert rows[2] == ["30", "20", "10"]  # Follows existing header order

    def test_dict_keys_order_for_new_file(self, temp_csv_dir):
        """Test that dict keys order is used when columns=None and file is new."""
        csv_path = temp_csv_dir / "test.csv"

        # Python 3.7+ preserves dict insertion order
        row = {"Epoch": 0, "Loss": 0.5, "Accuracy": 0.85}
        append_row(csv_path, row, columns=None)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        # Should match dict insertion order
        assert header == ["Epoch", "Loss", "Accuracy"]

    def test_missing_keys_become_empty_string(self, temp_csv_dir):
        """Test that missing keys in row_dict become empty strings."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["A", "B", "C", "D"]
        row = {"A": 1, "C": 3}  # Missing B and D

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)

        assert data_row == ["1", "", "3", ""]

    def test_extra_keys_ignored(self, temp_csv_dir):
        """Test that extra keys in row_dict are ignored."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["A", "B"]
        row = {"A": 1, "B": 2, "C": 3, "D": 4}  # C and D are extra

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_row = next(reader)

        assert header == ["A", "B"]
        assert data_row == ["1", "2"]


class TestAppendRowDataTypes:
    """Test handling of different data types."""

    def test_numeric_types(self, temp_csv_dir):
        """Test that numeric types are converted to strings correctly."""
        csv_path = temp_csv_dir / "test.csv"
        row = {
            "int_val": 42,
            "float_val": 3.14159,
            "scientific": 1.5e-4,
            "zero": 0,
        }
        columns = list(row.keys())

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)

        assert data_row[0] == "42"
        assert data_row[1] == "3.14159"
        assert "1.5" in data_row[2] or "0.00015" in data_row[2]  # Scientific notation
        assert data_row[3] == "0"

    def test_string_values(self, temp_csv_dir):
        """Test that string values are written correctly."""
        csv_path = temp_csv_dir / "test.csv"
        row = {
            "name": "test_run",
            "status": "completed",
            "empty": "",
        }
        columns = list(row.keys())

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)

        assert data_row == ["test_run", "completed", ""]

    def test_mixed_types(self, temp_csv_dir):
        """Test mixed data types in one row."""
        csv_path = temp_csv_dir / "test.csv"
        row = {
            "epoch": 5,
            "loss": 0.342,
            "metric": "auroc",
            "value": 0.89,
        }
        columns = list(row.keys())

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)

        assert data_row == ["5", "0.342", "auroc", "0.89"]


class TestAppendRowEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_file_needs_header(self, temp_csv_dir):
        """Test that an empty file triggers header writing."""
        csv_path = temp_csv_dir / "test.csv"

        # Create empty file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.touch()
        assert csv_path.stat().st_size == 0

        row = {"A": 1}
        append_row(csv_path, row, ["A"])

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2  # Header + data
        assert rows[0] == ["A"]
        assert rows[1] == ["1"]

    def test_pathlib_path_input(self, temp_csv_dir):
        """Test that Path objects are handled correctly."""
        csv_path = Path(temp_csv_dir) / "test.csv"
        row = {"A": 1}

        append_row(csv_path, row, ["A"])  # Pass Path object

        assert csv_path.exists()

    def test_string_path_input(self, temp_csv_dir):
        """Test that string paths are handled correctly."""
        csv_path = str(temp_csv_dir / "test.csv")
        row = {"A": 1}

        append_row(csv_path, row, ["A"])  # Pass string

        assert Path(csv_path).exists()

    def test_empty_row_dict(self, temp_csv_dir):
        """Test behavior with empty row_dict."""
        csv_path = temp_csv_dir / "test.csv"
        columns = ["A", "B", "C"]
        row = {}  # Empty

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)

        # All values should be empty strings
        assert data_row == ["", "", ""]

    def test_unicode_content(self, temp_csv_dir):
        """Test that Unicode content is handled correctly."""
        csv_path = temp_csv_dir / "test.csv"
        row = {
            "name": "测试",
            "symbol": "α",
            "emoji": "🎉",
        }
        columns = list(row.keys())

        append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)

        assert data_row == ["测试", "α", "🎉"]


class TestAppendRowTrainingMetricsScenario:
    """Test realistic training metrics logging scenario."""

    def test_training_step_csv_schema(self, temp_csv_dir):
        """Test the exact schema used in run.py for training_step.csv."""
        csv_path = temp_csv_dir / "training_step.csv"

        # Schema from logging_overview.md line 23
        primary_metric = "auroc"
        secondary_metric = "recall"
        columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            f"Train {primary_metric}",
            f"Train {secondary_metric}",
            f"Val {primary_metric}",
            f"Val {secondary_metric}",
            "Learning Rate",
        ]

        # Simulate first epoch
        row1 = {
            "Epoch": 0,
            "Epoch Time": 125.3,
            "Train Loss": 0.6543,
            "Val Loss": 0.6201,
            f"Train {primary_metric}": 0.7234,
            f"Train {secondary_metric}": 0.6891,
            f"Val {primary_metric}": 0.7456,
            f"Val {secondary_metric}": 0.7012,
            "Learning Rate": 0.0003,
        }
        append_row(csv_path, row1, columns)

        # Simulate second epoch
        row2 = {
            "Epoch": 1,
            "Epoch Time": 123.1,
            "Train Loss": 0.5987,
            "Val Loss": 0.5834,
            f"Train {primary_metric}": 0.7689,
            f"Train {secondary_metric}": 0.7234,
            f"Val {primary_metric}": 0.7812,
            f"Val {secondary_metric}": 0.7456,
            "Learning Rate": 0.0002,
        }
        append_row(csv_path, row2, columns)

        # Verify the CSV
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        assert rows[0] == columns

        # Check data rows count
        assert len(rows) == 3  # Header + 2 epochs

        # Check first data row
        assert rows[1][0] == "0"  # Epoch
        assert "125.3" in rows[1][1]  # Epoch Time
        assert "0.6543" in rows[1][2]  # Train Loss

        # Check second data row
        assert rows[2][0] == "1"  # Epoch
        assert "123.1" in rows[2][1]  # Epoch Time

    def test_multiple_runs_append_correctly(self, temp_csv_dir):
        """Test that multiple training runs append correctly to the same CSV."""
        csv_path = temp_csv_dir / "training_step.csv"
        columns = ["Epoch", "Loss", "Accuracy"]

        # Simulate 5 epochs
        for epoch in range(5):
            row = {
                "Epoch": epoch,
                "Loss": 0.5 - epoch * 0.05,
                "Accuracy": 0.7 + epoch * 0.03,
            }
            append_row(csv_path, row, columns)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 6  # Header + 5 epochs
        assert rows[0] == columns
        assert rows[1][0] == "0"
        assert rows[5][0] == "4"
