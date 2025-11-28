"""
CSV logging utilities for DIPPI pipeline.

Provides minimal helpers for appending metric rows to CSV files.
No side effects: no logging setup, no prints, no global state.
"""

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union


def append_row(
    csv_path: Union[str, Path],
    row_dict: Mapping[str, object],
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    Append one row to a CSV at csv_path.

    - Creates parent directories if needed.
    - Writes header if file does not exist or is empty.
    - Column order:
        1) If columns is given, use it (missing keys become "", extra keys ignored).
        2) Else if file has existing header, reuse that order.
        3) Else use row_dict keys in insertion order (Python 3.7+).

    Args:
        csv_path: Path to CSV file (created if needed)
        row_dict: Row data as {column_name: value}
        columns: Optional explicit column order; if None, inferred from file or dict

    Raises:
        OSError: If file cannot be read/written (permissions, disk full, etc.)
    """
    path = Path(csv_path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect if we need to write header
    needs_header = (not path.exists()) or (path.stat().st_size == 0)

    # Determine fieldnames (priority: explicit > existing > dict keys)
    if columns is not None:
        fieldnames = list(columns)
    elif path.exists() and not needs_header:
        # File exists with content: read existing header
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)  # First line is header
    else:
        # New file or empty: use dict keys
        fieldnames = list(row_dict.keys())

    # Append row to CSV
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if needed
        if needs_header:
            writer.writeheader()

        # Normalize row: missing keys become ""
        normalized = {name: row_dict.get(name, "") for name in fieldnames}
        writer.writerow(normalized)
