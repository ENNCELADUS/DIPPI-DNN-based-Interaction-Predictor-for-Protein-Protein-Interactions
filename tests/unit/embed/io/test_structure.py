"""Tests for src.embed.io.structure."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.embed.io.structure import (
    StructureDataNotFoundError,
    load_backbone_coordinate_set,
    load_multimodal_structure,
    load_residue_mask,
    load_structure_metadata,
    materialize_residue_masks,
)


@pytest.fixture()
def multimodal_data_root(tmp_path: Path) -> Path:
    root = tmp_path
    consolidated = root / "consolidated"
    consolidated.mkdir(parents=True)
    structures = root / "structures" / "masks"
    structures.mkdir(parents=True)

    # Backbone coordinates NPZ
    npz_path = consolidated / "backbone_coordinates_20240101_000000.npz"
    residues = np.empty(2, dtype=object)
    residues[0] = {
        "N": [0.0, 0.0, 0.0],
        "CA": [1.0, 1.0, 1.0],
        "C": [2.0, 2.0, 2.0],
        "O": [3.0, 3.0, 3.0],
    }
    residues[1] = {
        "N": [0.5, 0.5, 0.5],
        "CA": [1.5, 1.5, 1.5],
        "C": [2.5, 2.5, 2.5],
        "O": [np.nan, np.nan, np.nan],
    }
    np.savez_compressed(npz_path, P12345=residues)

    # Metadata CSV without heavy payloads
    csv_path = consolidated / "esm3_ready_proteins_20240101.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "uniprot_id",
                "pdb_id",
                "original_sequence",
                "structure_sequence",
                "coordinate_file",
                "chain_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uniprot_id": "P12345",
                "pdb_id": "1abc",
                "original_sequence": "AAA",
                "structure_sequence": "AA",
                "coordinate_file": "coordinates/p12345.cif.gz",
                "chain_id": "A",
            }
        )

    # Mask file (length matches residues)
    mask_path = root / "structures" / "masks" / "P12345.npy"
    np.save(mask_path, np.array([True, False]))

    return root


def test_load_backbone_coordinate_set(multimodal_data_root: Path) -> None:
    coordinate_set = load_backbone_coordinate_set(multimodal_data_root, "P12345")

    assert coordinate_set.values.shape == (2, 4, 3)
    assert coordinate_set.residue_count == 2
    np.testing.assert_array_equal(coordinate_set.values[0, 1], [1.0, 1.0, 1.0])


def test_load_structure_metadata(multimodal_data_root: Path) -> None:
    metadata = load_structure_metadata(multimodal_data_root, "P12345")

    assert metadata["pdb_id"] == "1abc"
    assert metadata["chain_id"] == "A"
    assert metadata["structure_sequence"] == "AA"


def test_load_residue_mask_prefers_saved_file(multimodal_data_root: Path) -> None:
    coordinate_set = load_backbone_coordinate_set(multimodal_data_root, "P12345")
    mask = load_residue_mask(multimodal_data_root, "P12345", coordinate_set)

    assert mask.values.tolist() == [True, False]
    assert mask.source.endswith("P12345.npy")


def test_load_residue_mask_infers_when_missing(multimodal_data_root: Path) -> None:
    mask_path = multimodal_data_root / "structures" / "masks" / "P12345.npy"
    mask_path.unlink()

    coordinate_set = load_backbone_coordinate_set(multimodal_data_root, "P12345")
    mask = load_residue_mask(multimodal_data_root, "P12345", coordinate_set)

    assert mask.values.tolist() == [True, False]
    assert mask.source == "inferred_from_coordinates"


def test_load_residue_mask_requires_file_when_disabled(
    multimodal_data_root: Path,
) -> None:
    mask_path = multimodal_data_root / "structures" / "masks" / "P12345.npy"
    mask_path.unlink()

    coordinate_set = load_backbone_coordinate_set(multimodal_data_root, "P12345")

    with pytest.raises(StructureDataNotFoundError):
        load_residue_mask(
            multimodal_data_root,
            "P12345",
            coordinate_set,
            allow_infer=False,
        )


def test_load_multimodal_structure(multimodal_data_root: Path) -> None:
    payload = load_multimodal_structure(multimodal_data_root, "P12345")

    assert payload.sequence == "AA"
    assert payload.metadata["coordinate_file"] == "coordinates/p12345.cif.gz"
    assert payload.coordinates.values.shape == (2, 4, 3)
    assert payload.mask.length == 2
    assert payload.metadata["aligned_sequence_source"] == "metadata.structure_sequence"
    assert payload.metadata["aligned_sequence_trimmed"] is False
    assert payload.metadata["aligned_sequence_delta_vs_original"] == -1


def test_load_multimodal_structure_prefers_provided_sequence(
    multimodal_data_root: Path,
) -> None:
    payload = load_multimodal_structure(multimodal_data_root, "P12345", sequence="AA")

    assert payload.sequence == "AA"
    assert payload.metadata["aligned_sequence_source"] == "provided_sequence"
    assert payload.metadata["aligned_sequence_trimmed"] is False


def test_load_multimodal_structure_trims_with_offsets(tmp_path: Path) -> None:
    root = tmp_path
    consolidated = root / "consolidated"
    consolidated.mkdir(parents=True)
    structures = root / "structures" / "masks"
    structures.mkdir(parents=True)

    npz_path = consolidated / "backbone_coordinates_20240202_000000.npz"
    residues = np.empty(2, dtype=object)
    residues[0] = {
        "N": [0.0, 0.0, 0.0],
        "CA": [1.0, 1.0, 1.0],
        "C": [2.0, 2.0, 2.0],
        "O": [3.0, 3.0, 3.0],
    }
    residues[1] = {
        "N": [0.5, 0.5, 0.5],
        "CA": [1.5, 1.5, 1.5],
        "C": [2.5, 2.5, 2.5],
        "O": [np.nan, np.nan, np.nan],
    }
    np.savez_compressed(npz_path, Q99999=residues)

    csv_path = consolidated / "esm3_ready_proteins_20240202.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "uniprot_id",
                "pdb_id",
                "original_sequence",
                "coordinate_file",
                "chain_id",
                "residue_start",
                "residue_end",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uniprot_id": "Q99999",
                "pdb_id": "2xyz",
                "original_sequence": "MABC",
                "coordinate_file": "coordinates/q99999.cif.gz",
                "chain_id": "B",
                "residue_start": "2",
                "residue_end": "3",
            }
        )

    mask_path = root / "structures" / "masks" / "Q99999.npy"
    np.save(mask_path, np.array([True, True]))

    payload = load_multimodal_structure(root, "Q99999", sequence="MABC")

    assert payload.sequence == "AB"
    assert payload.metadata["aligned_sequence_source"] == "provided_sequence"
    assert payload.metadata["aligned_sequence_trimmed"] is True
    assert payload.metadata["aligned_sequence_start_trim"] == 1
    assert payload.metadata["aligned_sequence_end_trim"] == 1
    assert "aligned_sequence_warning" not in payload.metadata


def test_load_multimodal_structure_trims_naively_when_no_hints(
    tmp_path: Path,
) -> None:
    root = tmp_path
    consolidated = root / "consolidated"
    consolidated.mkdir(parents=True)
    structures = root / "structures" / "masks"
    structures.mkdir(parents=True)

    npz_path = consolidated / "backbone_coordinates_20240203_000000.npz"
    residues = np.empty(2, dtype=object)
    residues[0] = {
        "N": [0.0, 0.0, 0.0],
        "CA": [1.0, 1.0, 1.0],
        "C": [2.0, 2.0, 2.0],
        "O": [3.0, 3.0, 3.0],
    }
    residues[1] = {
        "N": [0.5, 0.5, 0.5],
        "CA": [1.5, 1.5, 1.5],
        "C": [2.5, 2.5, 2.5],
        "O": [np.nan, np.nan, np.nan],
    }
    np.savez_compressed(npz_path, Q11111=residues)

    csv_path = consolidated / "esm3_ready_proteins_20240203.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "uniprot_id",
                "pdb_id",
                "original_sequence",
                "coordinate_file",
                "chain_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uniprot_id": "Q11111",
                "pdb_id": "3qwe",
                "original_sequence": "AAC",
                "coordinate_file": "coordinates/q11111.cif.gz",
                "chain_id": "C",
            }
        )

    mask_path = root / "structures" / "masks" / "Q11111.npy"
    np.save(mask_path, np.array([True, True]))

    payload = load_multimodal_structure(root, "Q11111", sequence="AAC")

    assert payload.sequence == "AA"
    assert payload.metadata["aligned_sequence_source"] == "provided_sequence"
    assert payload.metadata["aligned_sequence_trimmed"] is True
    assert (
        payload.metadata["aligned_sequence_warning"]
        == "trimmed_tail_without_offset_metadata"
    )


def test_materialize_residue_masks(multimodal_data_root: Path) -> None:
    mask_path = multimodal_data_root / "structures" / "masks" / "P12345.npy"
    mask_path.unlink()  # ensure helper must create it

    written = materialize_residue_masks(multimodal_data_root)

    assert "P12345" in written
    assert written["P12345"].exists()
    mask_values = np.load(written["P12345"])
    assert mask_values.dtype == np.bool_
    assert mask_values.tolist() == [True, False]

    # Re-run without overwrite to ensure existing file is reused
    mask_values[:] = True
    np.save(written["P12345"], mask_values)
    materialize_residue_masks(multimodal_data_root, overwrite=False)
    persisted = np.load(written["P12345"])
    assert persisted.tolist() == [True, True]

    # Overwrite restores inferred values
    materialize_residue_masks(multimodal_data_root, overwrite=True)
    refreshed = np.load(written["P12345"])
    assert refreshed.tolist() == [True, False]


# ===== Integration Tests with Real Consolidated Data =====


REAL_DATA_ROOT = Path("data/embed")
REAL_CONSOLIDATED_DIR = REAL_DATA_ROOT / "consolidated"


@pytest.fixture(scope="module")
def real_consolidated_data() -> tuple[Path, Path]:
    """Locate real consolidated data files.

    Returns paths to the latest NPZ and PKL consolidated files.
    Fails if data is missing (data must exist for validation).
    """
    npz_candidates = sorted(REAL_CONSOLIDATED_DIR.glob("backbone_coordinates_*.npz"))
    pkl_candidates = sorted(REAL_CONSOLIDATED_DIR.glob("esm3_ready_proteins_*.pkl"))

    if not npz_candidates:
        pytest.fail(
            f"No backbone coordinate NPZ files found in {REAL_CONSOLIDATED_DIR}. "
            "Data must exist for alignment validation."
        )
    if not pkl_candidates:
        pytest.fail(
            f"No protein metadata PKL files found in {REAL_CONSOLIDATED_DIR}. "
            "Data must exist for alignment validation."
        )

    # Use latest files by modification time
    npz_path = max(npz_candidates, key=lambda p: p.stat().st_mtime)
    pkl_path = max(pkl_candidates, key=lambda p: p.stat().st_mtime)

    return npz_path, pkl_path


@pytest.mark.integration
@pytest.mark.slow
def test_alignment_with_real_consolidated_data(
    real_consolidated_data: tuple[Path, Path],
) -> None:
    """Validate alignment reconciliation against real production data.

    Tests diverse protein samples with varying sequence/coordinate relationships:
    - Exact matches
    - Structure sequence shorter than original
    - Structure sequence longer than original
    """
    npz_path, pkl_path = real_consolidated_data

    # Representative samples covering diverse alignment scenarios
    test_samples = [
        ("P20472", "all_match"),
        ("A0FGR8", "struct_slightly_shorter"),
        ("A0AVK6", "struct_shorter"),
        ("A4UGR9", "struct_much_shorter"),
        ("A0PK00", "struct_slightly_longer"),
        ("B1AL88", "struct_longer"),
        ("A0A087X2I1", "struct_much_longer"),
    ]

    for uniprot_id, category in test_samples:
        payload = load_multimodal_structure(REAL_DATA_ROOT, uniprot_id)

        # Core alignment invariants
        assert (
            len(payload.sequence)
            == payload.coordinates.residue_count
            == payload.mask.length
        ), (
            f"{uniprot_id} ({category}): Length mismatch - "
            f"sequence={len(payload.sequence)}, "
            f"coords={payload.coordinates.residue_count}, "
            f"mask={payload.mask.length}"
        )

        # Metadata completeness
        assert "aligned_sequence_source" in payload.metadata, (
            f"{uniprot_id}: Missing aligned_sequence_source"
        )
        assert "aligned_sequence_length" in payload.metadata, (
            f"{uniprot_id}: Missing aligned_sequence_length"
        )
        assert "aligned_sequence_trimmed" in payload.metadata, (
            f"{uniprot_id}: Missing aligned_sequence_trimmed"
        )
        assert payload.metadata["aligned_sequence_length"] == len(payload.sequence), (
            f"{uniprot_id}: aligned_sequence_length mismatch"
        )

        # Coordinate validity where mask is True
        finite_count = np.isfinite(payload.coordinates.values).all(axis=(1, 2)).sum()
        true_mask_count = payload.mask.values.sum()
        # Allow some NaN in masked regions, but most True mask positions should have finite coords
        assert finite_count >= true_mask_count * 0.5, (
            f"{uniprot_id}: Too many NaN coordinates in masked regions - "
            f"finite={finite_count}, true_mask={true_mask_count}"
        )

        # Required metadata fields
        assert "pdb_id" in payload.metadata, f"{uniprot_id}: Missing pdb_id"
        assert "coordinate_file" in payload.metadata or payload.metadata.get(
            "pdb_id"
        ), f"{uniprot_id}: Missing coordinate provenance"


@pytest.mark.integration
@pytest.mark.slow
def test_alignment_regression_snapshots(
    real_consolidated_data: tuple[Path, Path],
) -> None:
    """Regression test: validate exact alignment outcomes for known proteins.

    Hardcoded expected values ensure alignment logic changes are caught.
    If this test fails after data updates, verify alignment is still correct
    before updating expected values.
    """
    # Expected alignment outcomes for specific proteins
    expected = {
        "P20472": {
            "aligned_sequence_length": 110,
            "aligned_sequence_source": "metadata.structure_sequence",
            "aligned_sequence_trimmed": False,
            "aligned_sequence_start_trim": 0,
            "aligned_sequence_end_trim": 0,
            "aligned_sequence_delta_vs_original": 0,
            "original_sequence_length": 110,
        },
        "A0AVK6": {
            "aligned_sequence_length": 216,
            "aligned_sequence_source": "metadata.structure_sequence",
            "aligned_sequence_trimmed": False,
            "aligned_sequence_start_trim": 0,
            "aligned_sequence_end_trim": 0,
            "aligned_sequence_delta_vs_original": -651,
            "original_sequence_length": 867,
        },
        "A0A087X2I1": {
            "aligned_sequence_length": 13430,
            "aligned_sequence_source": "metadata.structure_sequence",
            "aligned_sequence_trimmed": False,
            "aligned_sequence_start_trim": 0,
            "aligned_sequence_end_trim": 0,
            "aligned_sequence_delta_vs_original": 13027,
            "original_sequence_length": 403,
        },
    }

    for uniprot_id, expected_meta in expected.items():
        payload = load_multimodal_structure(REAL_DATA_ROOT, uniprot_id)

        for key, expected_value in expected_meta.items():
            actual_value = payload.metadata.get(key)
            assert actual_value == expected_value, (
                f"{uniprot_id}: {key} mismatch - "
                f"expected {expected_value}, got {actual_value}"
            )

        # Additional invariant: sequence length matches aligned_sequence_length
        assert len(payload.sequence) == expected_meta["aligned_sequence_length"], (
            f"{uniprot_id}: Sequence length doesn't match expected aligned_sequence_length"
        )


@pytest.mark.integration
def test_alignment_metadata_completeness(
    real_consolidated_data: tuple[Path, Path],
) -> None:
    """Validate that alignment metadata has correct types and constraints."""
    # Use a simple representative sample
    payload = load_multimodal_structure(REAL_DATA_ROOT, "P20472")

    # Type checks
    assert isinstance(payload.metadata["aligned_sequence_length"], int)
    assert isinstance(payload.metadata["aligned_sequence_source"], str)
    assert isinstance(payload.metadata["aligned_sequence_trimmed"], bool)
    assert isinstance(payload.metadata["aligned_sequence_start_trim"], int)
    assert isinstance(payload.metadata["aligned_sequence_end_trim"], int)

    # Constraint checks
    assert payload.metadata["aligned_sequence_length"] > 0
    assert payload.metadata["aligned_sequence_start_trim"] >= 0
    assert payload.metadata["aligned_sequence_end_trim"] >= 0

    # Source must be a known value
    valid_sources = {
        "provided_sequence",
        "metadata.structure_sequence",
        "metadata.aligned_sequence",
        "metadata.sequence",
        "metadata.original_sequence",
    }
    assert payload.metadata["aligned_sequence_source"] in valid_sources, (
        f"Unknown aligned_sequence_source: {payload.metadata['aligned_sequence_source']}"
    )

    # If trimmed, at least one trim value should be > 0
    if payload.metadata["aligned_sequence_trimmed"]:
        assert (
            payload.metadata["aligned_sequence_start_trim"] > 0
            or payload.metadata["aligned_sequence_end_trim"] > 0
        ), "Trimmed=True but no trim values > 0"

    # If original_sequence exists, delta should be computable
    if "original_sequence_length" in payload.metadata:
        assert "aligned_sequence_delta_vs_original" in payload.metadata
        expected_delta = (
            payload.metadata["aligned_sequence_length"]
            - payload.metadata["original_sequence_length"]
        )
        assert (
            payload.metadata["aligned_sequence_delta_vs_original"] == expected_delta
        ), "aligned_sequence_delta_vs_original calculation error"
