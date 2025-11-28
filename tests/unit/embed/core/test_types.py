"""Tests for src.embed.core.types."""

from __future__ import annotations

import numpy as np
import pytest

from src.embed.core.types import (
    BackboneCoordinateSet,
    EmbeddingConfig,
    MultimodalStructurePayload,
    ResidueMask,
)


def test_embedding_config_with_updates_preserves_original():
    cfg = EmbeddingConfig(model_name="esm3_small")
    updated = cfg.with_updates(batch_size=16)

    assert cfg.batch_size == 1  # default remains unchanged
    assert updated.batch_size == 16
    assert updated.model_name == "esm3_small"


def test_embedding_config_remote_token_requirement():
    remote = EmbeddingConfig(use_local_model=False, forge_api_token=None)
    assert remote.is_remote is True
    assert remote.requires_remote_token() is True

    supplied = remote.with_updates(forge_api_token="secret")
    assert supplied.requires_remote_token() is False


def test_embedding_config_resolved_device_is_known_value():
    cfg = EmbeddingConfig(device="cpu")
    assert cfg.resolved_device() == "cpu"

    auto_cfg = EmbeddingConfig(device="auto")
    assert auto_cfg.resolved_device() in {"cpu", "cuda", "mps"}


def test_backbone_coordinate_set_from_legacy_records_round_trips():
    residues = [
        {
            "N": [0.0, 1.0, 2.0],
            "CA": [3.0, 4.0, 5.0],
            "C": [6.0, 7.0, 8.0],
            "O": [9.0, 9.0, 9.0],
        },
        {
            "N": [1.0, 1.0, 1.0],
            "CA": [2.0, 2.0, 2.0],
            "C": [3.0, 3.0, 3.0],
            "O": [4.0, 4.0, 4.0],
        },
    ]

    coordinate_set = BackboneCoordinateSet.from_legacy_records(residues)

    assert coordinate_set.residue_count == 2
    assert coordinate_set.values.shape == (2, 4, 3)
    np.testing.assert_array_equal(coordinate_set.values[0, 0], [0.0, 1.0, 2.0])


def test_residue_mask_inferred_from_coordinates():
    coords = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]],
            [
                [np.nan, np.nan, np.nan],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )
    coordinate_set = BackboneCoordinateSet(coords)
    mask = ResidueMask.from_coordinates(coordinate_set)

    assert mask.length == 2
    assert mask.values.dtype == np.bool_
    assert mask.values.tolist() == [True, False]


def test_multimodal_structure_payload_validates_lengths():
    coords = BackboneCoordinateSet(np.zeros((2, 4, 3), dtype=np.float32))
    mask = ResidueMask(np.array([True, True]))

    payload = MultimodalStructurePayload(
        uniprot_id="P12345",
        sequence="AA",
        coordinates=coords,
        mask=mask,
        metadata={"pdb_id": "1abc"},
    )

    assert payload.has_complete_structure is True

    with pytest.raises(ValueError):
        MultimodalStructurePayload(
            uniprot_id="P12345",
            sequence="AAA",
            coordinates=coords,
            mask=mask,
            metadata={},
        )
