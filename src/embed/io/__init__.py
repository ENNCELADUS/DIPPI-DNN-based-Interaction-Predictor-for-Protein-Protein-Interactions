"""Filesystem adapters for the embed package."""

from .filesystem import load_sequences, save_results
from .structure import (
    StructureDataNotFoundError,
    load_backbone_coordinate_set,
    load_multimodal_structure,
    load_residue_mask,
    load_structure_metadata,
    materialize_residue_masks,
)

__all__ = [
    "load_sequences",
    "save_results",
    "StructureDataNotFoundError",
    "load_backbone_coordinate_set",
    "load_residue_mask",
    "load_structure_metadata",
    "load_multimodal_structure",
    "materialize_residue_masks",
]
