# Multimodal Input Contract

This document describes the data interfaces that the future multimodal embedder
will consume. It establishes the required fields, shapes, and provenance so that
IO helpers can be implemented without re-reading the legacy scripts.

## Overview
- **Scope**: define the canonical inputs for combining sequence and structure
  signals when invoking ESM3 multimodal features.
- **Source of truth**: all reusable artefacts live under `data/embed/` in the
  dedicated `sequences/`, `metadata/`, `consolidated/`, and `structures/`
  directories. No code should reference `archives/legacy_module/...` directly.
- **Consumers**: upcoming loaders in `src/embed/io/` and the
  `MultimodalEmbedder` pipeline.

## Required Inputs

| Field | Description | Canonical Location |
|-------|-------------|--------------------|
| `sequence` | Canonical amino-acid string for the UniProt accession. | `data/embed/sequences/all_proteins.fasta` with metadata in `data/embed/metadata/` |
| `structure_coordinates` | Per-residue backbone coordinates (atoms `N`, `CA`, `C`, `O`) aligned to the sequence. Stored as object arrays in the consolidated NPZ bundle. | `data/embed/consolidated/backbone_coordinates_*.npz` |
| `residue_mask` | Boolean mask aligned to `sequence` marking residues with resolved coordinates. Not yet materialised; generated masks must be persisted under `data/embed/structures/masks/`. | *(to be produced by Milestone B step 2)* |
| `metadata` | Mapping with provenance (`pdb_id`, `chain_id`, file paths, timestamps). | `data/embed/consolidated/esm3_ready_proteins_*.{csv,pkl}` |

### Sequence
- Read via FASTA loader (e.g., `load_sequences`) using IDs from
  `data/embed/metadata/protein_index.json`.
- Validation: enforce `validate_protein_sequence` with `max_length` from config.
- Alignment: `load_multimodal_structure` reconciles the requested sequence with
  the backbone residue count, preferring provided values before falling back to
  metadata fields such as `structure_sequence`. When trimming is required the
  helper records provenance via `aligned_sequence`, `aligned_sequence_source`,
  `aligned_sequence_trimmed`, `aligned_sequence_start_trim`, and
  `aligned_sequence_end_trim`, plus an optional `aligned_sequence_warning` when
  heuristics are applied. Differences versus `original_sequence` are surfaced as
  `aligned_sequence_delta_vs_original` alongside `original_sequence_length`.

### Structure Coordinates
- Load from the compressed NPZ: each key is a UniProt ID, value is a list of
  residue dictionaries with backbone atoms.
- Coordinate ordering: `[N, CA, C, O]`, each 3-vector in Ångström. Missing atoms
  should be stored as `[None, None, None]` and masked out downstream.
- When deserialising, normalise into a `BackboneCoordinateSet` (see planned
  dataclass) with dtype `float32` and shape `(L, 4, 3)` after gap filling.

### Residue Mask
- Boolean vector of length `L == len(sequence)`. `True` where coordinates are
  present for all backbone atoms, `False` otherwise.
- Masks should be versioned and saved under
  `data/embed/structures/masks/{uniprot_id}.npy` alongside the existing
  coordinate files once generated.
- Until masks are generated, callers must derive a provisional mask by checking
  for `None` entries in the coordinate payload.

### Metadata
- Join the sequence and coordinate artefacts with consolidated tables:
  - `data/embed/consolidated/esm3_ready_proteins_*.pkl` contains `uniprot_id`,
    `pdb_id`, `structure_sequence`, and summary statistics.
  - `data/embed/structures/mapping_index_*.json` records the source PDB
    assemblies and chain assignments.
- Required metadata keys for the embedder:
  - `pdb_id`
  - `chain_id`
  - `coordinate_file`
  - `structure_source` (`"pdb"`, `"alphafold"`, etc.)
  - `processing_timestamp`
  - Alignment audit fields emitted by `load_multimodal_structure`

## Assembly Checklist

When preparing a multimodal batch item:
1. **Lookup sequence** from FASTA/metadata.
2. **Load coordinates** from the NPZ bundle; reshape to `(L, 4, 3)`.
3. **Derive or load residue mask** ensuring no mismatch with `L`.
4. **Attach metadata** fields listed above (including alignment provenance).
5. **Validate**:
   - `len(sequence) == coordinates.shape[0] == mask.shape[0]`
   - Coordinate entries are finite where mask is `True`.
   - Metadata includes at least `pdb_id` and `coordinate_file`.

### Helper API

`src.embed.io.structure` provides reusable loaders that enforce this contract:
- `load_backbone_coordinate_set(data_root, uniprot_id)` → `BackboneCoordinateSet`
- `load_residue_mask(data_root, uniprot_id, coordinates)` → `ResidueMask`
- `load_structure_metadata(data_root, uniprot_id)` → metadata mapping
- `load_multimodal_structure(...)` → `MultimodalStructurePayload` bundle
- `materialize_residue_masks(data_root, overwrite=False, uniprot_ids=None)`
  writes persistent masks if they are missing.
- `MultimodalEmbedder` consumes these helpers and falls back to `SequenceEmbedder`
  when coordinates or the multimodal backend are unavailable.
- `scripts/verify_multimodal_backend.py` provides a manual check that runs the
  embedder against a real UniProt entry (requires local ESM3 weights).
- `tests/integration/embed/test_multimodal_smoke.py` exercises the end-to-end
  path with real assets when `esm` is installed; it skips automatically when
  dependencies or weights are absent.

All helpers expect `data_root=data/embed/` (or an equivalent override during
tests) and will infer residue masks from coordinates when dedicated mask files
are absent.

## Future Work
- Step 3 will update `MultimodalEmbedder` to accept a structured payload that
  bundles the fields described here and integrates with the sequence pipeline.

Maintaining this contract ensures we can evolve the multimodal pipeline while
keeping artefact provenance and validation consistent across the codebase.
