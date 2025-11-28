# Embed Module

This directory hosts the live documentation for the simplified embed package.

## Layout Overview
- `core/` – shared dataclasses, base classes, validation helpers
- `io/` – filesystem adapters and converters
- `pipelines/` – user-facing embedding strategies
- `cli/` – thin wrappers that expose pipelines as scripts
- `config.py` – centralized configuration primitives

See `../archives/README.md` for the preserved legacy workflow notes.

## Quickstart

### Sequence-Only Mode

**Input Data:**
- **Primary**: `data/embed/sequences/all_proteins.fasta` (9,418 proteins, normalized)
  - Pre-processed FASTA with consistent UniProt IDs
  - Header format: `>UniProt_ID|variant|length=N|source=dataset`
- **Alternative**: CSV inputs (e.g., `data/TMP_protein/processed/unique_proteins.csv`)
  - Provide `--input-format csv` and specify the identifier/sequence columns
    via `--csv-id-column` (default `uniprotID`) and `--csv-sequence-column`
    (default `sequence`).

**Example Usage:**

```bash
# Using the normalized FASTA (recommended)
python -m src.embed.cli.main \
  data/embed/sequences/all_proteins.fasta \
  outputs/embeddings.npz \
  --input-format fasta

# Using CSV with UniProt ID + sequence columns
python -m src.embed.cli.main \
  data/TMP_protein/processed/unique_proteins.csv \
  outputs/embeddings.npz \
  --input-format csv \
  --csv-id-column uniprotID \
  --csv-sequence-column sequence

# HPC-friendly entrypoint (mirrors the CLI)
python src/embed/run.py \
  data/TMP_protein/processed/unique_proteins.csv \
  outputs/embeddings.npz \
  --input-format csv
```

The CLI automatically instantiates `SequenceEmbedder`, which runs entirely against
locally cached ESM3 checkpoints. Remote Forge execution is blocked until a later
milestone, so ensure `EMBED_USE_LOCAL_MODEL=true` (default) and the checkpoint is
present under `EMBED_MODEL_CACHE`.

**Output Format:**
- Compressed `.npz` archives containing object arrays for `ids`, `embeddings`, metadata, and original/cleaned sequences
- Load via `numpy.load(..., allow_pickle=True)` or `src.utils.data_io.load_embeddings` for Torch-friendly mapping
- Each embedding is shape `(L, D)` where `L` is sequence length and `D` is embedding dimension (typically 1536 for ESM3)

### Multimodal Mode (Sequence + Structure)

For proteins with available 3D structures, use `MultimodalEmbedder` to leverage both
sequence and structure information.

**Input Data:**
- **Backbone coordinates**: `data/embed/consolidated/backbone_coordinates_20250830_024050.npz` (4,573 proteins)
  - Per-residue `[N, CA, C, O]` atom coordinates in Ångström units
  - Stored as object arrays keyed by UniProt ID
- **Metadata**: `data/embed/consolidated/esm3_ready_proteins_20250830_024050.{pkl,csv}`
  - Contains `uniprot_id`, `pdb_id`, `structure_sequence`, `original_sequence`
- **Residue masks**: `data/embed/structures/masks/{uniprot_id}.npy` (auto-generated)
  - Boolean arrays indicating which residues have complete backbone coordinates
  - Auto-inferred from coordinates if files are missing
- **Sequences**: Provided directly or loaded from metadata (prefers `structure_sequence`)

**Example Usage:**

```python
from pathlib import Path
from src.embed.pipelines.multimodal import MultimodalEmbedder
from src.embed.core.types import EmbeddingConfig

config = EmbeddingConfig(data_root=Path("data/embed"))
embedder = MultimodalEmbedder(config)

# For a protein with known structure (e.g., P20472 has 110 residues)
result = embedder.embed_one(uniprot_id="P20472", sequence="MLSDEDFK...")
```

**Pipeline Workflow:**

1. **Input Assembly** (via `load_multimodal_structure`):
   - Loads backbone coordinates from consolidated NPZ
   - Loads metadata from consolidated PKL/CSV
   - Loads or infers residue mask
   - Validates all inputs are consistent

2. **Sequence-Structure Alignment**:
   - Reconciles input sequence with backbone coordinate count
   - Prefers provided sequence → `structure_sequence` → `original_sequence`
   - Applies trimming with metadata hints (`residue_start`, `residue_end`) when needed
   - Records alignment provenance: `aligned_sequence_source`, `aligned_sequence_trimmed`,
     `aligned_sequence_start_trim`, `aligned_sequence_end_trim`
   - Validates invariant: `len(sequence) == len(coordinates) == len(mask)`

3. **ESM3 Multimodal Encoding**:
   - Builds `ESMProtein` object with aligned sequence + Atom37 coordinates
   - Runs two embedding tracks:
     - **sequence+structure**: Combined signals (default output)
     - **structure-only**: Pure structural embeddings
   - Each track produces per-residue embeddings and optional mean embeddings

4. **Fallback Handling**:
   - Falls back to `SequenceEmbedder` if:
     - Structure data not found
     - ESM3 multimodal backend unavailable
     - Alignment reconciliation fails
   - Fallback reason recorded in `metadata["fallback_reason"]`

**Expected Outputs:**

```python
result = embedder.embed_one(uniprot_id, sequence)
# result.embedding: np.ndarray, shape (L, D) - per-residue embeddings
# result.metadata: dict with:
#   - pipeline: "multimodal" or "sequence" (if fallback)
#   - sequence_length: int
#   - residue_count: int
#   - structure_coverage: float (fraction of residues with coordinates)
#   - aligned_sequence_source: str (provenance)
#   - aligned_sequence_trimmed: bool
#   - aligned_sequence_start_trim: int
#   - aligned_sequence_end_trim: int
#   - aligned_sequence_delta_vs_original: int
#   - original_sequence_length: int
#   - pdb_id: str
#   - multimodal_tracks: dict with shape info for both embedding tracks
#   - mean_embedding: list (optional, protein-level representation)
```

**Common Failure Modes & Solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| `StructureDataNotFoundError` | UniProt ID not in consolidated data | Check `data/embed/consolidated/esm3_ready_proteins_*.pkl` for coverage. Falls back to sequence-only. |
| `MultimodalBackendError: backend did not produce embeddings` | ESM3 model issue or incompatible inputs | Check ESM3 installation: `pip install fair-esm`. Verify structure data integrity. |
| `LocalModelBackendError` | Model weights not cached | Set `EMBED_MODEL_CACHE` and download weights. Or use `EMBED_USE_LOCAL_MODEL=false` for API. |
| `ValueError: Sequence length exceeds max_length` | Protein too long | Increase `EMBED_MAX_SEQUENCE_LENGTH`, enable `EMBED_TRUNCATE_LONG_SEQUENCES=true` / `--truncate-long-sequences`, or run once normally and re-embed failures via `--retry-truncate-errors --truncate-retry-length <L>`. Default max: 1024 residues. |
| `Sequence-structure length mismatch` | Alignment reconciliation failed | Check metadata `residue_start`/`residue_end` fields. Alignment trimming should handle automatically. |
| `CUDA out of memory` | Batch size too large or sequence too long | Reduce `EMBED_BATCH_SIZE` or use CPU. ESM3 memory scales O(L²) with sequence length. |
| Slow performance | Running on CPU | Use GPU with `EMBED_DEVICE=cuda`. Install PyTorch with CUDA support. |
| `fallback_reason` in metadata | Multimodal failed, using sequence-only | Check logs for details. Common: missing structure files, model loading issues. |

**Logging:**

Enable detailed logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail

# Example output:
# INFO:src.embed.pipelines.multimodal:Loading multimodal model: esm3_sm_open_v1 on device cuda
# INFO:src.embed.pipelines.multimodal:Multimodal backend ready
# INFO:src.embed.pipelines.multimodal:P20472: Structure loaded - residues=110, coverage=100.00%, source=1rjv
# INFO:src.embed.pipelines.multimodal:P20472: Multimodal embedding complete - shape=(1, 110, 1536)
```

## Configuration
- Defaults are derived from environment variables via `src.embed.config.default_config()`.
- Key variables:
  - `EMBED_WORKSPACE` – base path for outputs; defaults to current working directory.
  - `EMBED_DATA_ROOT` – root directory for data artefacts (`<workspace>/data`).
  - `EMBED_CACHE_ROOT` – cache directory (`<workspace>/.cache`).
  - `EMBED_MODEL_CACHE` – where ESM weights are cached (`<cache_root>/models`).
  - `EMBED_MODEL_NAME` / `EMBED_MODEL_REVISION` – identify the ESM3 checkpoint to load.
  - `EMBED_USE_LOCAL_MODEL` – toggle local vs remote model usage (`true` by default).
  - `EMBED_DEVICE` – device string (`cpu`, `cuda`, `mps`, or `auto`).
  - `EMBED_BATCH_SIZE` / `EMBED_MAX_SEQUENCE_LENGTH` – runtime sizing controls.
  - `EMBED_TRUNCATE_LONG_SEQUENCES` – set to `true` to automatically truncate sequences that exceed `EMBED_MAX_SEQUENCE_LENGTH` (defaults to `false`).

Call `default_config()` to obtain an `EmbeddingConfig` instance that already reflects these settings.

## Tests
Run the targeted suite for this module with:

```bash
python -m pytest tests/unit/embed tests/integration/embed \
  --cov=src.embed.core.validation \
  --cov=src.embed.pipelines.sequence \
  --cov=src.embed.cli.main \
  --cov=src.embed.io.filesystem
```

The integration smoke test `tests/integration/embed/test_sequence_e2e.py` uses
legacy FASTA snippets and stubbed backends, so it executes in seconds on CPU-only
machines.
