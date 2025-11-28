# Embed Roadmap

This document tracks the plan for rebuilding the embed module on top of the new lightweight structure. It also catalogs the high-value assets preserved in `archive/legacy_module/` that can be reused instead of reimplemented.

## Legacy Assets Worth Reusing

- **Sequence normalization outputs (Milestone 1)**
  - `archive/legacy_module/data/sequences/all_proteins.fasta`
  - Metadata: `protein_index.json`, `sequence_choice.json`
  - Helpful for validating the new loaders and as fixtures when writing tests.
- **Structure discovery and consolidation (Milestones 2 & 3)**
  - Consolidated tables/NPZs under `archive/legacy_module/data/consolidated/`
  - Raw coordinate downloads in `archive/legacy_module/data/structures/coordinates/`
  - These provide ready-made structure inputs for the future multimodal pipeline to consume.
- **Vendored UniProt→PDB mapper**
  - `archive/legacy_module/Uniprot-PDB-mapper/` originates from https://github.com/iriziotis/Uniprot-PDB-mapper.git.
  - The script and harness (`m2_process_all_proteins.sh`) are production-grade for resolving UniProt IDs to PDB assemblies and should be wrapped, not rewritten.
- **Reference scripts**
  - `consolidate_structure_data.py` and `consolidate_esm3_data.py` document how consolidated datasets were produced; reuse the parsing logic when recreating IO helpers.
- **Documentation**
  - `archive/legacy_module/plan.md` and `status.md` contain acceptance criteria, logging expectations, and DoD definitions that can inform new milestones.

## Milestone Plan

### Milestone A – Sequence Pipeline Integration
1. **Stabilize configuration**
   - Flesh out `EmbeddingConfig` and `default_config()` with all fields needed by the pipeline (model cache paths, API tokens, device selection).
   - Add environment-driven overrides and document them in `docs/`.
2. **Port core utilities**+
   - Move validation/cleaning helpers from legacy `core/utils.py` into the new `core/validation.py` (and adjacent modules), ensuring tests cover edge cases.
3. **Implement `SequenceEmbedder`**
   - Reuse the logic from `archive/legacy_module/pipelines/sequence_only_esm3.py`, adapting it to the new base class and IO layer.
   - Provide clean error handling for missing local models and optional Forge API usage.
4. **Update CLI**
   - Wire the new pipeline into `src/embed/cli/main.py`, enabling batch processing from FASTA or JSON inputs.
5. **Add unit tests and smoke fixtures** ✅
   - Mocked ESM3 dependencies cover single/batch paths and CLI execution (see `tests/unit/embed/` and `tests/integration/embed/`).
   - Legacy FASTA snippets provide a CPU-friendly smoke test for `SequenceEmbedder` and the CLI.
6. **Documentation** ✅
   - Quickstart, configuration, and testing instructions updated in `src/embed/README.md` and `src/embed/docs/README.md`.
7. **Shell entrypoints** ✅
   - `scripts/embed.sh` helper wraps `python -m src.embed.cli.main` for both sequence and multimodal modes.
   - Both modes support: `--input`, `--output`, `--input-format`, `--data-root`, `--device`.
   - Multimodal mode replaces placeholder implementation with functional pipeline.
   - CLI supports `--mode` flag with precedence-based `--data-root` resolution (CLI > env > default).
   - Help text updated with multimodal examples and usage instructions.

### Testing Requirements
- Mirror `src/embed/` in `tests/unit/embed/` with one-to-one module coverage (e.g., `core/`, `io/`, `pipelines/`, `cli/`).
- Keep each directory lightweight with focused unit tests; prefer fixtures over large data drops.
- Add smoke tests for CLI entrypoints and serialization helpers alongside unit coverage.
- **Style**: PEP 8 + type hints for public APIs.
- **Formatting/Lint**: run `ruff check .` (and `ruff format .` if enabled) before `pytest`.
- Target `pytest` statement coverage ≥ 90% for every module before marking milestones complete.

### Milestone B – Multimodal Pipeline Implementation
1. **Define data contracts**
   - Specify required inputs (sequence, coordinate arrays, residue masks) and how they map to consolidated artifacts.
2. **IO layer enhancements**
   - Promote reusable loaders from legacy consolidation scripts into `io/` to pull backbone coordinates, masks, and metadata.
   - Provide a helper to materialize residue masks and populate `data/embed/structures/masks/`.
3. **Model integration**
   - Implement structure-aware embedder that leverages ESM3 multimodal interfaces; ensure it can fall back to sequence mode when structures are missing.
4. **Backbone alignment validation** ✅
   - Reconcile consolidated sequences with backbone coordinate arrays so that every multimodal sample uses sequence-aligned per-residue tensors (`[N, CA, C, O]`).
   - Regenerate or trim sequences as needed when coordinates are missing residues; emit updated metadata documenting the aligned sequence used for embeddings.
   - **Integration tests**: Added comprehensive tests in `tests/unit/embed/io/test_structure.py` that validate alignment reconciliation against real production data:
     - `test_alignment_with_real_consolidated_data`: Validates 7 diverse proteins covering all categories (exact match, shorter/longer structure sequences with varying ratios).
     - `test_alignment_regression_snapshots`: Hardcoded regression tests for 3 specific proteins (P20472, A0AVK6, A0A087X2I1) with exact expected alignment outcomes.
     - `test_alignment_metadata_completeness`: Type and constraint validation for alignment metadata fields.
     - All tests use real consolidated data (`data/embed/consolidated/`) and fail if data is missing.
     - Tests marked with `@pytest.mark.integration` and `@pytest.mark.slow` where appropriate.
     - All 13 tests (10 existing + 3 new) pass successfully.
5. **Testing** ✅
   - Added deterministic fixtures with synthetic 3-residue proteins in `tests/unit/embed/pipelines/test_multimodal.py`.
   - Tests cover complete structures, partial structures (missing atoms), and fallback scenarios.
   - No network access required; all tests use hand-crafted NPZ/CSV fixtures.
6. **Instrumentation & logging** ✅
   - Added comprehensive logging to `src/embed/pipelines/multimodal.py` at INFO and DEBUG levels.
   - Logs capture: structure loading, backend initialization, embedding progress, and fallback triggers.
   - Example output: `INFO: P20472: Structure loaded - residues=110, coverage=100.00%, source=1rjv`.
7. **Docs & examples** ✅
   - Updated `src/embed/docs/README.md` with multimodal quickstart and workflow.
   - Added runtime expectations (0.5-5s per protein, memory requirements).
   - Documented common failure modes with solutions (8 scenarios covered).
   - Included logging configuration examples.

### Milestone C – AlphaFold3 & Structure Preparation
1. **Strategy decision**
   - Evaluate existing PDB downloads vs. generating AF3 predictions for missing structures.
   - Document coverage gaps: proteins in the dataset without experimental structures.
   - Define priority criteria for AF3 prediction (e.g., novel interactions, high-confidence membrane proteins).
2. **UniProt-PDB mapper integration**
   - Hook in the existing mapper harness from `archives/legacy_module/Uniprot-PDB-mapper/`.
   - Wrap the shell-based workflow (`m2_process_all_proteins.sh`) with Python adapter.
   - Add caching and incremental updates to avoid re-downloading existing structures.
3. **AlphaFold3 prediction pipeline**
   - Set up AF3 prediction workflow (API or local installation).
   - Batch prediction for proteins without PDB structures.
   - Quality control: pLDDT scores, coordinate validation, expected vs actual sequence lengths.
4. **Structure consolidation**
   - Merge AF3 predictions with existing PDB coordinates.
   - Update metadata to distinguish sources (`structure_source: "pdb"` vs `"alphafold3"`).
   - Regenerate consolidated NPZ and metadata files with expanded coverage.
5. **Validation & testing**
   - Compare AF3 vs PDB structures for proteins with both (correlation checks).
   - Integration tests ensuring AF3-predicted structures work seamlessly with multimodal embedder.
   - Document prediction confidence thresholds and filtering criteria.

### Milestone D – Operational polish (parallel work)
1. **Environment tooling**
   - Provide `Makefile`/Typer commands for common tasks (normalize sequences, run sequence/multimodal batch, consolidate outputs).
2. **Data directory realignment**
   - Introduce `data/embed/` with named subdirectories that replace hard-coded legacy paths.
   - **Planned next actions**:
     - Keep the canonical layout under `data/embed/` in sync with pipeline needs (currently `sequences/`, `metadata/`, `consolidated/`, `structures/`).
     - Refresh reusable artifacts from `archives/legacy_module/data/` as new snapshots are produced and ensure `data/membrane_protein/unique_proteins_with_sequences.csv` remains available.
     - Document layout and provenance changes in `data/embed/README.md`, linking back to the source directories for traceability.
3. **Archival clean-up**
   - Annotate legacy files with pointers to their new equivalents and remove duplicate docs once parity is achieved.
4. **CI hooks**
   - Add lint/type/test jobs specific to the embed module to catch regressions early.
5. **Release notes**
   - Maintain a changelog section in `docs/` summarizing milestone completions and migration guidance for downstream consumers.
