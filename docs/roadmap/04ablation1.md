# Ablation 1 Roadmap · Minimalist Baseline (V2)

## 🎯 Goals
- Ship **V2** ablation model that skips Siamese transformer layers while preserving cross-attention + head.
- Enable training/eval via existing factory + configs with zero duplication.
- Capture experiment hygiene: tests, config diff, logging, reporting.

---

## 🔍 Context & Inputs
- Reference architecture in `src/model/v3.py`.
- Experiment spec in `docs/report/report_10xx/ablation.md` (Setting 1).
- CI/test expectations in `AGENTS.md`.

---

## 🧭 Implementation Roadmap

### 1. Design Alignment
- [x] Re-read `docs/report/report_10xx/ablation.md` section for intent + metrics to monitor.
- [x] Inventory reusable components in `src/model/v3.py` (stem projection, cross-attn, MLP).
- [x] Confirm any config knobs required (e.g., `encoder_layers: 0` handling).

### 2. Code Changes

#### 2.1 New Model Definition
- [x] **Create** `src/model/v2.py`
  - Import `V3` + reuse helper modules (e.g., `SiameseSharedEncoder` pieces as needed).
  - Implement lightweight `SiameseStemEncoder` that:
    - Projects embeddings via linear stem (`input_dim → d_model`).
    - Applies token dropout when configured.
    - Bypasses transformer blocks; emits length-masked outputs for cross-attn stage.
  - Define `class V2(V3)` overriding `_build_model()` (or equivalent) to wire stem encoder.
  - Ensure `forward` diagnostics remain consistent (`diagnostics` keys present even if empty).
  - Provide module `__all__`, docstring, and register decorator if following V3 style.

#### 2.2 Registry & Exports
- [x] **Update** `src/model/__init__.py` to expose `V2`.
- [x] **Register** model name(s) within `src/model/v2.py` (e.g., `register("v2", V2)`).
- [x] **Amend** `src/model/factory.py`
  - Ensure `model: "v2"` reuses V3-style parameter pass-through.
  - Add log messaging parity for new model.
- [x] Update `src/run.py` orchestration to respect dynamic `model_config.model`.

### 3. Config & CLI Wiring
- [x] Decide whether to **add** dedicated config (`configs/v2_minimal.yaml`) or document override instructions.
- [x] If new config needed:
  - Derive from `configs/v3.yaml` with `model_config.model: "v2"` and adjusted notes.
- [x] Confirm CLI wrappers (e.g., `scripts/*`, `src/run.py`) require no changes; if they rely on registry they should work automatically.
- [x] Add `scripts/v2.sh` wrapper to export `DIPPI_CONFIG_PATH` and run the full pipeline.

### 4. Testing Coverage
- [x] **Create** `tests/unit/model/test_v2.py`
  - Initialization/validation parity.
  - Forward pass shape checks with/without lengths + distances.
  - Diagnostics coverage (attention lists empty, etc.).
- [x] **Extend** `tests/unit/model/test_model_factory.py`
  - Register V2 and assert factory instantiates from config dict.
  - Verify parameter count decreases vs V3 baseline (optional assertion/tolerance).
- [x] Run `python -m pytest tests/unit/model/test_v2.py tests/unit/model/test_model_factory.py`.

### 5. Experiment Execution (Post-Merge)
- [ ] Prepare training overrides (epochs, LR) if Siamese depth removed.
- [ ] Launch finetune run using updated config; log run ID.
- [ ] Track metrics (AUC, loss) for comparison with baseline.

### 6. Reporting & Docs
- [ ] Update `docs/report/report_10xx/ablation.md` with methodology, hyperparams, and results once experiment completes.
- [ ] Add summary entry to `docs/roadmap/changelog` or higher-level roadmap if maintained.
- [ ] Draft PR description following Context / Changes / Tests / Impact / Risks template.

---

## 🏃 Run Notes
- Use `configs/v2.yaml` as the canonical baseline configuration (sets `run.mode=full_pipeline`).
- Launch locally via `scripts/v2.sh` which exports `DIPPI_CONFIG_PATH` and runs `python -m src.run`.
- For manual runs, set `export DIPPI_CONFIG_PATH=$(pwd)/configs/v2.yaml` then execute `python -m src.run`.

---

## ✅ Acceptance Criteria
- V2 model available through factory + registry, sharing inference API with V3.
- Unit tests green; coverage demonstrates stem-only path.
- Roadmap tasks tracked; post-experiment results documented.
- No regression to existing V3 behaviour or configs.
