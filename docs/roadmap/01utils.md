# V3 Pipeline Debug & Fix Plan (LLM‑Ready Roadmap)

> **Purpose**: Give an LLM (e.g., Cursor/Claude Code) a precise, testable roadmap to **debug, patch, and verify** the v3 pretrain→finetune pipeline. Format follows our vibe‑PM / coding‑agent best practices: clear objectives, constraints, stepwise tasks, diffs-first changes, tests, end‑to‑end (E2E) validation, and Definition of Done.

---

## 0) Context Snapshot

* **Observed issues**

  1. **Missing pretrain checkpoint** due to wrong actual path without timestamped subfolder. Log:

     * Expected (correct): `models/v3_pipeline/v3_pretrain/v3_pipeline_YYYYmmdd_HHMMSS_pretrain/best_model.pth`
     * Actual (wrong):  `models/v3_pipeline/v3_pretrain/best_model.pth` (no timestamp subfolder)
     * Side effect: later runs overwrite `best_model.pth` because same filename.
  2. **Finetune save path bug**: script saves to `models/v3_finetune/<ts>/...` but correct is `models/v3_pipeline/v3_finetune/<ts>/...`.
  3. **Finetune dataset path mismatch**: Config specifies

     * `train_csv: data/membrane_protein/splits/finetune_train.csv`
     * `valid_csv: data/membrane_protein/splits/finetune_validation.csv`
       but log shows legacy paths (`data/membrane_protein/train.csv`, `valid.csv`) and crashes.
  4. **logs/v3\_pipeline/training\_steps.csv** collision: pretrain and finetune both write same filename → overwrites. Need timestamped filenames.

* **Key files to inspect & patch**

  * Shell: `scripts/train_v3_pipeline.sh`, `scripts/train_v3_light_pipeline.sh`
  * Python (examples; search for these):

    * `src/pretrain/v3_pretrain.py` (and any `save_checkpoint` / path utils)
    * `src/finetune/v3_finetune.py`
    * `src/utils/io.py` / `src/utils/paths.py` / `src/utils/logging_utils.py` (or similarly named util modules)
    * Any Hydra/argparse/YAML loader resolving `train_csv` / `valid_csv`

* **Principles** (per our AI‑assisted coding rules)

  * *Single‑source‑of‑truth* for paths; avoid duplicating string literals.
  * *No silent fallback*: raise on missing files; print resolved config.
  * *Deterministic artifacts*: include timestamp + stage + run‑id in artifact file/dir names.
  * *CLI > YAML precedence*; env vars optional; log the final effective config.

---

Following the **minimal engineer rule** (DRY + single source of truth), the cleanest solution is to centralize all path- and artifact-related behaviors into **three unified utility functions** inside `src/utils`:

### 1. `save_checkpoint(model, stage, run_id, best=True, ema=False, swa=False)`

* Builds canonical checkpoint path automatically based on:
  * `model`: `v3` or `v1` or etc.
  * `stage`: `"pretrain"` or `"finetune"`
  * `run_id`: timestamp, e.g. `20250909_153012`
  * `best`: decide between `best_model.pth` and `checkpoint_epoch_X.pth`
* Ensures directories exist.
* Unified dir: `models/<model_name>/<stage>/<RUN_ID>/best_model.pth or checkpoint_epoch_X.pth`
* if ema=True then save both raw model and ema model(default), if swa=True, then also save swa:
`.../best_model.pth` and `.../best_model_ema.pth`

### 2. `load_checkpoint(model, stage, run_id, best=True, ema=False, swa=False)`

* load the check point from the unified dir: `models/<model_name>/<stage>/<RUN_ID>/best_model.pth or checkpoint_epoch_X.pth`
* Validates that the checkpoint exists; raises clear `FileNotFoundError` otherwise.
* Prints/logs the resolved path for transparency.

### 3. `save_log(model, stage, run_id, row_dict)`

* Writes training logs into:

  ```
  logs/<model_name>/<stage>/<RUN_ID>/log.log
  logs/<model_name>/<stage>/<RUN_ID>/training_step.csv
  ```
* Appends safely if file exists.

---

## M1 — Plan `paths.py` (canonical path resolver)

**Goal:** freeze the directory scheme used by save/load/log.

> **Role:** You are a senior ML infra engineer.
> **Task:** Produce a concise implementation plan for a minimal `src/utils/paths.py` that centralizes artifact paths for a PPI prediction project. Follow single responsibility, reuse, simplicity.
> **Canonical layout:**
>
> * Checkpoints: `models/<model_name>/<stage>/<RUN_ID>/best_model.pth` or `checkpoint_epoch_<E>.pth`
> * Optional variants when saving: `best_model_ema.pth`, `best_model_swa.pth`
> * Logs:
>
>   * `logs/<model_name>/<stage>/<RUN_ID>/log.log`
>   * `logs/<model_name>/<stage>/<RUN_ID>/training_step.csv`
>
> **Expose exactly these functions (final names and signatures):**
>
> * `run_dir(model: str, stage: str, run_id: str) -> Path`
> * `checkpoint_paths(model: str, stage: str, run_id: str, *, best: bool, epoch: int | None, ema: bool, swa: bool) -> dict[str, Path]`
>
>   * Returns a dict with keys: `"dir"`, `"model"`, and optionally `"ema"`, `"swa"` when `ema/swa` are True.
>   * If `best is False` and `epoch is None`, raise `ValueError("epoch required when best=False")`.
> * `log_paths(model: str, stage: str, run_id: str) -> dict[str, Path]` with keys `"dir"`, `"log_file"`, `"csv_file"`.
> * `ensure_dir(p: Path) -> Path` (idempotent mkdir, returns `p`).
>
> **Constraints & style:**
>
> * No business logic (no torch here), just paths + mkdir.
> * Validate `model` in `{v1, v3, ...}` and `stage` in `{pretrain, finetune}` via small allow-lists that are easy to extend (do not over-engineer).
> * Keep functions pure except directory creation when explicitly called.
>
> **Deliverables:**
>
> 1. A short design rationale (200–300 words).
> 2. A checklist “Definition of Done”.
> 3. Optional minimal code (≤80 lines) with docstrings and 3–4 tiny unit tests (path shape, epoch check, ema/swa keys, mkdir idempotency).
> 4. A usage snippet showing the returned dict for `best=True` vs `best=False, epoch=12`.
>
> **Non-goals:** logging, saving tensors, CSV writing.

---

## M2 — Plan `checkpoint.py` (unified save/load wrappers)

**Goal:** implement the three behaviors around checkpoints using `paths.py`.

> **Role:** You are a pragmatic PyTorch engineer.
> **Task:** Write a plan for a minimal `src/utils/checkpoint.py` that depends on `paths.py` and implements:
>
> 1. `save_checkpoint(model, stage, run_id, best=True, ema=False, swa=False, *, model_name: str, optimizer=None, scheduler=None, epoch: int | None=None, extra: dict | None=None) -> dict[str, Path]`
>
>    * Use `paths.checkpoint_paths(...)` to resolve paths.
>    * Save a dict with keys: `"model_state"`, and when provided `"optimizer_state"`, `"scheduler_state"`, `"epoch"`, `"extra"`.
>    * If `ema=True`, require `getattr(model, "ema_state_dict", None)` or accept a passed `ema_state_dict` kwarg; gracefully skip and log a one-liner if unavailable.
>    * If `swa=True`, similarly accept `swa_state_dict` or `getattr(model, "swa_state_dict", None)`.
>    * Return the dict of actually written paths (e.g., keys `"model"`, `"ema"`, `"swa"`).
> 2. `load_checkpoint(model, stage, run_id, best=True, ema=False, swa=False, *, model_name: str, optimizer=None, scheduler=None, epoch: int | None=None, strict: bool=True) -> dict`
>
>    * Resolve the path via `paths.checkpoint_paths(...)` and choose the correct file (`model`, `ema`, or `swa` based on flags).
>    * Validate file exists; otherwise raise `FileNotFoundError` with the resolved absolute path in the message.
>    * Load state dicts safely (`model.load_state_dict(...)`; load optimizer/scheduler only if provided).
>    * Return a small report dict: `{"path": Path, "epoch": int|None, "keys": list}`.
>
> **API notes:**
>
> * Keep the public signature aligned with the user contract (model\_name, stage, run\_id, best/ema/swa). Because `best=False` requires an epoch in the filename, accept `epoch` and validate it.
> * Never perform logging config here; use a passed-in logger or simple `print` for the resolved path (one line).
>
> **Deliverables:**
>
> 1. A bullet list of steps for save and load flows.
> 2. DoD checklist (state\_dict-only, epoch validation, clear error on missing file, returns saved/loaded paths).
> 3. Minimal code (≤120 lines) with docstrings and 3 tiny tests (best vs epoch path; ema/swa optional; missing file error shows absolute path).
> 4. Example usage snippets for both best and epoch checkpoints.

---

## M3 — Plan `logging.py` (single logger + CSV appender)

**Goal:** single place for `log.log` and `training_step.csv`.

> **Role:** You are designing a tiny logging utility for training loops.
> **Task:** Plan a minimal `src/utils/logging.py` that uses `paths.log_paths(...)` and implements:
>
> * `get_logger(name: str, model_name: str, stage: str, run_id: str, level: str="INFO") -> logging.Logger`
>
>   * Create directory via `paths.ensure_dir(log_paths(... )["dir"])`.
>   * Configure a single file handler to `log.log` and a stream handler. Compact formatter: `'%(asctime)s | %(levelname)s | %(name)s | %(message)s'`.
>   * Be idempotent: avoid duplicate handlers on repeated calls.
> * `append_training_row(model_name: str, stage: str, run_id: str, row: dict) -> Path`
>
>   * Append to `training_step.csv`, creating header on first write.
>   * Stable, explicit column order: if file exists, reuse existing header; else use `sorted(row.keys())`.
>   * Return the CSV file path.
>
> **Deliverables:**
>
> 1. Short rationale and DoD checklist (idempotent handlers, header once, safe append).
> 2. Minimal implementation (≤100 lines) with docstrings.
> 3. 2 quick tests (header creation; second append keeps header & order).
> 4. One usage snippet that pairs with `get_logger(...).info("...")` and `append_training_row(...)`.

---

## M4 — Plan `seed.py` (reproducibility switches)

**Goal:** Centralized seeding for reproducible experiments with optional deterministic mode.

> **Role:** You are a reproducibility engineer ensuring consistent ML experiments.
> **Task:** Write a minimal `src/utils/seed.py` that implements:
>
> * `set_seed(seed: int, deterministic: bool = False, logger: Optional[logging.Logger] = None) -> Dict[str, Any]`
>
>   * Seeds Python `random`, NumPy, PyTorch (CPU + CUDA if available).
>   * When `deterministic=True`: sets `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `benchmark=False`.
>   * Automatically sets environment variables: `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` (fallback to `":16:8"`).
>   * Returns state dict: `{"seed": int, "prev": {...}, "cuda": bool, "numpy": bool, "deterministic": bool}`.
>   * If `logger` provided, log seed values and deterministic mode warning.
>
> **Error handling:**
>
> * Raise clear errors for missing dependencies (NumPy, PyTorch).
> * No graceful fallbacks—fail fast if imports missing.
>
> **Reversible deterministic mode:**
>
> * Return previous state in `"prev"` dict for restoration.
> * Caller responsible for restoration logic.
>
> **Deliverables:**
>
> 1. Minimal implementation (≤80 lines) with Google-style docstrings.
> 2. Return value spec and example usage.
> 3. 3 minimal tests: same seed consistency, different seed variance, CUDA smoke test.
> 4. Single warning for deterministic mode performance impact.
>
> **Non-goals:** Worker seeding helpers, state restoration functions, complex validation.

---

`metrics.py`:

### Function

* `bin_classification_metrics(y_true, *, y_pred=None, y_score=None) -> dict`

  * **Inputs:**

    * `y_true`: 1D array-like of {0,1}.
    * `y_pred` (optional): 1D array-like of {0,1}. If provided, used for acc/precision/recall/specificity/F1/MCC/confusion.
    * `y_score` (optional): 1D probabilities for the positive class. If provided, used for AUROC/AUPRC/AP.

      * If `y_pred` is **None**, compute `y_pred = (y_score >= 0.5)` (no threshold arg needed).
  * **At least one** of `y_pred` or `y_score` must be given.

### Outputs (single dict, no thresholds to choose)

* `accuracy`, `precision`, `recall` (== sensitivity), `specificity`, `f1`, `mcc`
* `auroc`, `auprc`, `average_precision` (AP)
* `tp`, `fp`, `tn`, `fn` (from confusion matrix)

### Requirements

* **Keep it sklearn-backed & framework-agnostic** (accept numpy/torch lists; internally convert to numpy).
* **No domain logic** (no subset filtering).
* **Robust defaults:** `zero_division=0`; if a metric can’t be computed (e.g., `y_score` missing for AUROC), return `np.nan`.
* **No thresholds in API.** If only `y_score` is given, use **0.5** to create `y_pred` just for label-based metrics.

---

## M5 - `utils/metrics.py` Module

**Clarifications:**
- Binary enforcement: Strict {0,1} or bool validation, raise ValueError for multi-class
- Score validation: Require finite values within [0,1] range, no auto-coercion from logits
- Empty arrays: Raise ValueError for empty or mismatched length inputs
- Warning handling: Suppress sklearn warnings internally, return np.nan for uncomputable metrics
- Framework agnostic: Accept numpy/torch/lists, convert to numpy internally

### Function

* `bin_classification_metrics(y_true, *, y_pred=None, y_score=None) -> dict`

  * **Inputs:**
    * `y_true`: 1D array-like of {0,1} or bool. Strict binary validation.
    * `y_pred` (optional): 1D array-like of {0,1} or bool. If provided, used for label-based metrics.
    * `y_score` (optional): 1D probabilities [0,1] for positive class. Used for AUROC/AUPRC/AP.
    * If `y_pred` is None, compute `y_pred = (y_score >= 0.5).astype(int)`
  * **At least one** of `y_pred` or `y_score` must be given.
  * **All inputs** must be non-empty and equal length.

### Outputs (single dict)

* **Label-based:** `accuracy`, `precision`, `recall`, `specificity`, `f1`, `mcc`
* **Score-based:** `auroc`, `auprc`, `average_precision` 
* **Confusion:** `tp`, `fp`, `tn`, `fn`

### Requirements

* **Sklearn-backed & framework-agnostic:** Accept numpy/torch/lists; convert to numpy internally
* **Strict validation:** Binary-only {0,1}, scores in [0,1], non-empty equal-length arrays
* **Robust defaults:** `zero_division=0`; return `np.nan` for uncomputable metrics (e.g., single-class AUROC)
* **Clean output:** Suppress sklearn warnings internally

---

## M6 - `utils/data_io.py` Module

**Clarifications:**
- Variable sequence lengths: Return dict format `{protein_id: embedding_array}` for flexibility
- Single file policy: Multiple embedding files should raise error (pipeline uses single files)
- Function scope: Create new minimal functions fitting existing data formats, ignore existing functions
- Error handling: Defensive-lite validation (file existence, basic format checks, UniProt patterns)
- Path flexibility: Accept `str | Path | list[str | Path]`, normalize internally to Path

### Functions

* `read_pairs_csv(path: Union[str, Path]) -> pd.DataFrame`

  * **Inputs:** CSV file path with protein pairs
    * Auto-detect legacy columns `(p1, p2, Truth)` vs standard `(uniprotID_A, uniprotID_B, isInteraction)`
    * Normalize to standard `['protein_a', 'protein_b', 'label']` output columns
    * Validate binary labels `{0,1}` and basic UniProt ID pattern `[A-NR-Z0-9]{6,10}`
  * **Outputs:** Normalized DataFrame with validated protein pairs
  * **Error handling:** Warn and drop invalid rows, raise ValueError for fatal format issues

* `load_embeddings(paths: Union[str, Path, list]) -> Dict[str, Union[np.ndarray, torch.Tensor]]`

  * **Inputs:** Single embedding file path (raise error if multiple files provided)
    * Expected format: pickle dict with `{protein_id: {'embeddings': array, ...}}`
    * Validate embedding shape rank `==3` and last dimension `==1536`
  * **Outputs:** Dict mapping protein_id to embedding arrays (preserve numpy/torch type)
  * **Error handling:** File existence, readable pickle, required keys, shape validation

### Requirements

* **Minimal scope:** Light I/O utilities without heavy transformations or domain logic
* **Framework agnostic:** Preserve numpy/torch types as found in source files
* **Memory efficient:** Load per-file and merge, avoid loading everything simultaneously
* **Path normalization:** Accept flexible path types, no glob expansion (caller responsibility)

---

## M7 - `utils/early_stop.py` Module

**Clarifications:**
- Mode direction: Add `mode={'min','max'}` with default `'min'` (most monitor val_loss)
- Initial state: `best_score = None` until first update (explicit and simple)
- Min delta: Absolute improvement (not relative) - `new <= best - min_delta` for min mode
- Invalid metrics: Ignore `None/NaN/inf`, don't bump patience, emit warning once
- Patience semantics: Count epochs with no improvement, stop when `> patience`
- Reset functionality: Provide `reset()` method to clear state for phase changes

### Class

* `EarlyStopping(patience: int = 10, min_delta: float = 0.0, mode: str = 'min')`

  * **Methods:**
    * `update(metric: float) -> bool` - Update with new metric, return True if should stop
    * `reset() -> None` - Clear best_score and counters for reuse
  * **Properties:**
    * `best_score: Optional[float]` - Best metric seen so far, None before first update
    * `epochs_since_improvement: int` - Counter for patience tracking
  * **Error handling:** Ignore invalid metrics (None/NaN/inf), emit warning, don't count toward patience

### Requirements

* **Pure utility:** No I/O, no checkpointing, no training-loop logic
* **Framework alignment:** Follow Keras/PyTorch Lightning/XGBoost patterns
* **PPI training ready:** Monitor val_loss (min) or validation metrics (max) with small patience

## M8 - **`__init__.py` — curated public surface**

  * Export only the stable helpers you expect other packages to use (e.g., `from .checkpoint import save_checkpoint, load_checkpoint`), which encourages reuse over re-implementation.

---

## Working rules for this folder (keep it simple)

1. **One file = one responsibility.** If a function doesn’t fit the file’s name, it’s probably in the wrong place (SRP). ([Wikipedia][7], [Stackify][9])
2. **Prefer reuse over new code.** Before adding a helper, search for an existing one (e.g., don’t write another `parse_data()` or metrics calculator). ([Medium][8])
3. **Name things clearly and consistently.** Lowercase module names; avoid long or clever names. ([Python Guide][1], [Dagster][2])
4. **Keep utilities framework-agnostic when possible.** (e.g., metrics should accept NumPy/torch arrays alike.)
5. **Centralize logging, seeding, and checkpointing.** These are cross-cutting concerns—one place prevents “script sprawl.” Use the unified helpers above. ([Python documentation][5], [PyTorch Documentation][6])

## Prompt template

We need to implement the M. Tell me your plan first; don't write any code yet. Think as long as you need and ask me any questions if you need clarifications before coding.
We need to implement the M5(metrics.py) in /src/utils which I just give a brief intro. Tell me your plan first; don't write any code yet. Think as long as you need and ask me any questions if you need clarifications before coding.