# Refactoring Plan for DIPPI v3

This document outlines the plan to refactor the `DIPPI` codebase into a more modular, object-oriented, and configuration-driven system.

## 1. High-Level Plan

This refactor will introduce an object-oriented, configuration-driven, and testable structure for the core ML pipeline (`model`, `train`, `finetune`, `evaluate`). We will replace monolithic scripts with composable classes that adhere to the Single Responsibility Principle. A `BaseTrainer` will encapsulate shared training logic (epochs, optimization, checkpointing, EMA/SWA), with `v3Pretrainer` and `v3Finetuner` subclasses implementing task-specific logic. A dedicated `Evaluator` class will handle metrics calculation. All components will be driven by a unified YAML configuration, eliminating hardcoded paths and hyperparameters. This approach will improve code reuse, simplify testing, and make experiments more reproducible, leveraging the existing `src/utils` module for common functionalities like logging, seeding, and I/O.

## 2. File-by-File Responsibilities

`src/model/base.py`: Defines a minimal `BaseModel` abstract class (`nn.Module`) with a `forward(self, batch)` interface.
`src/model/v3.py`: Defines `v3` class inheriting from `BaseModel`.
`src/factories.py`: (New) Implements `ModelFactory` and `TrainerFactory` to resolve classes from the config.
`src/utils/device.py`: (New/Refactor) A `DeviceManager` utility to handle device selection (`cpu/gpu`) and model wrapping (`DDP`).
`src/utils/logging.py`: (Modify/Ensure) Defines a `Logger` abstraction that `BaseTrainer` will use.
`src/train/base.py`: Implement a minimal `BaseTrainer` with the core training loop, device/seed wiring via `DeviceManager`, and lifecycle hooks/callbacks. It will be injected with a `Logger`.
`src/train/train_v3.py`: Build a `v3Pretrainer` class, inheriting from `BaseTrainer`, that implements pre-training logic.
`src/finetune/strategies.py`: (New) Defines a `FinetuneStrategy` class to encapsulate finetuning logic (e.g., layer freezing patterns, discriminative LR parameter groups).
`src/finetune/finetune_v3.py`: Implements a `v3Finetuner` that inherits from `BaseTrainer` and composes a `FinetuneStrategy`.
`src/evaluate/base.py`: Create a stateless `Evaluator` class with a method like `evaluate(model, dataloader, metrics) -> dict`.
`configs/v3.yaml`: a complete, combined config for `v3` model, including all the stages of `v3` pipeline.
`src/run.py`: A main entry point script that uses the factories and `DeviceManager` to instantiate components and run the specified task.

## 3. YAML Config

refer to `configs/v3.yaml`

## 4. Concrete improvements & recommendations

1.  **Prefer composition + callbacks over deep inheritance**: Keep `BaseTrainer` minimal (core loop, device/seed wiring, lifecycle hooks).
2.  **Provide a tiny public interface for `BaseModel`**: Minimal methods: `forward(self, batch)`. Keep these shallow adapters over `nn.Module` to enable consistent serialization and unit tests.
3.  **Model/Trainer factory & registry**: Add a `ModelFactory` and `TrainerFactory` that resolves class by `cfg.model.name` / `cfg.run.mode`. This centralizes mapping and simplifies `src/run.py`.
4.  **Distributed / mixed-precision / device handling**: Encapsulate device strategy in a small utility: `DeviceManager` that chooses `cpu/gpu` and handles model wrapping (DDP only in run script, not inside Trainer constructor). Ensure Trainer is checkpoint-aware of AMP/grad-scaler state.
5.  **Finetune-specific concerns**: Prefer a `FinetuneStrategy` that Trainer can accept (freeze patterns, per-layer lr multipliers). Provide utilities to produce parameter groups from regex patterns + lr multipliers.
6.  **Stateless Evaluator**: Make it stateless: `Evaluator.evaluate(model, dataloader, metrics) -> dict`. Keep heavy I/O (report writing) outside.
7.  **Testing & Contracts**: Unit test for each base class and abstraction. Document required config keys and types in `configs/README.md`.
8.  **Logging & Observability**: Inject a `Logger` abstraction (e.g., wrapping wandb/tensorboard). The trainer should only call `logger.log(metric_dict, step)`.

## 5. Test Checklist

-   **Unit Tests (`tests/unit/model`)**:
    -   `test_model.py`: Check `BaseModel` interface and `v3` model instantiation/forward pass.
    -   `test_factories.py`: Verify `ModelFactory` and `TrainerFactory` resolve correct classes.
    -   `test_strategies.py`: Test `FinetuneStrategy` parameter grouping logic.
    -   `test_trainer.py`: One-batch smoke test for `BaseTrainer` with mock components (logger, model, optimizer).
    -   `test_evaluator.py`: Check that the stateless `Evaluator` returns a correctly formatted metrics dictionary.
    -   `test_device.py`: Test `DeviceManager` device selection.
    -   `test_config.py`: Validate configuration loading.
-   **Integration Test (`tests/integration`)**:
    -   `test_e2e_pipeline.py`: A smoke test running a minimal pre-training and evaluation cycle on a small, synthetic dataset.

## 6. CI Hooks

-   **CI Workflow (`.github/workflows/ci.yaml`)**:
    -   **Linting**: Run `ruff check .` on every push.
    -   **Formatting**: Run `ruff format --check .` on every push.
    -   **Unit Tests**: Run `pytest tests/unit` on every push to `main` and on PRs.

## 7. Implementation Plan

# PR1 — Model Layer Refactor (BaseModel → ModelFactory → v3)

## Scope & Goals (brief)

* Extract a **stable, testable** model API.
* Centralize model construction via a **factory/registry** (no ad-hoc instantiation).
* Wrap the existing `v3` logic so it **accepts config via DI**, has **no hardcoded hyperparams/paths**, and **reuses `utils`** only (no re-implementations).
* Produce **unit tests + one-batch smoke** to protect the API.

---

## Milestone M1 — `BaseModel`

**Objective**: Define the minimal, stable interface all models must implement.

**Files (new/changed)**

* `src/model/base.py` (new)
* `tests/unit/model/test_base_model.py` (new)
* `configs/README.md` (append: required model keys)

**Public API (must)**

* `class BaseModel(nn.Module)`
  * `forward(self, batch: dict) -> dict` (abstract)
  * `predict(self, batch: dict) -> dict` (default wrapper)
  * `num_parameters(self, trainable_only: bool = True) -> int`
  * `name: str` (optional class attribute)

**API Design: Batch-Dict Contract**

* **Signature**: Both `forward()` and `predict()` use `batch: dict -> dict`
* **Required keys**: `{"emb_a", "emb_b"}` (protein embeddings)
* **Optional keys**: `{"len_a", "len_b"}` (sequence lengths)
* **Extra keys**: Ignored (forward compatibility)
* **Private helper**: `_forward_tensors(emb_a, emb_b, len_a=None, len_b=None)` for concrete models

**Rules & Constraints**

* **BaseModel stays abstract**: No config object or `**kwargs` in constructor
* **No hardcoding**: Model hyperparams passed via DI in concrete models
* **No I/O**: Model must not read files or touch filesystem
* **No global state**: Everything comes through constructor arguments
* **Loss lives outside**: Trainer or Loss module, not inside model
* **predict() wrapper**: `eval()` + `torch.no_grad()` + `return self.forward(batch)`

**Config (consumed later via DI)**

* `model_config.model` (name)
* `model_config.params.*` (primitive types only; nested OK)
* `model_config.pretrained_path` (optional; loaded by factory, not here)

**DoD (Definition of Done)**

* `BaseModel` has docstring, type hints, and raises `NotImplementedError` where appropriate
* `num_parameters()` calls `utils.param_count` (add to todos if missing)
* `predict()` properly enforces eval mode and no_grad
* Unit test validates the abstract interface (using a trivial mock subclass)

**Tests (must)**

* `test_abstract_methods_raise()`
* `test_predict_enforces_eval_and_no_grad()`
* `test_num_parameters_counts_trainable_only()`
* `test_batch_dict_contract()` (required/optional keys)

**Non-Goals**

* No type protocols/big typing scaffolds (add later if needed)
* No factory/registry patterns (that's M2)
* No config handling, I/O, or loss logic
* No serialization helpers here (use native `state_dict()`)

---

## Milestone M2 — `ModelFactory`

**Objective**: Centralize model construction + (optional) weight loading; zero logic outside.

**Files (new/changed)**

* `src/model/registry.py` (new, minimal name→class map)
* `src/model/factory.py` (new, ModelFactory with dual API)
* `src/model/__init__.py` (updated, exports `ModelFactory`, `register`)
* `tests/unit/model/test_model_factory.py` (new)

**Public API (must)**

* `class ModelFactory:`
  * `@staticmethod def from_config(cfg: dict, device=None, logger=None) -> BaseModel`
  * `@staticmethod def from_model_config(model_cfg: dict, device=None, logger=None) -> BaseModel`
* `def register(name: str, cls: Type[BaseModel]) -> None` *(from registry.py)*

**Dual API Design**

* **`from_config(cfg)`**: Extract `cfg["model_config"]` internally, supports full config files
* **`from_model_config(model_cfg)`**: Direct model config dict, thin convenience wrapper
* **No duplicate parsing**: both paths converge to same implementation

**Behavior & Requirements**

* **Resolve** model class by config `model` field (string, case-insensitive)
* **DI**: instantiate class with `**params` from config (no hardcoded defaults in factory)
* **Device**: if `device` provided, use simple `model.to(device)` (defer DDP/DeviceManager)
* **Pretrained weights**: if `pretrained_path` exists, load with configurable `strict` flag
* **State dict verification**: inspect `missing_keys`/`unexpected_keys`, log them (no checksum hashing)
* **Logging**: Accept optional `logger` param; if None, use `utils.get_logger("model_factory")`
* **Errors**: clear `ValueError` on unknown model with available options; `FileNotFoundError` on missing weights

**Config (read)**

```yaml
model_config:
  model: v3                # case-insensitive model name
  params: { ... }          # only primitives/arrays, injected via **kwargs
  pretrained_path: null    # optional path to weights
  strict_load: true        # optional, controls state_dict loading
```

**Testing Strategy**

* **Mock models only**: no dependency on real v1/v3 implementations
* **Use actual config**: test with `configs/v3.yaml` for realistic scenarios
* **Registry isolation**: tests register/cleanup mock models independently

**DoD (Definition of Done)**

* Factory constructs models with config params via DI; **no hardcoded defaults** in factory
* Pretrained loading works with `strict` control and proper error reporting
* Unknown model names raise helpful error listing available registry keys
* Both factory methods work correctly with no duplicate parsing logic

**Tests (must)**

* `test_from_config_extracts_model_config_section()`
* `test_from_model_config_direct_usage()`
* `test_resolves_model_case_insensitive()`
* `test_injects_params_no_hardcode()`
* `test_loads_pretrained_when_path_present()`
* `test_state_dict_verification_logging()`
* `test_raises_on_unknown_model_with_suggestions()`
* `test_uses_actual_v3_config()` *(with configs/v3.yaml)*

**Non-Goals**

* No optimizer/scheduler creation; no DDP wrapping; no dataset logic
* No DeviceManager implementation (defer to later milestone)
* No complex state dict checksum verification (simple inspection only)

---

## Milestone M3 — `v3` (Model Adaptation)

**Objective**: Wrap existing `v3` into `BaseModel`, remove hardcodes, and route **all** shared ops to `utils`.

**Files (new/changed)**

* `src/model/v3.py` (refactor)
* `tests/unit/model/test_v3.py` (new)

**Public API (must)**

* `class V3(BaseModel):`

  * `model = "v3"` # model name
  * `__init__(self, **params)` where `params` is exactly the subset used (validate keys; error on unknown if `cfg.model_config.params.strict=True`)
  * `forward(self, batch: dict) -> dict`

    * Input (minimum): embeddings or token indices for **two proteins** (`"a"`, `"b"`) or a pre-built pair (document accepted keys).
    * Output: `{"logits": Tensor[B], "aux": {"attn": ..., "gates": ...}}` (aux optional).
  * `predict(self, batch: dict) -> dict` (inherited behavior from BaseModel)

**Parameter Intake (no hardcode; all DI)**

* Architecture: `input_dim`, `d_model`, `encoder_layers`, `cross_attn_layers`, `n_heads`, `max_sequence_length`
* Regularization: `dropout`, `mlp_dropout`, `cross_attention_dropout`, `stochastic_depth`, `token_dropout`, head-gating flags/weights (nested under `regularization`)
* Geometry: `use_distance_bias`, `rbf_bins`, `rbf_max_distance` (nested under `geometry`)
* MLP Head: `hidden_dims`, `dropout`, `activation`, `norm` (nested under `mlp_head`)
* MC Dropout: `use_mc_dropout_eval`, `mc_dropout_samples`
* Spectral norm: `spectral_norm` flag
* Any other toggles strictly from `params` (no defaults hidden inside; if defaults needed, define them **in YAML** and keep `v3` dumb)

**Must Reuse from `utils`**

* **init**: parameter initialization helper (e.g., kaiming/xavier via `utils.init`)
* **seed**: read current seed from `utils.random` if needed (no set here)
* **checkpoint**: rely on `state_dict()` only; saving is trainer’s job
* **typing**: any shared `Batch`/`TensorDict` types from `utils.typing` (if present)
* **logging**: `utils.logger` for one-line summary (param counts, arch fingerprint)

**Behavioral Requirements**

* `eval()` disables dropout etc.; `predict()` must be deterministic (unless MC-dropout explicitly requested by caller via batch flag).
* `.to(device)` must work with all tensors created inside.
* Batched pair processing: ensure shapes `[B, …]` match between partners; clear error if mismatch.
* No file I/O; no dataset assumptions beyond the declared batch keys.

**Config (consumed via factory)**

**Strict OOD Compliance (Refinements)**

* **No hardcoded dimensions**: MLP head sizes, RBF parameters from config only
* **No heuristics in model**: Token stripping moved to preprocessing utilities
* **No legacy batch keys**: Model accepts only canonical `dist_a/dist_b` keys
* **Utils extraction**: Spectral norm application, RBF encoder, batch normalization moved to `utils`
* **Structured config**: Nested sections for `geometry`, `mlp_head`, `regularization`

**Utility Extractions**

* `utils.data_io.clean_tokens(emb, lengths, strip_cls_eos: bool)` - CLS/EOS token removal
* `utils.data_io.normalize_batch_keys(batch)` - Legacy key normalization with deprecation warnings  
* `utils.model.apply_spectral_norm_(module, selectors)` - Spectral norm application
* `utils.model.RadialBasisFunction` - Reusable distance-to-bias encoder

**DoD**

* `v3` **imports zero** helpers duplicated from old code; only calls `utils.*` where needed.
* Constructor takes **only** config-injected params (no hidden defaults).
* `forward()` works on a synthetic minimal batch (two protein embeddings, pre-cleaned).
* Save→Load round-trip (`state_dict`) preserves outputs bit-exact in `eval()` (same seed & weights).
* Param count equals the logged one; gradients flow through all trainable tensors in a 1-batch autograd check.
* Model is pure: no heuristics, no legacy compatibility, strict config-driven behavior.

**Tests (must)**

* `test_init_with_complete_params()` (fails fast on missing/unknown)
* `test_forward_shape_and_keys()` (returns logits\[B], aux dict optional)
* `test_predict_is_deterministic_in_eval_mode()`
* `test_state_dict_roundtrip_equivalence()`
* `test_grad_flow_single_batch_autograd()` (loss = logits.mean(); backprop ok)

**Migration & Compatibility**

* If external code imports `from src/model/v3 import v3`, add a temporary alias `v3 = V3` and a deprecation warning (printed once via `utils.logger.warn`).
* Provide a short migration note in `CHANGELOG.md` (“use `ModelFactory.from_config` instead of direct constructor”).

# PR2: BaseTrainer -> v3Pretrainer

* Always **reuse from `src/utils`** (DeviceManager, checkpoint, logging) instead of re-implementing.
* **No duplication**: if a pattern exists (loops, seeding, logger), call it — don’t re-write.
* Keep **Base classes minimal & closed**: extend via subclass/hooks, never patch the loop.
* **Inject dependencies** (model, optimizer, logger), don’t import singletons inside.
* **Rank-0 only** side-effects (logging, saving).
* **Document contracts** (expected inputs/outputs) in each public method.

> Goal: add a minimal, OOD-clean training runtime with device/seed wiring and lifecycle hooks, plus a V3 pretrainer built on top. No bells & whistles; no duplicated logic.

---

## Milestone 1 — Core Runtime (DeviceManager + BaseTrainer)

### Scope (Single Responsibility)

**`src/utils/device.py` — DeviceManager**

* Select device: `cpu` or best available `cuda` based on a simple flag/config.
* Set global seeds across `random/np/torch` (and `torch.cuda` if present).
* Wrap/unwrap model for DDP (no training logic here; just conditional wrapping).
* Move batch tensors to device (shallow recursive: dict/list/tuple, tensor only).

**`src/train/base.py` — BaseTrainer**

* Own *only* the orchestration of a training loop; never define model math.
* Accept all runtime dependencies via constructor (Dependency Inversion):

  * `model`, `optimizer`, `loss_fn` (optional if model returns loss), `scheduler` (optional)
  * `train_loader`, `val_loader` (val optional)
  * `device_manager`, `logger`, `callbacks` (optional list)
  * `max_epochs`, `gradient_clip` (optional), `seed`, `amp` (off by default)
* Minimal lifecycle with overridable hooks (Open/Closed, Liskov):

  * `on_run_start/end`, `on_epoch_start/end`, `on_batch_start/end`, `on_validation_start/end`
* Minimal loop:

  * `fit()` → sets seed/device, runs epochs, trains each batch, optional val pass.
  * `train_one_epoch()` → iterates train loader, calls `self.step(batch)` for loss/metrics.
  * `validate_once()` (if val loader provided) → same as train but no optimizer step.
* Logging & callbacks:

  * Logger is *injected*; BaseTrainer only calls `logger.info()` and `logger.log_metrics(dict, step=...)`.
  * Callbacks are *passive* functions/objects with a minimal interface (see API).
* Error boundaries:

  * If `loss_fn` is None, `step()` implementation must return a scalar loss.
  * If both provided, `step()` may return per-batch preds/feats but BaseTrainer only needs loss/metrics.

* **Callbacks (minimal, optional):**

  * Protocol: any object with subset of methods:
    `on_run_start(trainer)`, `on_epoch_end(trainer, logs)`, `on_batch_end(trainer, logs)`, `on_run_end(trainer)`

### Out of Scope (Defer by design)

* Checkpointing/EMA/SWA, mixed precision AMP on by default, gradient accumulation, early stopping, LR finders, progress bars, rich callback system, metrics registry.

### Definition of Done (M1)

* Training on a tiny dataset(sample from the actual data path) runs for 1 epoch on CPU and GPU (if available) without errors.
* Seeds produce identical loss curves across two runs on same device.
* Hooks and callbacks are invoked in the expected order (smoke test).
* When `use_ddp=False`, wrapping is a no-op; when `use_ddp=True`, model is wrapped (no multi-process launcher in scope).

---

## Milestone 2 — V3 Pretrainer (Subclass Only)

### Scope (Single Responsibility)

**`src/train/train_v3.py` — v3Pretrainer**

* Provide *only* the pre-training step logic by subclassing `BaseTrainer`.
* Implement `step(batch)` to:

  1. move batch to device via `self.device_manager.to_device`,
  2. run forward on injected `model`,
  3. compute/return `(loss, metrics_dict)` using either:

     * injected `loss_fn` with model outputs, or
     * model-computed loss (if model returns `{"loss": ..., "metrics": ...}`).
* No data munging beyond device movement (keep preprocessing elsewhere).
* No checkpointing or schedulers beyond a single `scheduler.step()` per epoch if provided.

### Minimal Batch Contract (for PR2 only)

* Batch is a `dict[str, Any]`. The trainer is agnostic to specific keys; it passes the batch to the model/loss as-is.
* Optional helper (not implemented here): normalization utilities live outside the trainer.

### Out of Scope (Defer by design)

* Contrastive queue/memory banks, geometry/RBF features wiring, MC-Dropout, spectral norm toggles, advanced logging, checkpoint save/load, evaluation suites.

---

## OOD & Clean Architecture Rules (Enforced)

* **SRP**: DeviceManager handles device/seed/wrapping only; BaseTrainer handles orchestration only; v3Pretrainer handles per-batch training math delegation only.
* **OCP/LSP**: BaseTrainer is closed to modification; new trainers override `step` without changing loop semantics.
* **DIP**: Logger, DeviceManager, Model, Optimizer, Scheduler are injected; no hard imports/singletons.
* **No repetition**: Training loop exists only in `BaseTrainer`; v3Pretrainer contributes zero loop code.

## Deliverables Summary

* `src/utils/device.py` — `DeviceManager` (minimal).
* `src/train/base.py` — `BaseTrainer` with hooks/callbacks and core loop.
* `src/train/train_v3.py` — `v3Pretrainer` overriding only `step()`.

# PR3: 

## Rules
* Always **reuse from `src/utils`** (DeviceManager, checkpoint, logging, etc.) instead of re-implementing.
* **No duplication**: if a pattern exists (loops, seeding, logger), call it — don’t re-write.
* Keep **Base classes minimal & closed**: extend via subclass/hooks, never patch the loop.
* **Inject dependencies** (model, optimizer, logger), don’t import singletons inside.
* **Rank-0 only** side-effects (logging, saving).
* **Document contracts** (expected inputs/outputs) in each public method.

> Goal: implement a minimal, OOD-clean finetuning, and evaluate module.

# Milestone 1 — `FinetuneStrategy`

**File**: `src/finetune/strategies.py`
**Role (SRP):** Pure policy object for finetuning. It only (a) freezes/unfreezes layers, (b) builds optimizer param groups, (c) optionally returns a scheduler factory. No training loop, no logging.

**Minimal API**

* `FinetuneStrategy.from_config(cfg) -> FinetuneStrategy`
* `apply_freezing(model) -> None`
  Sets `requires_grad` according to strategy.
* `make_param_groups(model) -> list[dict]`
  Returns optimizer param groups (e.g., `{"params": [...], "lr": <float>}`).
* `build_optimizer(param_groups, optim_cfg) -> torch.optim.Optimizer`
* `build_scheduler(optimizer, sched_cfg, steps_per_epoch:int|None) -> _Scheduler|None`

**Supported minimal strategies**

* `"head_only"`: freeze backbone (by name/regex list), train head only.
* `"full"`: train all; two groups (backbone, head) with optional LR multipliers.
* `"staged_unfreeze"` (optional but still minimal): start as `head_only`, then unfreeze all at `cfg.unfreeze_at_epoch`.

**Config (minimal)**

```yaml
finetune:
  strategy: head_only | full | staged_unfreeze
  freeze_patterns: ["encoder.*"]          # used by head_only, staged_unfreeze
  lr: 3e-4                                # base lr
  lr_head_multiplier: 1.0                 # full only
  lr_backbone_multiplier: 0.5             # full only
  weight_decay: 0.01
  scheduler: none | cosine
  cosine_T0: 10                           # minimal knobs; ignore if scheduler=none
  unfreeze_at_epoch: 3                    # staged_unfreeze only
```

**Non-goals**

* No model/head construction
* No metric logic
* No dataloaders/loop

**DoD**

* Unit smoke test: `apply_freezing()` flips `requires_grad` as expected.
* `make_param_groups()` returns 1–2 groups deterministically.
* `build_optimizer()` returns AdamW without side effects.
* All functions are idempotent; no global state.

---

# Milestone 2 — `v3Finetuner`

**File**: `src/finetune/finetune_v3.py`
**Role (SRP):** A thin trainer for the V3 model’s **finetune** stage. Inherits `BaseTrainer`; composes a `FinetuneStrategy`. Implements **only** the pieces specific to finetune (loss, metrics hook wiring, staged unfreeze hook).

**Minimal API / Construction**

* `v3Finetuner(model, train_loader, val_loader, device_manager, logger, strategy, criterion, num_epochs, callbacks=None)`
* Uses `strategy.apply_freezing(model)` in `__init__` (or `setup()`).
* Builds `optimizer`/`scheduler` via strategy; plugs into `BaseTrainer` orchestration.

**Trainer surface (minimal overrides)**

* `step(batch) -> dict[str, float]`

  * Move `batch` to device via `self.device_manager.to_device`.
  * Forward: `logits = model(**batch_subset)` (reuse existing batch contract).
  * Loss: minimal **BCEWithLogitsLoss** (binary PPI).
  * Backprop handled by `BaseTrainer`.
  * Return `{"loss": float}` for logger.
* (Optional) `on_epoch_start(epoch)`

  * If `staged_unfreeze` & `epoch==unfreeze_at_epoch`: call `strategy.apply_freezing()` again, rebuild param groups/optimizer **once** (preserve scheduler if present).

**Validation/Eval hookup**

* Use a passed `Evaluator` in `validate_epoch()` (call once per epoch end).
* Track `val_loss` and (if available) `val_auc`.
* Select best checkpoint on `val_auc` if present; otherwise `val_loss`.

**Checkpoints & logs (reuse existing utils)**

* Save dir: `models/v3_pipeline/v3_finetune/<run_id>/`.
* Files: `last.pth`, `best_model.pth`, plus `metrics.jsonl` via logger helper.

**Non-goals**

* No data building, no model factory (run.py handles).
* No exotic losses, no multi-task.

**DoD**

* One epoch on a tiny loader runs end-to-end on CPU.
* Checkpoints written; best model updated when metric improves.
* `staged_unfreeze` path executes without realloc leaks (optimizer swap once).

---

# Milestone 3 — `Evaluator`

**File**: `src/evaluate/base.py`
**Role (SRP):** Stateless evaluation utility. It **does not** log, save, or mutate models. Just computes metrics on a dataloader.

**Minimal API**

* `Evaluator.evaluate(model, dataloader, metrics: list[str|Callable], device_manager, criterion=None) -> dict`

  * Sets `model.eval()` + `torch.no_grad()`.
  * Collects `logits`, `labels` across loader.
  * Computes requested metrics and (optionally) `val_loss` if `criterion` is provided.

**Built-in metric names (minimal)**

* `"loss"` (when `criterion` provided)
* `"accuracy@0.5"` (threshold 0.5)
* `"auroc"` (binary; if tensor-only impl available, else skip gracefully)
* `"ap"` (average precision; optional—compute if simple impl is present)

**Behavior**

* Unknown metric strings are ignored with a warning (not an error).
* Returns a flat `dict[str, float]`.

**Non-goals**

* No logging/printing, no plotting, no confusion matrix.

**DoD**

* On a dummy dataloader, returns stable keys/values.
* Works with CPU, small batches, and empty final partial batch.
* Doesn’t move or change model weights; leaves device state intact.


---

## PR3-wide Guardrails (OOD + Minimalism)

* **Single Responsibility** per module; compose, don’t entangle.
* **No duplication**: all I/O (ckpt/log paths, device moves) reuse existing utilities.
* **Small surface**: each class exposes only the methods listed above.
* **Determinism**: all randomness gated by `seed` from `run.py`.
* **Graceful degradation**: metrics not available → skipped with warning, not crash.

---

# PR4

## OBJECTIVE
Design a thin, deterministic `run.py` that:
- Parses a config e.g. `configs/v3.yaml`. If the existing v3.config doesn’t align with the implemented modules, you should adjust the config, not the modules, but must show me what's not aligned first before coding.
- Constructs components via existing factories/utilities.
- Runs one of the supported modes with zero duplicated logic:
  1) full_pipeline
  2) finetune_from_pretrain
  3) resume_finetune
  4) eval_only
- Uses canonical paths and unified checkpoint/log helpers.
- Exits with clear success/failure signals.

## AVAILABLE MODULES / APIS (IMPLEMENTED AND USED)
All APIs extracted during OOD refactor, no re-implementation in run.py:

- `src.utils`
  - Device & reproducibility: `DeviceManager`, `set_seed`
  - Logging: `get_logger`, `append_training_row`, `get_stage_logger`, `log_eval_summary`
  - Checkpointing: `save_checkpoint`, `load_checkpoint`, `resolve_checkpoint`
  - Paths: `run_dir`, `checkpoint_paths`, `log_paths`, `ensure_dir`
  - Data I/O: `read_pairs_csv`, `load_embeddings`, `clean_tokens`, `normalize_batch_keys`, `ProteinPairDataset`, `collate_variable_pairs`, `build_emb_index`, `build_dataloaders`
  - Configuration: `validate_sections`, `resolve_run_id`, `get_mode`, `normalize_evaluate_cfg`, `load_config_from_env`
  - Metrics: `bin_classification_metrics`

- `src.model`
  - `ModelFactory.from_config(cfg, device=None, logger=None)`

- `src.train`
  - `BaseTrainer`, `v3Pretrainer`, `v3Pretrainer.from_config()`

- `src.finetune`
  - `FinetuneStrategy.from_config(finetune_cfg)`, `v3Finetuner`

- `src.evaluate`
  - `Evaluator.evaluate(model, dataloader, metrics, device_manager, criterion=None)`

- External (minimal)
  - `torch.nn.BCEWithLogitsLoss`

## REQUIRED MODES & PIPELINES (SPECIFY EXACT CONTROL FLOW)

Define the precise orchestration for each mode, including which API is called, when, and what artifacts are read/written.

A) full_pipeline (IMPLEMENTED)
1. `cfg = load_config_from_env()`, `mode = get_mode(cfg)`, `validate_sections(cfg, mode)`.
2. `device_manager, minimal_logger = bootstrap_runtime(cfg)` → `DeviceManager(...)`, `set_seed(...)`.
3. `model = ModelFactory.from_config(cfg, device=device_manager.select_device(), logger=minimal_logger)`.
4. `emb_index = build_emb_index(data_config.embeddings_path)`, `dataloaders = build_dataloaders(cfg, emb_index, mode_to_loaders("full_pipeline"))`.
5. PRETRAIN:
   - `pretrain_run_id = resolve_run_id(cfg.run.pretrain_run_id)`, `pretrain_logger = get_stage_logger("pretrain", pretrain_run_id)`.
   - `pretrainer = v3Pretrainer.from_config(model, dataloaders.pretrain_train, dataloaders.pretrain_val, device_manager, pretrain_logger, cfg)` → builds optimizer internally.
   - `pretrainer.fit()` → trainer handles checkpointing automatically.
6. FINETUNE:
   - `finetune_run_id = resolve_run_id(cfg.run.finetune_run_id)`, `finetune_logger = get_stage_logger("finetune", finetune_run_id)`.
   - `pretrain_checkpoint = resolve_checkpoint("v3", "pretrain", pretrain_run_id, best=True)`, `load_checkpoint(model, checkpoint_path=str(pretrain_checkpoint))`.
   - `strategy = FinetuneStrategy.from_config(cfg.finetune_config.strategy_block)`, `evaluator = Evaluator()`.
   - `finetuner = v3Finetuner(model, dataloaders.finetune_train, dataloaders.finetune_val, device_manager, finetune_logger, strategy, ...)` → `finetuner.fit()`.
7. EVALUATE:
   - `run_evaluation(model, dataloaders, device_manager, cfg.evaluate, "finetune", finetune_run_id)` → handles multiple test sets, prints results, calls `log_eval_summary()`.
8. Exit with 0 on success.

B) finetune_from_pretrain (IMPLEMENTED)
1–4. Same as full_pipeline steps 1–4.
5. Load pretrain checkpoint:
   - `checkpoint_path = resolve_checkpoint("v3", "pretrain", pretrain_run_id, best=True, direct_path=cfg.run.checkpoint.path)`.
   - `load_checkpoint(model, checkpoint_path=str(checkpoint_path))`.
6. Run finetune and evaluate as in full_pipeline steps 6–7.

C) resume_finetune (IMPLEMENTED)
1–4. Same as full_pipeline steps 1–4.
5. Load finetune checkpoint:
   - `checkpoint_path = resolve_checkpoint("v3", "finetune", finetune_run_id, best=True, direct_path=cfg.run.checkpoint.path)`.
   - `load_checkpoint(model, checkpoint_path=str(checkpoint_path))`.
6. Continue finetune and evaluate as in full_pipeline steps 6–7.

D) eval_only (IMPLEMENTED)
1–4. Same as full_pipeline steps 1–4, `dataloaders = build_dataloaders(cfg, emb_index, {"eval"})`.
5. Load checkpoint:
   - `stage = cfg.run.checkpoint.stage or "finetune"`, `target_run_id = resolve_run_id(cfg.run[f"{stage}_run_id"])`.
   - `checkpoint_path = resolve_checkpoint("v3", stage, target_run_id, best=cfg.run.checkpoint.best, direct_path=cfg.run.checkpoint.path)`.
   - `load_checkpoint(model, checkpoint_path=str(checkpoint_path))`.
6. `run_evaluation(model, dataloaders, device_manager, cfg.evaluate, stage, eval_run_id)` → calls `log_eval_summary()`.
7. Exit.

## CONFIG CONTRACTS
No CLI. Config path is taken from env variable `DIPPI_CONFIG_PATH`.

Final minimal layout aligned to implemented modules:

```yaml
run:
  mode: full_pipeline | finetune_from_pretrain | resume_finetune | eval_only
  seed: 42
  pretrain_run_id: null  # Auto-timestamp if null
  finetune_run_id: null  # Auto-timestamp if null
  eval_run_id: null      # Auto-timestamp if null
  checkpoint:
    path: null           # Direct path or null for structured loading
    stage: pretrain | finetune
    run_id: null
    best: true

top_level_config:
  device: "cuda"

data_config:
  embeddings_path: "data/embedding/complete_soluble_proteins_embeddings.pkl"
  pretrain:
    train_csv: "data/membrane_protein/splits/pretrain_train.csv"
    valid_csv: "data/membrane_protein/splits/pretrain_validation.csv"
  finetune:
    train_csv: "data/membrane_protein/splits/finetune_train.csv"
    valid_csv: "data/membrane_protein/splits/finetune_validation.csv"
  evaluate:
    test_balanced: "data/membrane_protein/test_balanced.csv"    # Automatically handled
    test_realistic: "data/membrane_protein/test_realistic.csv" # by normalize_evaluate_cfg()
  dataloader:
    num_workers: 0
    pin_memory: true

model_config:
  model: "v3"
  input_dim: 1536
  d_model: 512
  encoder_layers: 2
  cross_attn_layers: 2
  n_heads: 8
  max_sequence_length: 1024
  spectral_norm: false
  use_mc_dropout_eval: false
  mc_dropout_samples: 8
  geometry:
    use_distance_bias: true
    rbf_bins: 16
    rbf_max_distance: 20.0
  mlp_head:
    hidden_dims: [256, 64]
    dropout: 0.25
    activation: "gelu"
    norm: "layernorm"
  regularization:
    dropout: 0.10

pretrain_config:
  epochs: 1
  batch_size: 64
  learning_rate: 6.0e-5
  weight_decay: 0.015
  gradient_clip_norm: 0.5

finetune_config:
  epochs: 1
  batch_size: 64
  gradient_clip_norm: 0.5
  strategy_block:
    strategy: head_only
    freeze_patterns: ["encoder.*"]
    lr: 5.0e-5
    weight_decay: 0.015
    scheduler: none
    cosine_T0: 10
    unfreeze_at_epoch: 2

evaluate:
  metrics: ["accuracy@0.5", "auroc", "ap", "loss"]
```

Changes vs original `configs/v3.yaml`:
- Added: `run.*` section for mode and run ID management.
- Added: `finetune_config.strategy_block.*` (required by `FinetuneStrategy`).
- Changed: `data_config.evaluate` keeps `test_balanced/test_realistic` (normalized by `normalize_evaluate_cfg()`).
- Simplified: Removed multi-gpu, onecycle schedulers, EMA/SWA, verbose logging toggles.

## PATHS & ARTIFACTS (CANONICAL; DO NOT DEVIATE)

Canonical per `src/utils/paths.py` (no deviation):

- Pretrain: `models/v3/pretrain/<run_id>/` (files saved by `save_checkpoint`)
- Finetune: `models/v3/finetune/<run_id>/` (owned by `v3Finetuner`)
- Logs: `logs/v3/<stage>/<run_id>/{log.log, training_step.csv}` via `get_logger`/`append_training_row`
- Eval: append one summary row to the finetune CSV; no separate eval file in M4.

## ERROR HANDLING & GUARDRAILS

* Hard error if `DIPPI_CONFIG_PATH` missing or YAML invalid.
* Validate required config sections per mode (see above). Fail fast with explicit messages.
* Checkpoint resolution errors print the attempted absolute path(s).
* Unknown metric names → warn and skip; never crash.
* No duplicated training logic in `run.py` (trainers own loops). No hidden defaults for run IDs.

## LOGGING / TELEMETRY

* Use `get_logger(name, model_name="v3", stage, run_id)` per stage.
* Trainers handle per-step/epoch logging. `run.py` appends only a final eval row.
* Print concise final metrics dict on eval completion.

## IMPLEMENTATION STATUS (COMPLETED ✅)

**All modes implemented and working:**
* `full_pipeline`: pretrain → finetune → evaluate with automatic checkpointing
* `finetune_from_pretrain`: load pretrain checkpoint → finetune → evaluate  
* `resume_finetune`: resume from finetune checkpoint → continue → evaluate
* `eval_only`: load checkpoint → evaluate on multiple test sets

**Key Implementation Features:**
* Thin orchestrator (432 lines) using only extracted utilities
* Zero training logic duplication - trainers own optimizer/scheduler construction
* `v3Pretrainer.from_config()` builds AdamW + loss internally
* `bootstrap_runtime()` for common setup (device, seed, logger)
* `run_evaluation()` handles multiple test sets (balanced + realistic)
* `resolve_checkpoint()` centralized path discovery with clear errors
* `build_dataloaders()` with mode-specific loader selection
* Config validation via `validate_sections()` and `normalize_evaluate_cfg()`

**Orchestration Implementation:**
* Components built via factories only: `ModelFactory`, `FinetuneStrategy`, `get_stage_logger`, `log_eval_summary`
* Checkpoint I/O via `resolve_checkpoint` + `load_checkpoint` 
* Multi-test evaluation with descriptive naming (eval_balanced, eval_realistic)
* Single exception boundary with clear error messages
* Automatic run ID generation with timestamp validation

## Quick Test Matrix
* **Integration**:

  * `full_pipeline`: pretrain(1) → finetune(1) → eval; both best checkpoints exist.
  * `finetune_from_pretrain`: resolves pretrain best via structured loader; trains and evals.
  * `resume_finetune`: resumes from highest `checkpoint_epoch_*.pth` and continues training.
  * `eval_only`: loads specified checkpoint, returns metrics dict, appends summary row.

## Implementation Plan (Refined)

### Core Strategy
- **File**: `src/run.py` (all helpers local, no new modules)
- **Testing**: `tests/integration/test_run_pipeline.py` with minimal config + tiny fixtures
- **Data**: Load all embeddings in memory upfront; dynamic batch-local padding

### Key Components
1. **Run ID Resolution**: Default to timestamp if None; validate if provided
2. **Config Validation**: Top-level sections only per mode (no deep validation)  
3. **Logging**: Minimal logger first (for seeding), upgrade to stage loggers later
4. **Dataset**: `ProteinPairDataset` class using `read_pairs_csv` + loaded `emb_index`
5. **Collate**: Dynamic padding to longest sample per batch (minimize waste)

### Integration Test Plan
- **Location**: `tests/integration/test_run_pipeline.py`
- **Fixtures**: `tests/fixtures/{tiny_config.yaml, tiny_pairs.csv, tiny_embeddings.pkl}`
- **Model**: Minimal size (d_model=64, 1 layer) for 4GB GPU compatibility
- **Coverage**: One test per mode (4 tests total)

### File Structure
```
src/run.py                           # Main implementation
tests/integration/test_run_pipeline.py   # Integration tests  
tests/fixtures/                      # Test data
  ├── tiny_config.yaml              # Minimal config
  ├── tiny_pairs.csv                # 20 protein pairs
  └── tiny_embeddings.pkl           # Corresponding embeddings
```

## Run.py OOD Minimal Orchestration Plan (v3)

### 1) Smells & Violations (Checklist)

- **`ProteinPairDataset` in `run.py`**: Data responsibility inside orchestrator; duplicates preprocessing (`clean_tokens`).
- **`collate_fn` in `run.py`**: Collate/padding logic embedded in orchestrator; not reusable/tested.
- **`build_dataloaders` in `run.py`**: Dataset/DataLoader assembly belongs to data utilities; repeated loader kwargs.
- **Optimizer construction in `run_full_pipeline`**: Orchestrator builds `AdamW`; should be handled by Trainer/Strategy.
- **Checkpoint saving in `run_full_pipeline`**: Manual `save_checkpoint`; should be owned by Trainer callbacks or a checkpoint helper invoked by Trainer.
- **Duplicated eval blocks**: Repeats `Evaluator()` creation, criterion selection, logging, and CSV append across modes.
- **Run bootstrapping repeated per mode**: Device selection, seeding, minimal logger setup repeated; not using canonical `get_logger`.
- **Checkpoint resolution logic spread in modes**: Direct `checkpoint_paths` calls and error wrapping; should be centralized.
- **Hidden globals / error handling**: Global `TIMESTAMP_PATTERN`; `print + sys.exit` scatter; prefer consistent exception boundaries and single exit point.
- **Config mismatch**: `data_config.evaluate` uses `test_balanced/test_realistic` while `run.py` expects `test_csv`.

### 2) Extraction Targets (Move out of `run.py`)

- **Config helpers → `src/utils/config.py`**
  - `validate_sections(cfg: dict, mode: str) -> None` — Top-level section presence per mode.
  - `resolve_run_id(value: Optional[str]) -> str` — Timestamp default + strict validation.
  - `get_mode(cfg: dict) -> str` — Extracts and validates `run.mode`.
  - `normalize_evaluate_cfg(data_cfg: dict) -> dict` — Returns `{"test_csv": str}` by choosing `test_balanced` or `test_realistic` via `data_cfg.get("evaluate.variant", "balanced")`.
  - **OOD rationale**: Config parsing/validation is cross-cutting infra; unit-testable and reusable.

- **Checkpoint resolution → `src/utils/checkpoint.py`**
  - `resolve_checkpoint(model_name: str, stage: str, run_id: Optional[str], *, best: bool = True, direct_path: Optional[str] = None) -> Path` — Returns canonical path or raises with absolute attempted path; uses `checkpoint_paths`.
  - **OOD rationale**: Single source of truth for checkpoint discovery and errors.

- **Data assembly → `src/utils/data_io.py`**
  - `build_emb_index(path: str|Path) -> dict` — Thin alias of `load_embeddings` with type normalization.
  - `ProteinPairDataset(csv_path: str, emb_index: dict, *, strip_tokens: bool = True) -> Dataset` — Move current class here; reuse `read_pairs_csv`, `clean_tokens`.
  - `collate_variable_pairs(batch: list[dict]) -> dict[str, Tensor]` — Move current `collate_fn`.
  - `build_dataloaders(cfg: dict, emb_index: dict, loaders: set[str]) -> dict[str, DataLoader]` — Centralize creation; keys: `pretrain_train`, `pretrain_val`, `finetune_train`, `finetune_val`, `eval`.
  - **OOD rationale**: Pure data responsibilities; easy to unit test; keeps orchestrator thin.

- **Logging bootstrap → `src/utils/logging.py`**
  - `get_stage_logger(stage: str, run_id: str, model_name: str = "v3") -> logging.Logger` — Thin wrapper over `get_logger` with stage/run-id semantics.
  - **OOD rationale**: Removes ad-hoc logger creation and handler wiring from orchestrator.

- **Trainer-owned components**
  - `v3Pretrainer.from_config(model, train_loader, val_loader, device_manager, logger, cfg: dict) -> v3Pretrainer` — Trainer builds optimizer/scheduler internally from `pretrain_config`.
  - `FinetuneStrategy.build_optimizer(...)` / `.build_scheduler(...)` — Already defined; ensure `v3Finetuner` uses them exclusively.
  - **OOD rationale**: Orchestrator never manages optimizers/schedulers.

- **Eval logging helper → `src/evaluate/base.py` or `src/utils/logging.py`**
  - `log_eval_summary(model_name: str, stage: str, run_id: str, metrics: dict) -> None` — Calls `append_training_row` and minimal stdout print.
  - **OOD rationale**: Removes duplicate append/print blocks from modes.

### 3) Mode Orchestration Refinement

- **Minimal state diagram (text)**
  - Start → Bootstrap (device, seed, logger) → BuildModel → BuildData → [LoadCheckpoint?] → [Train?] → [Evaluate?] → Exit
  - Transitions by mode:
    - `full_pipeline`: Bootstrap → BuildModel → BuildData → Pretrain → Finetune → Evaluate → Exit
    - `finetune_from_pretrain`: Bootstrap → BuildModel → BuildData → Load(pretrain) → Finetune → Evaluate → Exit
    - `resume_finetune`: Bootstrap → BuildModel → BuildData → Load(finetune) → Finetune → Evaluate → Exit
    - `eval_only`: Bootstrap → BuildModel → BuildData → Load(stage) → Evaluate → Exit

- **Control flow per mode (only orchestrate)**
  - Common prelude: parse config from `DIPPI_CONFIG_PATH` → `validate_sections(cfg, mode)` → `device_manager = DeviceManager(...)` → `set_seed(run.seed, deterministic=True, logger)` → `model = ModelFactory.from_config(cfg, device=device_manager.select_device(), logger)` → `emb_index = build_emb_index(data_config.embeddings_path)` → `dataloaders = build_dataloaders(cfg, emb_index, loaders_for_mode)`.
  - `full_pipeline`:
    - `pretrainer = v3Pretrainer.from_config(model, loaders.pretrain_train, loaders.pretrain_val, device_manager, get_stage_logger("pretrain", run.pretrain_run_id), cfg)` → `pretrainer.fit()` (trainer owns checkpointing).
    - `load_checkpoint(model, stage="pretrain", run_id=run.pretrain_run_id, model_name="v3", best=True)`.
    - `strategy = FinetuneStrategy.from_config(cfg.finetune_config.strategy_block)` → `finetuner = v3Finetuner(model, loaders.finetune_train, loaders.finetune_val, device_manager, get_stage_logger("finetune", run.finetune_run_id), strategy, epochs=cfg.finetune_config.epochs, evaluator=Evaluator())` → `finetuner.fit()`.
    - `metrics = Evaluator().evaluate(model, loaders.eval, metrics=cfg.evaluate.metrics, device_manager=device_manager, criterion=auto_if_requested)` → `log_eval_summary("v3", "finetune", run.finetune_run_id, metrics)`.
  - `finetune_from_pretrain`:
    - `ckpt = resolve_checkpoint("v3", "pretrain", run.pretrain_run_id, best=True, direct_path=run.checkpoint.path)` → `load_checkpoint(model, checkpoint_path=ckpt)`.
    - Finetune and Evaluate as above.
  - `resume_finetune`:
    - `ckpt = resolve_checkpoint("v3", "finetune", run.finetune_run_id, best=True, direct_path=run.checkpoint.path)` → `load_checkpoint(model, checkpoint_path=ckpt)`.
    - Finetune and Evaluate as above.
  - `eval_only`:
    - `ckpt = resolve_checkpoint("v3", stage=run.checkpoint.stage or "finetune", run_id=run.<stage>_run_id, best=run.checkpoint.best, direct_path=run.checkpoint.path)` → `load_checkpoint(model, checkpoint_path=ckpt)`.
    - Evaluate and `log_eval_summary` as above.

> Note: No training logic, optimizer creation, or checkpoint writes remain in `run.py`.

### 4) Factory Boundaries

- **Factories/utilities own construction**
  - `build_model` → `ModelFactory.from_config(...)`.
  - `build_dataloaders` → `src/utils/data_io.build_dataloaders(...)` (uses `ProteinPairDataset` and `collate_variable_pairs`).
  - `build_logger` → `get_stage_logger(stage, run_id, model_name)`.
  - `build_strategy` → `FinetuneStrategy.from_config(...)`.

- **Never in `run.py`**
  - Optimizer/scheduler construction (trainer/strategy only).
  - Checkpoint path guessing (use `resolve_checkpoint`).
  - Token cleaning/padding (data utilities only).

### 7) Acceptance Criteria (DoD)

- **Thin orchestrator**: `run.py` performs only parse → build → orchestrate → exit.
- **Zero duplication**: No logic duplicated from Trainers/Strategy/Evaluator/Utils.
- **Stable contracts**:
  - Trainers own optimizer/scheduler and checkpoint writes; `run.py` only calls `.fit()`.
  - Evaluator is stateless; logging of eval results done via a single helper.
  - Data utilities own dataset/collate/dataloader creation.
- **Config alignment**: `data_config.evaluate` normalized to a single `test_csv` (support variant selector for balanced/realistic).
- **Testability**: New helpers have single responsibility and are unit-testable in isolation.

# PR5: Minimal, implementation-ready milestone plan (config-TODO-driven)

Assumptions:
- Current v3 E2E pipeline works; we will not change public run modes or directory layout.
- We will only implement items explicitly called out in `configs/v3.yaml` TODOs, in small, composable steps.
- We will reuse existing abstractions (`BaseTrainer`, `FinetuneStrategy`, `Evaluator`, `utils/*`) and avoid new modules unless necessary for separation of concerns.

Milestone 1 — Logging cadence and validation frequency
- Scope:
  - Add `log_every_n_steps` and `validation_frequency` support for both pretrain and finetune.
  - No new metrics, no plotting, no fancy logging config yet.
- Modules:
  - `src/train/base.py`: gate per-batch logging via `log_every_n_steps`; run validation every N epochs via `validation_frequency`.
  - `src/run.py`: pass-through of these config values to trainers.
  - `src/utils/config.py`: validate optional fields.
- Acceptance:
  - If `log_every_n_steps` absent → behavior unchanged.
  - If `validation_frequency=N`, validation runs every N epochs (N>=1).
  - E2E still passes with defaults.
- Tests:
  - Unit: simulate few steps, verify logging cadence.
  - Unit: verify validation called at expected epochs.

Milestone 2 — Early stopping (minimal)
- Scope:
  - Enable `early_stopping_patience` (per TODO) for both stages.
  - Pretrain uses `val_loss`; finetune prefers `auroc` if available else `val_loss`.
- Modules:
  - `src/utils/early_stop.py`: already implemented; reuse.
  - `src/finetune/finetune_v3.py`: update `on_validation_end()` to update early stopper and set a stop flag.
  - `src/train/train_v3.py` (pretrainer): mirror logic using `val_loss`.
  - `src/train/base.py`: if stop flag set, break training loop cleanly.
  - `src/utils/config.py`: read/validate patience.
- Acceptance:
  - When patience is set and no improvement, training stops early.
  - Default absent → unchanged behavior.
- Tests:
  - Unit: synthetic dataloader with non-improving metric triggers stop at expected epoch.

Milestone 3 — Schedulers (OneCycle for pretrain; keep cosine for finetune)
- Scope:
  - Add `pretrain_config.scheduler: onecycle` with minimal args from config TODO.
  - Keep finetune using existing `FinetuneStrategy` scheduler (“none|cosine”).
- Modules:
  - `src/train/train_v3.py`: build OneCycleLR when configured; step scheduler per epoch (minimal).
  - `src/finetune/strategies.py`: no change (already supports “cosine”).
  - `src/utils/config.py`: validate scheduler fields if present.
- Acceptance:
  - If `scheduler: onecycle`, trainer constructs and steps OneCycleLR without errors.
  - Defaults remain unchanged.
- Tests:
  - Unit: pretrainer builds OneCycleLR with required params; LR evolves over epochs.

Milestone 4 — Mixed precision (AMP) toggle
- Scope:
  - Support `use_mixed_precision` and `amp_dtype` (`bf16` or `fp16`) for both stages.
- Modules:
  - `src/train/base.py`: wrap forward/backward in autocast + optional GradScaler; controlled by flags.
  - `src/run.py`: plumb flags from config.
  - `src/utils/config.py`: validate amp dtype.
- Acceptance:
  - AMP off by default; enabling produces numerically stable training on CUDA.
  - No API changes for models/strategies.
- Tests:
  - Unit: dry-run single epoch with AMP on; ensure no exceptions and optimizer steps occur.

Milestone 5 — Loss configuration: BCE options
- Scope:
  - Add optional `loss` block for both pretrain/finetune:
    - `type: "bce_with_logits"` (default)
    - `pos_weight: "auto" | float`
    - `label_smoothing: float` (0 defaults to off)
    - `use_class_weights: bool` (minimal: support per-sample weight via batch key if provided; no new data pass)
- Modules:
  - `src/train/train_v3.py` (pretrainer.from_config): build criterion per config; compute `pos_weight` as “auto” if available from training dataset metadata or a quick single pass over first K batches; fallback to 1.0 with warning.
  - `src/finetune/finetune_v3.py`: accept externally built criterion or build same way for symmetry.
  - `src/utils/config.py`: validate fields.
- Acceptance:
  - Defaults unchanged; setting fields adjusts BCE behavior.
- Tests:
  - Unit: criterion builds with configured label smoothing/pos_weight; shapes validated.

Milestone 6 — Minimal regularization hooks
- Scope:
  - Implement only low-effort, high-safety items from TODOs:
    - `regularization.l1_lambda`: add L1 penalty to loss (both stages).
    - `regularization.token_dropout`: optional dropout on inputs before encoder.
  - Defer complex items (stochastic depth, head gates, group lasso, entropy) to later PRs.
- Modules:
  - `src/model/v3.py`: optional `token_dropout` in encoder input path (flag read from model_config.regularization or model_config).
  - `src/train/train_v3.py` and `src/finetune/finetune_v3.py`: add L1 penalty term to loss if configured.
  - `configs/v3.yaml`: document supported subset.
- Acceptance:
  - With `l1_lambda>0`, total loss = base + l1.
  - With `token_dropout>0`, model forwards without shape errors; default off.
- Tests:
  - Unit: verify L1 term added; token dropout path runs with batch.

Milestone 7 — Lightweight logging enhancements
- Scope:
  - Add minimal support from `logging_config` TODOs:
    - `monitor_metric` (string) to decide “best” model (finetune); pretrain remains val_loss.
    - `save_best_only` (bool) to skip last checkpoint if desired.
    - `log_lr_schedule` (bool) to append current LR to per-step/epoch logs.
  - Defer plots and advanced per-step logs to later PRs.
- Modules:
  - `src/finetune/finetune_v3.py`: respect `monitor_metric`, `save_best_only`.
  - `src/train/base.py`: include LR in logs when enabled (use optimizer param group #0 LR).
  - `src/utils/logging.py`: no change; reuse CSV appender.
  - `src/utils/config.py`: validate fields.
- Acceptance:
  - Best checkpoint chosen by configured `monitor_metric`.
  - Optional LR column appears in CSV when enabled.
- Tests:
  - Unit: metric selection switches from AUC to loss by config; LR logged when requested.

Milestone 8 — Docs, config, and API surface updates
- Scope:
  - Update `README.md` and `instructions.md` signatures for new trainer knobs (AMP, early stopping, schedulers, loss options, minimal regularization, logging knobs).
  - Add commented examples to `configs/v3.yaml` (keeping safe defaults off).
- Modules:
  - `README.md`, `instructions.md`, `configs/v3.yaml`.
- Acceptance:
  - Clear, concise documentation; CI passes; examples runnable.

Out of scope for PR5 (defer to PR6+)
- Multi-GPU/DDP (complex orchestration and spawn/init; plan as a dedicated PR).
- EMA/SWA training (checkpoint saving is ready but needs state tracking during training).
- Advanced model regularization (stochastic depth, head gates, group lasso, entropy) and attention diagnostics/plots.

Execution notes
- Keep edits scoped; reuse existing utilities:
  - Early stopping reuses `utils/early_stop.EarlyStopping`.
  - Finetune scheduler remains in `FinetuneStrategy`; OneCycle only added for pretrain.
  - Logging uses `utils/logging` CSV appender; no new logging backends.
- Maintain default behavior: all new features off unless explicitly enabled.
- Follow coding standards, type hints, and update `instructions.md` when public APIs change.
- Run: ruff check/format + pytest; ensure e2e test stays green.

Deliverable checklist per milestone
- Edits made to specified modules only.
- Unit tests for each feature.
- Minimal config updates/documentation.
- CI green before merging.

# PR6 Roadmap — Minimal, implementation-ready milestones (reuse-first, no duplication)

Assumptions:
- Maintain PR5 behavior by default; all new features are opt-in via config.
- Reuse existing abstractions (`BaseTrainer`, `FinetuneStrategy`, `DeviceManager`, `utils/*`).
- Avoid re-implementing features that already exist (e.g., use `token_dropout` instead of adding a duplicate stem dropout path).

Milestone 1 — Mixed precision (AMP) toggle (minimal)
- Scope:
  - Add stage-level flags: `use_mixed_precision: bool`, `amp_dtype: "bf16"|"fp16"` under `pretrain_config`/`finetune_config`.
  - Wrap forward/backward in autocast; use GradScaler only for fp16; bf16 uses autocast only.
  - Plumb flags from config through `run.py` into trainers; validate in `utils/config.py`.
- Modules:
  - `src/train/base.py`: add optional autocast + scaler in training/validation loops (gated by flags).
  - `src/run.py`: pass AMP flags to trainers (no control flow changes).
  - `src/utils/config.py`: add `validate_amp_config(stage_cfg)` within `validate_training_config`.
- Acceptance:
  - With AMP off (default), behavior identical to PR5.
  - With AMP on, one epoch completes on CUDA without exceptions; losses finite; optimizer steps occur.
- Tests:
  - Unit: dry-run tiny epoch for both bf16 and fp16.
  - Integration: ensure e2e full pipeline runs with AMP off (default) unchanged.

Milestone 2 — Optimizer configuration block (AdamW only)
- Scope:
  - Add optional `optimizer` block per stage: `{ type: "adamw", beta1, beta2, eps, weight_decay, exclude_from_weight_decay: ["LayerNorm", "bias"] }`.
  - Reuse existing AdamW defaults when block is absent; only override provided fields.
- Modules:
  - `src/train/train_v3.py`: honor `pretrain_config.optimizer` when building AdamW.
  - `src/finetune/strategies.py`: allow overrides from `finetune_config.optimizer` when constructing optimizer.
  - `src/utils/config.py`: `validate_optimizer_config` called from `validate_training_config`.
- Acceptance:
  - Defaults unchanged without optimizer block.
  - With overrides, optimizer is constructed with specified hyperparameters; exclusions applied via param group filter.
- Tests:
  - Unit: parse and apply overrides; verify param group counts and hyperparameters.

Milestone 3 — Advanced regularization (safe, low-effort toggles only)
- Scope:
  - Implement config knobs that map directly to existing dropout sites:
    - `mlp_dropout` → wire to `MLPHead` dropout.
    - `cross_attention_dropout` → wire to `InteractionCrossAttention` attentions/ffn dropout.
  - Alias `stem_dropout` to existing `token_dropout` to avoid duplicate paths.
  - Defer complex items: `stochastic_depth`, head gates, group lasso, entropy to PR7.
- Modules:
  - `src/model/v3.py`: read new keys from `regularization` and pass through to existing components; treat `stem_dropout` as alias.
  - `src/utils/config.py`: extend `validate_model_config` for new keys and alias rule.
- Acceptance:
  - Setting each knob changes only the corresponding dropout rates; shapes and training remain stable.
- Tests:
  - Unit: instantiate model with each knob; single forward pass succeeds; dropout modules reflect configured p.

Milestone 4 — EMA/SWA (opt-in, minimal hooks)
- Scope:
  - Add simple EMA of weights; optionally SWA epoch window. Disabled by default.
  - Save EMA/SWA state in checkpoints; restore on resume.
- Modules:
  - `src/utils/ema.py` (new, tiny): `EMAModel` with update/apply/restore.
  - `src/train/base.py`: lifecycle hooks to update EMA at `on_batch_end` and swap weights before checkpoint if enabled.
  - `src/utils/config.py`: validate `{ use_ema: bool, ema_decay: float }` and `{ use_swa: bool, swa_start_epoch, swa_lr, swa_anneal_epochs }` (finetune only if desired).
- Acceptance:
  - Enabling EMA produces checkpoints; disabling restores PR5 behavior.
- Tests:
  - Unit: EMA updates numerically; checkpoint save/restore round-trip.

Milestone 5 — Multi-GPU (DDP) support (single-node, e.g., 4×A40)
- Scope:
  - Add single-node Distributed Data Parallel with `torch.distributed` and `torchrun` launcher.
  - Use `DistributedSampler` for train/val; rank-aware logging and checkpointing (rank0-only writes).
  - Initialize/finalize process group in `DeviceManager`; wrap model via `DistributedDataParallel`.
  - Keep AMP/EMA/SWA compatibility: unwrap `.module` where needed; `no_grad()` for swaps.
  - Config-driven knobs under `top_level_config.ddp`; disabled by default; CPU/GPU-1 unchanged.
- Modules:
  - `src/utils/device.py`: `init_process_group`, `cleanup_process_group`, `get_rank/world_size/is_primary`.
  - `src/train/base.py`: call `DeviceManager.wrap_model()` → DDP when enabled; handle `module` access safely.
  - `src/utils/checkpoint.py`: guard writes with `is_primary()`; add `barrier()` where appropriate.
  - `src/run.py`: respect `top_level_config.ddp` and environment (`torchrun`); set seeds with rank offset.
  - `src/utils/config.py`: validate `top_level_config.ddp` block.
- Acceptance:
  - With DDP off (default), behavior matches PR5/PR6 results.
  - With DDP on (e.g., 4×A40), one tiny epoch completes with synced gradients; only rank0 saves.
  - Functional parity with EMA/SWA: training runs; checkpoints valid; e2e smoke passes on CUDA.
- Tests:
  - Unit: mock `torch.distributed` to assert rank-aware logging/checkpointing gates.
  - Integration (optional, CUDA): run tiny epoch under `torchrun --nproc_per_node=2` smoke job.

Milestone 6 — Logging enhancements (tiny)
- Scope:
  - Optionally log `l1_reg` contribution when `l1_lambda>0`.
  - Keep attention entropy and plots out of scope.
- Modules:
  - `src/train/train_v3.py` and `src/finetune/finetune_v3.py`: when computing L1, add `l1_reg` to batch/epoch logs if `logging_config.log_regularization_losses`.
  - `src/utils/config.py`: extend `validate_logging_config` to include `log_regularization_losses: bool`.
- Acceptance:
  - When enabled, CSV rows include `l1_reg` without impacting training.
- Tests:
  - Unit: verify presence/absence of column based on flag.

Milestone 7 — Docs and config updates
- Scope:
  - Update signatures in `README.md` and `instructions.md` for AMP, optimizer block, advanced dropout aliases, EMA/SWA.
  - Add commented examples to `configs/v3.yaml` for new knobs; keep defaults off.
- Modules:
  - `README.md`, `instructions.md`, `configs/v3.yaml`.
- Acceptance:
  - Docs concise; configs load; CI green.

Milestone 8 — Tests and CI hardening
- Scope:
  - Add unit tests per above; extend e2e smoke where needed.
  - Keep runtime fast; mock external I/O; maintain 90%+ coverage on core logic.
- Modules:
  - `tests/unit/*`, `tests/integration/*`.
- Acceptance:
  - All tests pass locally and in CI; existing behavior unchanged with new knobs off.

Non-goals (defer to PR7)
- Full DDP (spawn/init, gradient sync), stochastic depth, head gates, group lasso, attention diagnostics/plots.

# PR7 plan (crisp, implementation-ready, reuse-first, no repeat implementation)

- Milestone 0 — Config surfacing and plumbing (scaffold only)
  - Scope: `configs/v3.yaml` (uncomment/activate keys), `src/train/train_v3.py` (read flags), `src/finetune/finetune_v3.py` (propagate flags), `src/model/factory.py` (pass-through).
  - Config keys: `stochastic_depth`, `use_head_gates`, `head_gate_lambda`, `group_lasso_lambda`, `entropy_lambda`, `regularization_warmup_epochs`, `warmup_duration_epochs`; logging toggles under `logging.*`.
  - Reuse: existing utils modules including config loader, model factory, loss aggregation hooks, logging facade.
  - Done when: flags parse, appear in trainer/model init, all default-off, no behavior change.

- Milestone 1 — Stochastic depth in `model/v3.py` (minimal, opt-in)
  - Scope: `src/model/v3.py` only. Add per-block drop-path using existing dropout utilities if present; otherwise a lightweight module colocated in v3.
  - Behavior: survival prob schedule determined by `stochastic_depth` scalar; train-only; deterministic with seeds.
  - Reuse: current block/encoder stack; no rewire of attention/MLP; no new global utils.
  - Done when: unit tests validate forward equivalence at 0.0, non-zero affects outputs only in train, DDP-safe.

- Milestone 2 — Head gates (parameterized per attention head)
  - Scope: `src/model/v3.py` attention module(s). Add learnable gate per head applied to head outputs before projection.
  - Config: `use_head_gates` toggles creation/use; gates initialized to 1.0; respects AMP/gradient checkpointing.
  - Reuse: existing attention code path; no new attention types; no separate module or registry entry.
  - Done when: gates present and no-op when disabled; numerically stable with AMP; serialization works.

- Milestone 3 — Regularization losses (group lasso, entropy) with warmup
  - Scope: `src/train/train_v3.py` loss composition; optionally small helper in `src/utils/metrics.py` for entropy computation if exists, else local helper in train_v3.
  - Losses: apply `group_lasso_lambda` to head gates grouped by head; apply `entropy_lambda` to encourage sparse/confident gates; schedule via `regularization_warmup_epochs` and `warmup_duration_epochs`.
  - Reuse: existing loss aggregation and logging hooks; no new optimizer or schedulers.
  - Done when: losses toggle via config, scale with schedule, propagate in DDP, covered by unit tests for scaling/on-off.

- Milestone 4 — Advanced logging toggles
  - Scope: `src/utils/logging.py` (or current logging facade), `src/train/train_v3.py`.
  - Logging: `log_regularization_losses`, `log_attention_entropy`, `log_head_gate_values`, `log_ema_metrics`, `log_swa_metrics`, `save_plots` (default off).
  - Reuse: existing logger, writer, and artifact saving; for plots, reuse current plotting utilities; if none, text summaries only in this milestone.
  - Done when: metrics log conditionally without overhead when off; ranks>0 skip heavy ops under DDP.

- Milestone 5 — Diagnostics and plots (attention/head importance)
  - Scope: `src/evaluate/base.py` and eval entrypoints; optional helper in `src/utils/plots.py` if it already exists; otherwise keep inside evaluate path.
  - Features: per-layer head entropy histogram, gate value distribution, optional attention map snapshots on a small eval batch when `save_plots: true`.
  - Reuse: existing evaluation loop, dataset sampling, artifact storage layout.
  - Done when: artifacts saved under run dir, gated by config, CPU/VRAM bounded on small batch.

- Milestone 6 — DDP/AMP correctness and performance pass
  - Scope: `src/train/train_v3.py`, `src/finetune/finetune_v3.py`.
  - Tasks: ensure aux losses are reduced correctly, logging is rank-0 only, stochastic depth path is identical across ranks for the same seed; no extra syncs.
  - Reuse: existing DDP wrappers and AMP context managers.
  - Done when: multi-GPU smoke test passes; no hang; throughput regression <3%.

- Milestone 7 — Documentation and CI updates
  - Scope: `README.md` (config table entries), `instructions.md` (signatures/spec for new knobs), `docs/` short notes; `pytest` tests for new paths.
  - Tests: unit tests for gating tensors, loss scaling schedules, logging toggles; integration test with all toggles on for a tiny batch.
  - Reuse: existing test fixtures, tiny configs under `configs/e2e_test.yaml`.
  - Done when: ruff passes, tests green locally; minimal doc changes committed.

- Milestone 8 — Default-off config update and examples
  - Scope: `configs/v3.yaml` and `configs/e2e_test.yaml`.
  - Tasks: keep new features default-off in main; add an example commented “PR7 features” block showing a safe combo; tiny e2e variant that exercises all toggles for CI.
  - Reuse: current config structure and comments style.
  - Done when: loading both configs works; e2e variant runs in <2 minutes.

Assumptions
- EMA/SWA implementations already exist; PR7 only adds conditional logging for them.
- No new model version file; all changes stay within `model/v3.py` and current trainers.
- No new dependencies; use current stack.

Acceptance gating per milestone
- Code touches are minimal and scoped to the listed modules only.
- New behavior is strictly behind config flags; default-off preserves PR6 behavior.
- Tests and linting added alongside each milestone; no repeated implementations across modules.

Short summary
- Eight minimal milestones to introduce stochastic depth, head gates, regularization with warmup, and advanced diagnostics/logging—scoped to `model/v3.py`, current trainers, and logging/eval utilities—fully gated by config and reusing existing infrastructure.