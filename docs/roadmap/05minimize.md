## What TUnA Does

```15:33:TUnA/results/bernett/TUnA/inference.py
def initialize_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
    logging.info("Epoch Time              Train Loss          Test Loss           AUC                 PRC                 Accuracy            Sensitivity         Specificity         Precision           F1                  MCC                 Max AUC")
```

```229:236:TUnA/results/bernett/TUnA/utils.py
def log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev):
    metrics = [epoch, time, total_loss_train/total_train_size, total_loss_test/total_test_size, AUC_dev, PRC_dev,accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev]
    logging.info('\t'.join(map(str, metrics)))
```

Simple TSV with a fixed header, appended each epoch, and later parsed for plots.

## Minimal Compatibility Plan

### Task Summary
Mimic TUnA's simple per-epoch train→validate→log loop and metrics, and remove EMA/SWA to keep MVP minimal.

### What we will mirror from TUnA

1) Metrics function (to replicate downstream plots/logic):

```216:227:TUnA/results/bernett/TUnA/utils.py
def calculate_metrics(T, Y, S):
    AUC_dev = roc_auc_score(T, S)
    tpr, fpr, _ = precision_recall_curve(T, S)
    PRC_dev = auc(fpr, tpr)
    accuracy = accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc
```

2) Per-epoch loop (train once, validate once, then log), with epoch timing:

```104:121:TUnA/results/bernett/TUnA/utils.py
def train_and_validate_model(config, trainer, tester, scheduler, model, device):
    max_AUC_dev = 0
    train_dictionary = load_dictionary(config['directories']['train_dictionary'])
    test_dictionary = load_dictionary(config['directories']['validation_dictionary'])

    train_interactions = config['directories']['train_interactions']
    test_interactions = config['directories']['validation_interactions']
    start = timeit.default_timer()
    subset = config['training']['subset']
    if subset < 0:
        subset = None

    for epoch in range(1, config['training']['iteration'] + 1):
        if epoch != (config['training']['iteration']):
            total_loss_train, total_train_size = train_epoch(...)
            T, Y, S, total_loss_test, total_test_size = test_epoch(...)
```

And the TSV logging step:

```229:236:TUnA/results/bernett/TUnA/utils.py
def log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev):
    metrics = [epoch, time, total_loss_train/total_train_size, total_loss_test/total_test_size, AUC_dev, PRC_dev,accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev]
    logging.info('\t'.join(map(str, metrics)))
```

### DIPPI Minimal changes (plan)
- utils/metrics.py: add `calculate_metrics(y_true, y_pred, y_score)` that returns the same 8 values as TUnA (AUROC, AUPRC, accuracy, sensitivity, specificity, precision, f1, MCC). Implement via our sklearn backend to avoid duplicating logic.
- train/base.py (BaseTrainer):
  - For each epoch: train epoch → run evaluator once on `val_loader` → collect train loss/metrics and val loss/metrics.
  - Track epoch start/end time; compute per-epoch elapsed time and use it in the CSV as the second column.
  - Write one CSV row per epoch with strict header:
    - Epoch, Epoch Time, Train Loss, Val Loss, Train Metric 1, Val Metric 1, Train Metric 2, Val Metric 2, Learning Rate
  - The two metrics are resolved from `logging.training_step_metrics` (primary/secondary). If unset, default to `auc` and `recall`.
- Remove EMA/SWA runtime hooks (MVP): strip EMA/SWA from BaseTrainer (and ignore EMA/SWA branches in subclass logging), keeping early stopping/scheduler untouched.
- Configs: comment out `use_ema`, `ema_decay`, `use_swa`, `swa_*` keys in `configs/v3.yaml` and `configs/examples/full_config.yaml` to make the MVP config clean.

### CSV contract (training_step.csv)
- Column order (strict):
  - `Epoch`, `Epoch Time`, `Train Loss`, `Val Loss`, `Train Metric 1`, `Val Metric 1`, `Train Metric 2`, `Val Metric 2`, `Learning Rate`
- `Train Metric 1/2` and `Val Metric 1/2` map to `logging.training_step_metrics.primary/secondary` with aliases supported (e.g., `auc`→`auroc`, `recall`→`recall`).

### Acceptance checks
- Unit: BaseTrainer produces one CSV row per epoch with exact header and populated values; metrics align with evaluator outputs; the second column is epoch elapsed time (seconds).
- Unit: `calculate_metrics` returns 8 values matching TUnA for a small synthetic case.
- Integration: A 1-epoch smoke run writes `training_step.csv` with both metrics and LR, no EMA/SWA branches executed.
- Config: v3 configs load with EMA/SWA commented out; training runs without referencing EMA/SWA flags.

### Risks & notes
- Subclasses (`v3Pretrainer`, `v3Finetuner`) currently write CSV rows themselves. We will consolidate logging in `BaseTrainer` to prevent duplicate rows (and remove subclass CSV appends).
- Tests referencing the previous header will be updated to expect the epoch time as the second column and the two Train/Val metric columns.
- Epoch elapsed time replaces the prior Timestamp column in the CSV.

---

### Task-by-task plan

1) Remove EMA/SWA functionality
- Action: Remove EMA/SWA flags, state, and branches from `BaseTrainer`, `v3Finetuner`, and any related utilities; eliminate EMA/SWA checkpoints and validation branches.
- Scope: `src/train/base.py`, `src/finetune/finetune_v3.py`, `src/train/train_v3.py`, any `src/utils/ema.py` usage, and checkpoint helpers if they include ema/swa variants.

Remove the ema/swa from utils. clean up the useless utils functions and simplify the logic.

   **Actions & Scope**

   - Delete `src/utils/ema.py` and purge all EMA/SWA-related exports from `src/utils/__init__.py`.
   - Simplify checkpoint utilities: drop EMA/SWA flags from `src/utils/checkpoint.py`, remove `load_pretrain_ema_for_finetune` / `load_ema_state_for_resume`, and update docstrings plus return payloads accordingly.
   - Trim checkpoint path helpers: remove EMA/SWA branches in `src/utils/paths.py` and adjust tests that relied on those variants.
   - Collapse logging helpers to a single raw model flow within `src/utils/logging.py`, deleting EMA/SWA code paths.
   - Update orchestrator imports (`src/run.py`, `src/evaluate/base.py`) to match the reduced utils surface.
   - Excise orphaned tests covering EMA/SWA (`tests/unit/utils/test_ema.py`, EMA/SWA sections in utils/run/integration suites) and rewrite expectations for the simplified APIs.
   - Refresh developer docs (`instructions.md`, related roadmap entries) to reflect the absence of EMA/SWA utilities.

2) Comment out EMA/SWA in configs
- Action: Comment out `use_ema`, `ema_decay`, `use_swa`, `swa_*` keys in `configs/v3.yaml` and `configs/examples/full_config.yaml`. Ensure config validation tolerates these being absent/commented.
- Scope: `configs/v3.yaml`, `configs/examples/full_config.yaml`, `src/utils/config.py` (validation).

3) Implement TUnA-compatible metrics helper
- Action: Add `calculate_metrics(y_true, y_pred, y_score)` in `src/utils/metrics.py` returning (auroc, auprc, accuracy, sensitivity, specificity, precision, f1, mcc). Use `bin_classification_metrics` internally.
- Scope: `src/utils/metrics.py` only.

4) Centralize per-epoch train→validate in BaseTrainer
- Action: In `BaseTrainer.fit()` epoch loop, after training one epoch, call `Evaluator.evaluate` on `val_loader`; collect val loss and requested metrics.
- Scope: `src/train/base.py` only (no subclass CSV writes).
   - Task 4.1: Collapse `BaseTrainer` validation helpers into a single evaluator-driven flow (`fit` delegates to `Evaluator.evaluate` for every epoch; drop `validate_once`/`evaluate_model` duplicates).
     - Scope: `src/train/base.py`
   - Task 4.2: Refine evaluator API to accept logits/scores and surface the full 8-value tuple from `calculate_metrics`; ensure storage for later logging.
     - Scope: `src/evaluate/base.py`, `src/utils/metrics.py` (read-only dependency)
   - Task 4.3: Align epoch sequencing with TUnA (`train_epoch` → `test_epoch`) semantics; ensure trainer passes the right tensors to evaluator without embedding logging.
     - Scope: `src/train/base.py`, `src/evaluate/base.py`
   - Task 4.4: Remove redundant per-model overrides that re-implement validation; rely on the centralized evaluator flow everywhere.
     - Scope: `src/train/base.py`, `src/finetune/finetune_v3.py`, `src/train/train_v3.py`
   - Task 4.5: Drop multi-model evaluation entrypoints; consolidate on the single-model `run_evaluation` path.
     - Scope: `src/evaluate/base.py`, `src/evaluate/__init__.py`, `src/run.py`
   - Task 4.6: Strip plotting/diagnostic helpers from evaluation (no matplotlib/attention dumps) so evaluation stays text-only.
     - Scope: `src/evaluate/base.py`, `src/utils/logging.py` (if touched), dependent callers

5) Add epoch timing and update CSV schema
- Action: Measure epoch elapsed time; delegate header/row assembly to `src/utils/logging.log_training_epoch`, using canonical column order `Epoch, Epoch Time, Train Loss, Val Loss, Train Metric 1, Val Metric 1, Train Metric 2, Val Metric 2, Learning Rate`.
- Scope: `src/train/base.py`, `src/utils/logging.py`; ensure all trainers call the shared helper (no per-subclass overrides).

6) Centralize metric normalization utilities, default training-step metrics, and ensure eval CSV deduplicates aliases.

**Files**:  
- `src/utils/metrics.py`  
- `src/train/base.py`  
- `src/utils/logging.py`  
- `src/evaluate/base.py`  
- `tests/integration/test_real_e2e.py`

**Plan**:  
- Extend `src/utils/metrics.py` with shared alias map plus helpers (`normalize_metric_key`, `coerce_scalar`, `resolve_training_step_metric_names`) and export them.  
- Update `src/train/base.py` to import the shared helpers, reuse them for default metric resolution, and drop the local duplicates.  
- Refactor `src/utils/logging.py` to reuse `coerce_scalar` and ensure training CSV headers stay limited to the canonical schema.  
- Adjust `src/evaluate/base.py` to consume shared normalization, collapse alias keys before emitting metrics, and keep the default metric list canonical for CSV output.  
- Fix `tests/integration/test_real_e2e.py` to import the relocated helper and align expectations.

7) DDP official pytorch guide(@https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html). Align our single-node DDP setup with the official PyTorch tutorial for clearer initialization, sampling, and launch flow.

**Files**: src/utils/distributed.py, src/train/base.py, src/train/train_v3.py, scripts/v3.sh

**Plan**:
- src/utils/distributed.py: Replace the custom helper stack with the tutorial-style setup_ddp()/cleanup_ddp() that call dist.init_process_group("nccl", init_method="env://"), set torch.cuda.set_device(local_rank), and expose thin wrappers for rank, world_size, and is_primary. Retire defensive fallbacks that diverge from the guide.
- src/train/base.py: Require the new helpers during trainer bootstrap, enforce DistributedSampler usage with sampler.set_epoch(epoch) per the guide, and gate logging/checkpointing on is_primary() while keeping gradient sync behavior unchanged.
- src/train/train_v3.py: Wrap models with DistributedDataParallel(model, device_ids=[local_rank]), build samplers/loaders exactly once, and ensure evaluation paths reuse the rank-0 gather per the simplified helpers.
- scripts/v3.sh: Trim the launcher to the documented pattern (torchrun --standalone --nproc_per_node=$AVAILABLE_GPUS -m src.run ...), exporting only the required MASTER_ADDR/PORT and removing redundant environment plumbing so the shell aligns with the Python entrypoint.

8) Remove duplicate evaluation helper from src/run.py and simplify the entrypoint to mirror the concise workflow style of TUnA’s main.py.

**Files**:
- src/run.py
- src/evaluate/base.py
- src/pipeline/workflows.py (new, unless you prefer a different location)

**Plan**:
- Audit _evaluate_single_checkpoint against utilities in src/evaluate/base.py; design a single reusable helper (likely evaluate_checkpoint) so run logic can import it instead of maintaining a local copy.
- Extract the heavy mode functions (run_full_pipeline, run_eval_only, etc.) into src/pipeline/workflows.py, keeping their signatures intact and adding re-exports if callers still import from src.run.
- Refactor src/run.py to just: load config, resolve mode, dispatch via the extracted workflow module, and report success/failure—matching TUnA’s clean main structure.
- Wire the new evaluation helper into the extracted workflows so evaluation behavior stays the same while avoiding duplication.
- Adjust imports/tests/documentation stubs as needed so everything references the new module layout without regressions.

9) Update tests and smoke checks
- Action: Update tests expecting `training_step.csv` header to match the new schema (Epoch Time instead of Timestamp, plus Train/Val metric columns). Add a 1-epoch smoke to assert one row is written and Evaluator is called.
- Scope: `tests/unit`, `tests/integration` (minimal edits to align headers and defaults).