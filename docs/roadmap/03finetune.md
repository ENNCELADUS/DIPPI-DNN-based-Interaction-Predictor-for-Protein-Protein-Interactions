# Finetune Implementation Roadmap: EMA & Multi-Model Evaluation

## 🎯 Objectives

1. **Load pretrain EMA checkpoint** to continue EMA tracking in finetune
2. **Evaluate both raw and EMA models** during validation, log both metrics
3. **Track best model** considering both raw and EMA performance (aligned with pretrain logic)
4. **Final test evaluation** on raw, EMA, and SWA checkpoints with separate CSV outputs
5. **Support resume_finetune** with EMA state restoration

---

## 📊 Current State Analysis

### Pretrain EMA Logic (Baseline)
From `src/train/train_v3.py:on_validation_end()`:
- **Evaluates**: Raw model only (lines 510-546)
- **Best model tracking**: Based on raw model metrics (monitor_metric: val_auc or val_loss)
- **Checkpoint saving**: When `is_best=True`, saves BOTH:
  - `best_model.pth` (raw model)
  - `best_model_ema.pth` (EMA model, lines 625-645)
- **EMA not evaluated during training**: Only raw model performance determines "best"

### Finetune Current State
- **Evaluates**: Raw model only
- **Best model tracking**: Based on raw model metrics (val_loss by default)
- **Checkpoint saving**: Saves raw, EMA, and SWA checkpoints when available
- **Missing**: 
  - Does not load pretrain EMA checkpoint
  - Does not evaluate EMA during training
  - Final test evaluates current model state, not saved checkpoints

---

## 📋 Implementation Plan

### **Part 1: EMA Checkpoint Loading from Pretrain**

#### **1.1 Create Helper Function** (in `src/utils/checkpoint.py`)

**Function**: `load_pretrain_ema_for_finetune()`
- **Purpose**: Load pretrain EMA checkpoint to initialize finetune EMA
- **Location**: Add to `src/utils/checkpoint.py` exports and implement after `resolve_checkpoint()`
- **Rationale**: Checkpoint-related logic belongs in checkpoint utils (keeps core orchestration thin)
- **Parameters**:
  ```python
  def load_pretrain_ema_for_finetune(
      trainer: Any,
      pretrain_run_id: Optional[str],
      model_name: str = "v3",
      logger: Optional[logging.Logger] = None,
  ) -> None:
  ```
- **Logic**:
  ```
  1. Check if trainer.use_ema is True, otherwise return early
  2. Early return if pretrain_run_id is None (log info if logger available)
  3. Try to resolve pretrain EMA checkpoint path:
     - resolve_checkpoint(model_name, "pretrain", pretrain_run_id, best=True, ema=True)
  4. If not found (FileNotFoundError/ValueError), log warning and return gracefully
  5. Load checkpoint with torch.load(map_location="cpu", weights_only=False)
  6. Extract EMA state_dict from checkpoint (check "model_state_dict" then "model_state")
  7. Validate state is dict-like; warn and return if invalid
  8. Store in trainer.ema_pending_state
  9. Log success: "Loaded pretrain EMA checkpoint for finetune continuation"
  ```
- **Error Handling**: All graceful degradation with logging (no exceptions raised)

#### **1.2 Modify `run_finetune_from_pretrain()`** (line 468-569)

**Changes at line ~548** (after building finetuner, before fit()):
```python
# Import at top of file
from src.utils.checkpoint import load_pretrain_ema_for_finetune

# After finetuner is created (line 532-548)
# Before finetuner.fit() (line 549)

# Load pretrain EMA checkpoint if finetune uses EMA
if finetune_cfg.get("use_ema", False):
    load_pretrain_ema_for_finetune(
        trainer=finetuner,
        pretrain_run_id=cfg["run"].get("pretrain_run_id"),
        model_name="v3",
        logger=finetune_logger,
    )

finetuner.fit()  # Existing call
```

#### **1.3 Verify `BaseTrainer` EMA Initialization** ✅

**Check `src/train/base.py`** (around line 158-161):
- `self.ema_pending_state` already exists ✅
- EMA initialization in `_setup_training()` properly loads pending state (lines 256-259) ✅
- **No changes needed** - infrastructure already correct

---

### **Part 2: EMA Validation During Training** ✅ IMPLEMENTED

#### **2.1 Modify `v3Finetuner.on_validation_end()`** (line 480-853 in `finetune_v3.py`)

**Design: Dual Independent Best Model Tracking**:
- Track `best_metric` (raw) and `best_metric_ema` (EMA) **separately**
- Save `best_model.pth` when raw improves
- Save `best_model_ema.pth` when EMA improves  
- Both models saved independently in each validation cycle
- Aligns with pretrain stage checkpoint structure
- Early stopping remains raw-only (no EMA consideration)

**Implementation Steps**:

**Step 0: Monitor Metric Validation at Init** (~line 260 in `__init__`)
- Build canonical metrics list: `["val_auc", "auroc", "val_loss", "train_loss", "ap", "accuracy@0.5"]`
- Validate `self.monitor_metric` against allowed list
- If invalid: log warning, fallback to `"val_auc"`, continue (fail gracefully)

**Step 1: Add EMA Tracking State Variables** (~line 290 in `__init__`)
```python
# Dual best model tracking (independent raw vs EMA)
self.best_metric_ema: Optional[float] = None
self.best_epoch_ema: Optional[int] = None  
self.best_metric_name_ema: Optional[str] = None
```

**Step 2: Add EMA Evaluation** (after line 502 in `on_validation_end`)
```python
# Store raw validation metrics
raw_validation_metrics = validation_metrics.copy()

# Evaluate EMA model if enabled
ema_validation_metrics = {}
ema_monitor_value = None

if getattr(self, 'use_ema', False) and getattr(self, 'ema', None) is not None:
    try:
        # Unwrap model and apply EMA weights (using custom EMAModel from utils/ema.py)
        model_ref = self.device_manager.unwrap_model(self.model)
        self.ema.apply_to(model_ref)
        
        # Run evaluator with EMA-weighted model
        ema_eval_metrics = self.evaluator.evaluate(
            model=self.model,  # Model now has EMA weights applied
            dataloader=self.val_loader,
            metrics=["accuracy@0.5", "auroc", "ap", "loss"],
            device_manager=self.device_manager,
            criterion=self.criterion,
        )
        
        # Restore original weights
        self.ema.restore(model_ref)
        
        # Add _ema suffix to all metrics
        ema_validation_metrics = {f"{k}_ema": v for k, v in ema_eval_metrics.items()}
        
        # Extract EMA monitor metric for comparison
        if self.monitor_metric == "val_auc" and "auroc" in ema_eval_metrics:
            ema_monitor_value = ema_eval_metrics["auroc"]
        elif self.monitor_metric == "val_loss" and "loss" in ema_eval_metrics:
            ema_monitor_value = ema_eval_metrics["loss"]
        elif self.monitor_metric in ema_eval_metrics:
            ema_monitor_value = ema_eval_metrics[self.monitor_metric]
        
        self.logger.info(f"EMA evaluation metrics: {ema_eval_metrics}")
        
    except Exception as e:
        self.logger.warning(f"EMA evaluation failed: {e}")
```

**Step 3: Dual Independent Best Model Selection** (replace lines 577-593)
```python
# Raw model best tracking (existing logic, keep unchanged)
is_best_raw = False
if current_metric_value is not None:
    if self.best_metric is None:
        is_best_raw = True
    elif self.monitor_metric in ["val_auc", "auroc", "accuracy@0.5", "ap"]:
        is_best_raw = current_metric_value > self.best_metric
    else:
        is_best_raw = current_metric_value < self.best_metric

if is_best_raw:
    self.best_metric = current_metric_value
    self.best_metric_name = current_metric_name
    self.best_epoch = self.current_epoch

# EMA model best tracking (new, independent)
is_best_ema = False
if ema_monitor_value is not None:
    if self.best_metric_ema is None:
        is_best_ema = True
    elif self.monitor_metric in ["val_auc", "auroc", "accuracy@0.5", "ap"]:
        is_best_ema = ema_monitor_value > self.best_metric_ema
    else:
        is_best_ema = ema_monitor_value < self.best_metric_ema

if is_best_ema:
    self.best_metric_ema = ema_monitor_value
    self.best_metric_name_ema = current_metric_name  # Same metric type as raw
    self.best_epoch_ema = self.current_epoch
```

**Step 4: Update Checkpoint Saving Logic** (lines 596-682)
```python
# Save raw best model when is_best_raw=True (existing logic, keep lines 596-617)
if is_best_raw:
    save_checkpoint(..., best=True, model_name=self.model_name, ...)
    # Produces: best_model.pth

# Save EMA best model when is_best_ema=True (NEW, independent)
if is_best_ema and self.use_ema and self.ema is not None:
    save_checkpoint(
        ..., 
        best=True, 
        ema=True, 
        ema_state_dict=self.ema.state_dict(),
        ...
    )
    # Produces: best_model_ema.pth

# Cycle checkpoints (last) unchanged (lines 642-684)
# When NOT save_best_only, saves both checkpoint_epoch_N.pth and checkpoint_epoch_N_ema.pth
```

**Step 5: Update CSV Logging** (lines 738-747)
```python
training_row = {
    "epoch": self.current_epoch,
    "is_best": is_best_raw,              # Raw model improved flag
    "is_best_ema": is_best_ema,          # EMA model improved flag (NEW)
    "best_metric": self.best_metric,
    "best_metric_name": self.best_metric_name,
    "best_epoch": self.best_epoch,       # NEW: track which epoch raw was best
    "best_metric_ema": self.best_metric_ema,      # NEW
    "best_metric_name_ema": self.best_metric_name_ema,  # NEW
    "best_epoch_ema": self.best_epoch_ema,        # NEW
    **raw_validation_metrics,      # auroc, ap, loss, accuracy@0.5, ...
    **ema_validation_metrics,      # auroc_ema, ap_ema, loss_ema, ...
}

# Include l1_reg if enabled (existing code)
if avg_l1_reg is not None:
    training_row["l1_reg"] = avg_l1_reg

# Append to CSV (existing call)
csv_path = append_training_row(
    self.model_name, "finetune", self.run_id, training_row
)
```

**Logging Behavior**:
- When `is_best_raw=True`: Log "Saved best model checkpoint: ..."
- When `is_best_ema=True`: Log "Saved best EMA checkpoint: ..." (separate line)
- Both can be True in same epoch → two separate log lines

**Notes**:
- CSV backward compatible: old runs without EMA columns will work fine ✅
- `append_training_row` uses `DictWriter` which handles dynamic columns ✅
- Early stopping remains raw-only (lines 537-576 unchanged) ✅
- EMA unwrapping uses `EMAModel` from `src/utils/ema.py` (apply_to/restore pattern) ✅

---

### **Part 3: Multi-Model Final Test Evaluation**

**Overview**: Automatically evaluate raw, EMA, and SWA models on all test sets after training completes (or when early stopping triggers).

**Design Decisions**:
1. **Evaluator instantiation**: Create `Evaluator()` inside function by default; accept optional `evaluator` param for test injection
2. **Model instantiation**: Extract config from `cfg` dict and use `ModelFactory.from_config()`; single model instance reused for all checkpoints
3. **Backward compatibility**: Keep `run_evaluation()` as thin wrapper (can deprecate later)
4. **Diagnostics**: Run on ALL test sets with unique `run_id` per model type + test set combination
5. **Test loader discovery**: Automatically evaluate all test sets defined in `data_config.evaluate` section (e.g., `test_balanced`, `test_realistic`)

**CSV Output Structure**:
- Separate file per model type: `v3_finetune_{run_id}_eval_{model_type}.csv`
- Example: `v3_finetune_20240815_143022_eval_raw.csv`, `v3_finetune_20240815_143022_eval_ema.csv`
- Format: `test_set,auroc,auprc,ap,accuracy@0.5,loss,...`

---

#### **3.1 Create New Function** (in `src/run.py`)

**Function**: `run_multi_model_evaluation()`
- **Purpose**: Evaluate raw, EMA, and SWA models on all test sets defined in config
- **Location**: Before `run_evaluation()` 
- **Signature**:
  ```python
  def run_multi_model_evaluation(
      model_name: str,
      stage: str,
      run_id: str,
      dataloaders: Dict[str, DataLoader],
      device_manager: DeviceManager,
      evaluate_config: Dict[str, Any],
      cfg: Dict[str, Any],  # Full config for model creation
      logging_config: Optional[Dict[str, Any]] = None,
      evaluator: Optional[Evaluator] = None,  # For test injection
  ) -> None:
  ```

**Detailed Logic**:
```
1. Resolve available checkpoints:
   - raw: best_model.pth (REQUIRED - fail if missing)
   - ema: best_model_ema.pth (OPTIONAL - skip with warning if missing)
   - swa: best_model_swa.pth (OPTIONAL - skip with warning if missing)
   Use resolve_checkpoint() for structured path resolution

2. Identify test loaders from dataloaders dict:
   - Look for keys starting with "eval_"
   - Typically: eval_balanced, eval_realistic (from config data_config.evaluate)
   - If only one test file in config, loader named "eval"
   - Automatically evaluates ALL test sets defined in config

3. Create model instance ONCE (reuse for all checkpoints):
   - model = ModelFactory.from_config(cfg, device=device, logger=minimal_logger)
   
4. Create evaluator instance (or use injected one for tests):
   - evaluator = evaluator or Evaluator()

5. For each checkpoint type (raw, ema, swa):
   IF checkpoint exists:
     a. Load checkpoint into SAME model instance via load_checkpoint()
     b. Set model to eval mode
     c. For each test set (ALL test sets in config):
        - Run Evaluator.evaluate()
        - Store: results[model_type][test_set_name] = metrics
        - Print progress: "{model_type.upper()} on {test_name}: {metrics}"
     d. Free GPU memory if needed
   
6. Run diagnostics on ALL test sets if save_plots=True:
   - For each model type (raw, ema, swa):
     - For each test set:
       - Call evaluate_with_diagnostics() with unique run_id suffix
       - Use run_id=f"{run_id}_{model_type}_{test_set_name}" for isolation
       - Avoid filename collisions between raw/ema/swa and different test sets

7. Write results to separate CSV files per model type:
   - append_test_results_by_model(model_type="raw", test_results={...})
   - append_test_results_by_model(model_type="ema", test_results={...})
   - append_test_results_by_model(model_type="swa", test_results={...})
```

**Pseudocode**:
```python
def run_multi_model_evaluation(
    model_name: str,
    stage: str,
    run_id: str,
    dataloaders: Dict[str, DataLoader],
    device_manager: DeviceManager,
    evaluate_config: Dict[str, Any],
    cfg: Dict[str, Any],
    logging_config: Optional[Dict[str, Any]] = None,
    evaluator: Optional[Evaluator] = None,
) -> None:
    # Step 1: Resolve checkpoints
    checkpoints = {}
    try:
        checkpoints["raw"] = resolve_checkpoint(model_name, stage, run_id, best=True)
    except FileNotFoundError:
        raise ValueError("Raw checkpoint required but not found")
    
    for ckpt_type in ["ema", "swa"]:
        try:
            checkpoints[ckpt_type] = resolve_checkpoint(
                model_name, stage, run_id, best=True, 
                ema=(ckpt_type=="ema"), swa=(ckpt_type=="swa")
            )
        except FileNotFoundError:
            print(f"{ckpt_type.upper()} checkpoint not found, skipping")
    
    # Step 2: Identify test loaders (from config evaluate section)
    test_loaders = {}
    if "eval" in dataloaders:
        # Single test file case
        test_loaders["eval"] = dataloaders["eval"]
    else:
        # Multiple test files (eval_balanced, eval_realistic, etc.)
        test_loaders = {k: v for k, v in dataloaders.items() if k.startswith("eval_")}
    
    if not test_loaders:
        raise ValueError("No evaluation loaders found in dataloaders dict")
    
    # Step 3: Create model instance once (reuse for all checkpoints)
    device = device_manager.select_device()
    minimal_logger = logging.getLogger("run_orchestrator")
    model = ModelFactory.from_config(cfg, device=device, logger=minimal_logger)
    
    # Step 4: Create evaluator instance (or use injected for tests)
    if evaluator is None:
        evaluator = Evaluator()
    
    # Step 5: Prepare evaluation configuration
    metrics_list = evaluate_config.get("metrics", ["accuracy@0.5", "auroc", "ap"])
    criterion = nn.BCEWithLogitsLoss() if "loss" in metrics_list else None
    
    all_results = {}  # {model_type: {test_set: {metric: value}}}
    
    # Step 6: Evaluate each checkpoint type
    for model_type, ckpt_path in checkpoints.items():
        print(f"\nEvaluating {model_type.upper()} model...")
        
        # Load checkpoint weights into same model instance
        load_checkpoint(model, checkpoint_path=str(ckpt_path), strict=True)
        
        all_results[model_type] = {}
        
        # Evaluate on ALL test sets
        for test_name, test_loader in test_loaders.items():
            metrics = evaluator.evaluate(
                model=model,
                dataloader=test_loader,
                metrics=metrics_list,
                device_manager=device_manager,
                criterion=criterion,
            )
            all_results[model_type][test_name] = metrics
            print(f"{model_type.upper()} on {test_name}: {metrics}")
        
        # Step 7: Diagnostic evaluation on ALL test sets (if enabled)
        if logging_config and logging_config.get("save_plots", False):
            max_samples = logging_config.get("max_diagnostic_samples", 16)
            
            for test_name, test_loader in test_loaders.items():
                try:
                    # Unique run_id per model type AND test set
                    diagnostic_run_id = f"{run_id}_{model_type}_{test_name}"
                    
                    diagnostic_results = evaluator.evaluate_with_diagnostics(
                        model=model,
                        dataloader=test_loader,
                        device_manager=device_manager,
                        model_name=model_name,
                        stage=stage,
                        run_id=diagnostic_run_id,
                        max_diagnostic_samples=max_samples,
                        save_plots=True,
                    )
                    print(f"{model_type.upper()} diagnostics on {test_name} saved: {diagnostic_results['diagnostics_saved']}")
                except Exception as e:
                    print(f"Warning: {model_type.upper()} diagnostics on {test_name} failed: {e}")
    
    # Step 8: Write results to CSV files (one per model type)
    for model_type, test_results in all_results.items():
        csv_path = append_test_results_by_model(
            model_name=model_name,
            stage=stage,
            run_id=run_id,
            model_type=model_type,
            test_results=test_results,
        )
        print(f"{model_type.upper()} results saved: {csv_path}")
```

#### **3.2 Create Helper for CSV Writing** (in `src/utils/logging.py`)

**Function**: `append_test_results_by_model()`
- **Purpose**: Write test results for a specific model type to dedicated CSV
- **Location**: After `log_swa_metrics()` (around line 660)
- **Signature**:
  ```python
  def append_test_results_by_model(
      model_name: str,
      stage: str,
      run_id: str,
      model_type: str,  # "raw", "ema", or "swa"
      test_results: Dict[str, Dict[str, float]],  # {test_set: {metric: value}}
  ) -> Path:
  ```

**Implementation**:
```python
def append_test_results_by_model(
    model_name: str,
    stage: str,
    run_id: str,
    model_type: str,
    test_results: Dict[str, Dict[str, float]],
) -> Path:
    """
    Write test evaluation results to model-type-specific CSV file.
    
    Creates separate CSV files for raw, EMA, and SWA model evaluations
    to avoid confusion and enable easy comparison.
    
    CSV Format:
        test_set,auroc,auprc,ap,accuracy@0.5,loss,...
        test_balanced,0.85,0.82,0.80,0.78,1.12,...
        test_realistic,0.75,0.70,0.68,0.65,0.16,...
    
    Args:
        model_name: Model identifier (v3)
        stage: Stage context (finetune, pretrain)
        run_id: Run identifier
        model_type: "raw", "ema", or "swa"
        test_results: Nested dict {test_set_name: {metric: value}}
    
    Returns:
        Path to written CSV file
    """
    # Rank-0 only guard
    if not is_primary():
        log_paths_dict = log_paths(model_name, stage, run_id)
        log_dir = log_paths_dict["dir"]
        return log_dir / f"{model_name}_{stage}_{run_id}_eval_{model_type}.csv"
    
    # Construct file path
    log_paths_dict = log_paths(model_name, stage, run_id)
    log_dir = log_paths_dict["dir"]
    csv_path = log_dir / f"{model_name}_{stage}_{run_id}_eval_{model_type}.csv"
    
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all metric names for header
    all_metrics = set()
    for test_metrics in test_results.values():
        all_metrics.update(test_metrics.keys())
    
    fieldnames = ["test_set"] + sorted(all_metrics)
    
    # Write CSV
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for test_set_name, metrics in test_results.items():
            row = {"test_set": test_set_name, **metrics}
            writer.writerow(row)
    
    return csv_path
```

#### **3.3 Update Pipeline Mode Functions**

**A. `run_finetune_from_pretrain()`** (line 564-574):
```python
# REPLACE:
if "evaluate" in cfg:
    minimal_logger.info("Starting evaluation...")
    run_evaluation(
        model,
        dataloaders,
        device_manager,
        cfg["evaluate"],
        "finetune",
        finetune_run_id,
        cfg.get("logging_config"),
    )

# WITH:
if "evaluate" in cfg:
    minimal_logger.info("Starting multi-model evaluation...")
    run_multi_model_evaluation(
        model_name="v3",
        stage="finetune",
        run_id=finetune_run_id,
        dataloaders=dataloaders,
        device_manager=device_manager,
        evaluate_config=cfg["evaluate"],
        cfg=cfg,
        logging_config=cfg.get("logging_config"),
    )
```

**B. `run_full_pipeline()`** (line 449-459):
```python
# REPLACE:
if "evaluate" in cfg:
    minimal_logger.info("Starting evaluation...")
    run_evaluation(
        model,
        dataloaders,
        device_manager,
        cfg["evaluate"],
        "finetune",
        finetune_run_id,
        cfg.get("logging_config"),
    )

# WITH:
if "evaluate" in cfg:
    minimal_logger.info("Starting multi-model evaluation...")
    run_multi_model_evaluation(
        model_name="v3",
        stage="finetune",
        run_id=finetune_run_id,
        dataloaders=dataloaders,
        device_manager=device_manager,
        evaluate_config=cfg["evaluate"],
        cfg=cfg,
        logging_config=cfg.get("logging_config"),
    )
```

**C. `run_resume_finetune()`** (line 751-761):
```python
# REPLACE:
if "evaluate" in cfg:
    minimal_logger.info("Starting evaluation...")
    run_evaluation(
        model,
        dataloaders,
        device_manager,
        cfg["evaluate"],
        "finetune",
        finetune_run_id,
        cfg.get("logging_config"),
    )

# WITH:
if "evaluate" in cfg:
    minimal_logger.info("Starting multi-model evaluation...")
    run_multi_model_evaluation(
        model_name="v3",
        stage="finetune",
        run_id=finetune_run_id,
        dataloaders=dataloaders,
        device_manager=device_manager,
        evaluate_config=cfg["evaluate"],
        cfg=cfg,
        logging_config=cfg.get("logging_config"),
    )
```

**D. `run_eval_only()`** (line 803-817):
```python
# Keep existing run_evaluation() for backward compatibility
# This mode is for targeted evaluation of a specific checkpoint
# Users specify exact checkpoint path/type, not comprehensive comparison
# NO CHANGES NEEDED
```

#### **3.4 Keep `run_evaluation()` as Thin Wrapper**

**Purpose**: Backward compatibility for single-checkpoint evaluation
**Implementation Strategy**:
```python
def run_evaluation(...):
    """Legacy wrapper - calls run_multi_model_evaluation for raw checkpoint only."""
    # Can be deprecated later, but keep for now to avoid breaking existing code
    # Simply delegates to run_multi_model_evaluation with only raw checkpoint
```

---

### **Part 4: EMA State Loading for Resume Finetune**

#### **4.1 Modify `run_resume_finetune()`** (line 649-756)

**After line 723** (after `_apply_resume_state()`):
```python
# After _apply_resume_state(finetuner, checkpoint_report)

# Load EMA checkpoint for resume (already implemented helper)
_load_ema_state_for_resume(
    trainer=finetuner,
    model_name="v3",
    stage="finetune",
    run_id=finetune_run_id,
    best=best,
    epoch=checkpoint_report.get("epoch"),
    is_direct_path=bool(direct_path),
    logger=finetune_logger,
)
```

**Note**: 
- The function `_load_ema_state_for_resume()` already exists (line 196-247 in `run.py`)
- ✅ Just need to add the call

---

### **Part 5: Checkpoint Saving Verification**

#### **5.1 Verify `v3Finetuner` Saves EMA Checkpoints**

**Check `finetune_v3.py` lines 619-639**:
```python
# Save EMA checkpoint alongside best when enabled
if (
    getattr(self, "use_ema", False)
    and getattr(self, "ema", None) is not None
):
    try:
        ema_state = self.ema.state_dict()
        save_checkpoint(
            model=self.device_manager.unwrap_model(self.model),
            stage="finetune",
            run_id=self.run_id,
            best=True,
            model_name=self.model_name,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            ema=True,
            ema_state_dict=ema_state,
        )
    except Exception as e:
        self.logger.warning(f"Failed to save EMA checkpoint: {e}")
```
✅ **Already implemented** - saves both "best" and "last" EMA checkpoints

#### **5.2 Verify SWA Checkpoint Saving**

**Check `finetune_v3.py` lines 686-717**:
```python
# PR6 Milestone 4: Save SWA checkpoints when enabled and started
if self.use_swa and self.swa_started and self.swa_model is not None:
    try:
        swa_state = self.swa_model.module.state_dict()
        # Save last always; best if is_best
        save_checkpoint(
            model=self.device_manager.unwrap_model(self.model),
            stage="finetune",
            run_id=self.run_id,
            best=False,
            model_name=self.model_name,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            swa=True,
            swa_state_dict=swa_state,
        )
        if is_best:
            save_checkpoint(
                model=self.device_manager.unwrap_model(self.model),
                stage="finetune",
                run_id=self.run_id,
                best=True,
                model_name=self.model_name,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                swa=True,
                swa_state_dict=swa_state,
            )
    except Exception as e:
        self.logger.warning(f"Failed to save SWA checkpoint(s): {e}")
```
✅ **Already implemented** - saves both "best" and "last" SWA checkpoints

---

### **Part 6: Config Validation & Alignment**

#### **6.1 Current Config State** (`configs/v3.yaml`)

**Finetune EMA/SWA settings** (lines 175-181):
```yaml
# PR6-M4: EMA/SWA (enabled for finetune)
use_ema: true
ema_decay: 0.9995
use_swa: true
swa_start_epoch: 28
swa_lr: 0.000005
swa_anneal_epochs: 5
```
✅ **No changes needed**

**Pretrain EMA settings** (lines 138-139):
```yaml
# PR6-M4: EMA (enabled for pretrain)
use_ema: true
ema_decay: 0.999
```
✅ **No changes needed**

**Evaluate metrics** (lines 197-217):
```yaml
evaluate:
  metrics: [
    "accuracy@0.5",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "roc_auc",
    "auprc",
    "pr_auc",
    "ap",
    "average_precision",
    "tp", "tn", "fp", "fn",
    "loss"
  ]
```
✅ **No changes needed**

---

## 🎯 Summary of Changes

### **Files to Modify**:

1. **`src/run.py`**:
   - **Add** `_load_pretrain_ema_for_finetune()` helper function (after line 247)
   - **Add** `run_multi_model_evaluation()` function (replace `run_evaluation()`)
   - **Modify** `run_finetune_from_pretrain()` (line 468-569):
     - Load pretrain EMA checkpoint before `fit()` (line ~548)
     - Replace eval stage with multi-model evaluation (line 552-563)
   - **Modify** `run_resume_finetune()` (line 649-756):
     - Add EMA state loading call (after line 723)
     - Replace eval stage with multi-model evaluation (line 740-750)
   - **Modify** `run_full_pipeline()` (line 363-466):
     - Replace eval stage with multi-model evaluation (line 448-459)

2. **`src/utils/logging.py`**:
   - **Add** `append_test_results_by_model()` function (after line 660)

3. **`src/finetune/finetune_v3.py`**:
   - **Modify** `on_validation_end()` (line 480-853):
     - Add EMA evaluation after raw evaluation (after line 502)
     - Update best model selection to consider both raw and EMA (lines 507-593)
     - Update training row to include EMA metrics (lines 738-747)

### **New CSV Files Created**:

**During training** (existing file, enhanced):
- `logs/v3/finetune/{run_id}/training_step.csv`
  - **New columns**: `auroc_ema`, `ap_ema`, `loss_ema`, `accuracy@0.5_ema`, etc.
  - Example row:
    ```
    epoch,is_best,best_metric,auroc,ap,loss,auroc_ema,ap_ema,loss_ema
    1,True,0.85,0.83,0.80,0.25,0.85,0.82,0.23
    ```

**After final test** (new files):
- `logs/v3/finetune/{run_id}/v3_finetune_{run_id}_eval_raw.csv`
- `logs/v3/finetune/{run_id}/v3_finetune_{run_id}_eval_ema.csv` (if EMA used)
- `logs/v3/finetune/{run_id}/v3_finetune_{run_id}_eval_swa.csv` (if SWA used)

**Diagnostic outputs** (if `save_plots=true`):
- `logs/v3/finetune/{run_id}_raw/diagnostics/` (raw model diagnostics)
- `logs/v3/finetune/{run_id}_ema/diagnostics/` (EMA model diagnostics)
- `logs/v3/finetune/{run_id}_swa/diagnostics/` (SWA model diagnostics)

---

## ⚠️ Edge Cases & Error Handling

### **1. Pretrain EMA checkpoint missing**
**Scenario**: Finetune configured with `use_ema=true`, but pretrain didn't save EMA checkpoint
- **Handling**: Log warning and initialize fresh EMA from current raw model
- **Message**: `"Pretrain EMA checkpoint not found; initializing fresh EMA"`

### **2. EMA/SWA checkpoints missing at test time**
**Scenario**: Final test runs but EMA or SWA checkpoints don't exist
- **Handling**: Skip gracefully, only evaluate available checkpoints
- **Message**: `"EMA checkpoint not found, skipping EMA evaluation"`

### **3. Model loading failures**
**Scenario**: Checkpoint file corrupted or incompatible
- **Handling**: Catch exception, log error, continue with other models
- **Message**: `"Failed to load {model_type} checkpoint: {error}"`
- **Behavior**: Don't fail entire evaluation if one model type fails

### **4. Memory constraints**
**Scenario**: Multiple large models cause OOM
- **Handling**: 
  - Reuse same model instance (don't duplicate in memory)
  - Clear cache between checkpoint loads: `torch.cuda.empty_cache()`
  - Set model to eval mode to disable dropout/gradients

### **5. DDP mode**
**Scenario**: Running with DistributedDataParallel
- **Handling**:
  - Only rank 0 writes CSV files (already handled by `is_primary()`)
  - Only rank 0 runs diagnostics
  - Broadcast results if needed for synchronization

### **6. Evaluator not available**
**Scenario**: `v3Finetuner` created without evaluator
- **Handling**: Fall back to validation loss only (existing behavior)
- **Impact**: EMA evaluation skipped, only raw validation loss tracked

### **7. Dynamic CSV columns**
**Scenario**: Different epochs may have different metric sets
- **Handling**: `append_training_row()` uses `DictWriter` which handles dynamic columns
- **Verification**: Test with/without EMA to ensure CSV doesn't break

### **8. Diagnostic directory collisions**
**Scenario**: Raw, EMA, SWA diagnostics try to write to same directory
- **Handling**: Use unique run_id suffix per model type:
  - Raw: `{run_id}_raw/diagnostics/`
  - EMA: `{run_id}_ema/diagnostics/`
  - SWA: `{run_id}_swa/diagnostics/`

---

## ✅ Validation & Testing Plan

### **Test 1: Pretrain → Finetune → Eval**

**Steps**:
1. Run pretrain with `use_ema=true`
2. Verify checkpoint saved: `models/v3/pretrain/{run_id}/best_model_ema.pth`
3. Run finetune from pretrain
4. Check logs for: `"Loaded pretrain EMA checkpoint for continuation"`
5. Verify `training_step.csv` has EMA columns: `auroc_ema`, `ap_ema`, etc.
6. Verify final test produces 3 CSV files: `eval_raw.csv`, `eval_ema.csv`, `eval_swa.csv`
7. Compare metrics: EMA should typically match or exceed raw model

**Expected behavior**:
- EMA checkpoint loads successfully
- Both raw and EMA metrics logged every epoch
- Best model selection considers both raw and EMA
- Final test evaluates all three model variants

### **Test 2: Resume Finetune**

**Steps**:
1. Start finetune, stop mid-training (e.g., epoch 5/40)
2. Verify checkpoints: `best_model.pth`, `best_model_ema.pth`
3. Resume finetune from `best_model.pth`
4. Check logs for: `"Resume EMA inferred from..."` or similar
5. Verify `training_step.csv` continues with same columns
6. Training should continue from epoch 6

**Expected behavior**:
- EMA state restored from checkpoint
- EMA tracking continues from saved state (not reset)
- Metrics consistent with pre-interruption values

### **Test 3: Missing EMA Checkpoint**

**Steps**:
1. Manually delete `best_model_ema.pth` from pretrain
2. Run finetune with `use_ema=true`
3. Check logs for warning
4. Verify training proceeds with fresh EMA initialization

**Expected behavior**:
- Warning logged: `"Pretrain EMA checkpoint not found..."`
- Training continues normally
- EMA initialized from current raw model

### **Test 4: CSV Format Validation**

**Verify CSV structure**:
```csv
# training_step.csv
epoch,is_best,best_metric,best_metric_name,auroc,ap,loss,auroc_ema,ap_ema,loss_ema
0,True,0.80,val_auc,0.78,0.75,0.28,0.80,0.77,0.26
1,True,0.82,val_auc,0.82,0.79,0.24,0.83,0.80,0.23

# v3_finetune_{run_id}_eval_raw.csv
test_set,auroc,auprc,ap,accuracy@0.5,loss
eval_balanced,0.8018,0.7971,0.7973,0.7444,1.1223
eval_realistic,0.9026,0.5698,0.5700,0.9446,0.1623

# v3_finetune_{run_id}_eval_ema.csv
test_set,auroc,auprc,ap,accuracy@0.5,loss
eval_balanced,0.8125,0.8050,0.8055,0.7550,1.0890
eval_realistic,0.9100,0.5820,0.5825,0.9480,0.1580

# v3_finetune_{run_id}_eval_swa.csv
test_set,auroc,auprc,ap,accuracy@0.5,loss
eval_balanced,0.8200,0.8100,0.8105,0.7600,1.0500
eval_realistic,0.9150,0.5900,0.5905,0.9500,0.1500
```

### **Test 5: Diagnostic Outputs**

**Verify diagnostic directories**:
```
logs/v3/finetune/{run_id}_raw/diagnostics/
  - attention_entropy_histograms.png
  - attention_maps_sample_0.png
  - diagnostic_summary.json

logs/v3/finetune/{run_id}_ema/diagnostics/
  - (same structure, different results)

logs/v3/finetune/{run_id}_swa/diagnostics/
  - (same structure, different results)
```

---

## 🔍 Key Design Decisions

### **Decision 1: Best Model Selection**
**Question**: Should "best" be determined by raw model, EMA model, or both?
**Decision**: **Consider both** - An epoch is "best" if either raw OR EMA shows improvement
**Rationale**: 
- Aligns with user requirement to "consider both ema and raw model to choose the best model"
- More advanced than pretrain (which only tracks raw)
- Maximizes chance of capturing optimal checkpoint

### **Decision 2: Model Reuse vs. Recreation**
**Question**: Should we create new model instances for each checkpoint type?
**Decision**: **Reuse same model instance** with different weight loading
**Rationale**:
- Memory efficient (no duplication)
- Faster (no model compilation overhead)
- Simpler code (single model instance)

### **Decision 3: Diagnostic Outputs**
**Question**: Run diagnostics on all three models or just raw?
**Decision**: **All three models** with unique output directories
**Rationale**:
- Enables comparison of attention patterns across model types
- Helps understand how EMA/SWA affect model behavior
- Avoids file overwrites with suffix-based directory naming

### **Decision 4: CSV Format**
**Question**: Single CSV with model_type column or separate CSVs?
**Decision**: **Separate CSV files** per model type
**Rationale**:
- Easier to parse and compare
- Avoids confusion in downstream analysis
- Consistent with internal CSV structure (test_set as row identifier)

---

## 📚 References

**Related Files**:
- `src/train/train_v3.py:on_validation_end()` - Pretrain EMA checkpoint saving logic
- `src/train/base.py:__init__()` - EMA initialization and `ema_pending_state`
- `src/utils/checkpoint.py` - Checkpoint save/load utilities
- `src/utils/logging.py` - CSV logging utilities
- `configs/v3.yaml` - EMA/SWA configuration

**Design Patterns**:
- Dependency Injection: All components receive dependencies, no globals
- Single Responsibility: Each function has one clear purpose
- Fail-Safe: Catch exceptions, log warnings, continue gracefully
- DRY: Reuse existing checkpoint/logging infrastructure