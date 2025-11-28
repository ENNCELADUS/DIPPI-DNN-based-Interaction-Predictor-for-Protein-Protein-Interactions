# Roadmap: Imbalanced Batch Sampling + Distribution Alignment (DA)

**Purpose**: Implement two-stage training (pretrain → finetune) with 1:3 pos:neg batch sampling and Distribution Alignment to handle class imbalance and distribution shift between training (≈0.14 prior) and test (0.5 balanced prior).

**Status**: Planning
**Estimated effort**: 3-4 hours implementation + 1 hour testing

---

## 1. Problem Statement

### Current State
- **Data**: TMP protein interaction dataset with natural imbalance (≈14% positive, 86% negative)
- **Challenge**: Standard uniform sampling gives poor gradient signal for rare positive class
- **Distribution shift**: Training data has ≈0.14 prior, but final test is balanced (0.5 prior)

### Goals
1. **Pretrain stage**: Use 1:3 batch sampling to improve gradient signal without DA
2. **Finetune stage**: Use 1:3 sampling + DA to calibrate predictions for balanced test distribution
3. **Evaluation**: Apply saved DA bias + threshold to balanced test set

---

## 2. Solution Overview

### Two-Stage Training Pipeline

**Stage 1 — Pretrain (Large TMP-involving dataset)**
- Dataset: `pretrain_train.csv` (~917k pairs, ≈0.14 prior)
- Sampling: 1:3 pos:neg batch sampling (all positives once per epoch + sampled negatives)
- Loss: BCEWithLogitsLoss (no pos_weight, sampling handles balance)
- Validation: Standard (full val set, no sampling)
- Best model: Selected by raw validation metric (e.g., AUROC or val loss)
- **No DA applied**

**Stage 2 — Finetune (TMP-focused dataset + DA)**
- Dataset: `finetune_train.csv` (~245k pairs, ≈0.14 prior)
- Sampling: 1:3 pos:neg batch sampling
- Freeze: Encoder frozen, head-only training initially (staged unfreeze at epoch 3)
- Loss: BCEWithLogitsLoss + label_smoothing (no pos_weight or class weights)
- Validation: DA calibration on full val set (no separate raw validation)
- DA: After each epoch, estimate bias to shift predictions to target_prior=0.5, search threshold to maximize F1
- Best model: Selected by **DA-calibrated validation F1** (not raw loss)
- Checkpoint: Save model weights + da_bias + da_threshold
- Note: **All validation metrics during finetune are DA-calibrated** (no raw metrics logged)

**Stage 3 — Evaluation (Balanced TMP–TMP test)**
- Dataset: `test_balanced.csv` (~9.5k pairs, 0.5 prior)
- Load: Best finetune checkpoint with saved da_bias + da_threshold
- Inference: Apply `logits_calibrated = logits + da_bias`, then threshold
- Report: AUROC, PR-AUC, F1, balanced accuracy, MCC, etc.

---

## 3. Component Design

### 3.1 ImbalancedBatchSampler

**File**: `src/utils/samplers.py` (new file)

**Role**: Construct batches with target pos:neg ratio (default 1:3), ensuring all positives are sampled once per epoch.

**API**:
```python
class ImbalancedBatchSampler:
    """
    Batch sampler that maintains target pos:neg ratio per batch.
    
    Each epoch: iterate through all positives once (without replacement) + 
                sample negatives (with replacement) to achieve target ratio.
    """
    
    def __init__(
        self,
        labels: Sequence[int],           # All labels from dataset (0/1)
        batch_size: int,
        pos_neg_ratio: float = 3.0,      # neg_count = pos_count * ratio
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            labels: Binary labels (0=negative, 1=positive) for all samples
            batch_size: Target batch size
            pos_neg_ratio: Ratio of negatives to positives (default 3.0 → 1:3)
            shuffle: Whether to shuffle positive indices each epoch
            drop_last: Whether to drop incomplete final batch
            seed: Random seed for reproducibility
        """
        
    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batch indices with target pos:neg ratio.
        
        Each batch:
        - Sample pos_per_batch positives (without replacement until exhausted)
        - Sample neg_per_batch = pos_per_batch * ratio negatives (with replacement)
        - Shuffle and yield indices
        
        Returns:
            Iterator of lists of sample indices
        """
        
    def __len__(self) -> int:
        """Number of batches per epoch (based on positive count / pos_per_batch)."""
```

**Implementation notes**:
- Positive indices tracked and shuffled each epoch (if `shuffle=True`)
- Negative indices sampled with replacement using `random.choices()` or `np.random.choice()`
- Batch size calculation: 
  - `pos_per_batch = batch_size // (1 + ratio)`
  - `neg_per_batch = batch_size - pos_per_batch`
- Length = `ceil(num_positives / pos_per_batch)` batches per epoch

**DDP strategy**: Option A (simple)
- Each rank independently samples from full positive/negative index sets
- Each positive seen once **per rank** (not once globally)
- This matches standard PyTorch DDP behavior and is simpler than partitioning

**Example**:
```python
# Dataset with 100 positives, 900 negatives, batch_size=32, ratio=3.0
# → pos_per_batch = 32 / 4 = 8, neg_per_batch = 24
# → 100/8 = 13 batches per epoch
# → Each batch: 8 random positives (no replacement) + 24 random negatives (with replacement)
```

---

### 3.2 DistributionAligner

**File**: `src/finetune/distribution_alignment.py` (new file)

**Role**: Calibrate model predictions to target prior and search for optimal threshold.

**API**:
```python
class DistributionAligner:
    """
    Distribution Alignment for binary classification under prior shift.
    
    Given a model trained on imbalanced data (e.g., 0.14 prior), calibrate predictions
    to match a target prior (e.g., 0.5) for evaluation on balanced test data.
    
    Reference: Simplified binary version of DisAlign (long-tail learning).
    """
    
    def __init__(
        self,
        target_prior: float = 0.5,
        search_metric: str = "f1",       # Metric for threshold search
        search_steps: int = 100,         # Number of thresholds to try
    ):
        """
        Args:
            target_prior: Target positive prior (0.5 for balanced test)
            search_metric: Metric to optimize during threshold search
                          Options: "f1", "balanced_accuracy", "mcc"
            search_steps: Number of thresholds in [0,1] to evaluate
        """
        
    def calibrate_and_search(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Calibrate predictions and search for optimal threshold.
        
        Steps:
        1. Collect logits and labels from validation set
        2. Estimate predicted prior: mean(sigmoid(logits))
        3. Compute bias to shift prior: solve sigmoid(logit + bias).mean() = target_prior
        4. Apply bias and search threshold to maximize search_metric
        5. Compute final metrics with bias + threshold
        
        Args:
            model: Model in eval mode (caller must set model.eval())
            val_loader: Validation dataloader (full val set, no sampling)
            device: Device for inference
            
        Returns:
            {
                "bias": float,                    # Logit bias
                "threshold": float,               # Optimal threshold
                "loss": float,                    # Validation loss (computed during collection)
                "predicted_prior": float,         # Original predicted prior
                "calibrated_prior": float,        # After bias (should ≈ target_prior)
                "metrics": {                      # Metrics with bias + threshold
                    "f1": float,
                    "auroc": float,
                    "auprc": float,
                    "balanced_accuracy": float,
                    "precision": float,
                    "recall": float,
                    "mcc": float,
                }
            }
        """
```

**Implementation notes**:
- **Bias computation**: Use binary search or Newton's method to solve:
  ```
  mean(sigmoid(logits + bias)) = target_prior
  ```
  Closed-form approximation: `bias ≈ logit(target_prior) - logit(predicted_prior)`
  
- **Threshold search**: 
  - Try `search_steps` thresholds uniformly in [0, 1]
  - For each threshold, compute predictions = (calibrated_scores > threshold)
  - Compute search_metric (e.g., F1) and select best
  
- **Metrics**: Use sklearn or torchmetrics to compute:
  - F1, balanced accuracy, MCC (from binary predictions)
  - AUROC, AUPRC (from continuous calibrated scores)

**Example usage**:
```python
# In finetune loop after validation pass
aligner = DistributionAligner(target_prior=0.5, search_metric="f1")
model.eval()
with torch.no_grad():
    da_result = aligner.calibrate_and_search(model, val_loader, device)

# Log and checkpoint
logger.info(f"DA bias: {da_result['bias']:.4f}, threshold: {da_result['threshold']:.4f}")
logger.info(f"DA metrics: F1={da_result['metrics']['f1']:.4f}, AUROC={da_result['metrics']['auroc']:.4f}")

# Save if best
if da_result['metrics']['f1'] > best_f1:
    save_checkpoint(model, epoch, checkpoint_path, 
                   da_bias=da_result['bias'], 
                   da_threshold=da_result['threshold'],
                   da_metrics=da_result['metrics'])
```

---

## 4. Integration Points

### 4.1 Update `src/utils/data_io.py`

**Modify `build_dataloaders()`**:
- Detect when to use `ImbalancedBatchSampler` based on config
- For train loaders: use custom sampler if `sampling.strategy == "imbalanced"`
- For val/test loaders: keep standard `DistributedSampler`

**Pseudocode**:
```python
def build_dataloaders(cfg, emb_index, loaders):
    # ... existing setup ...
    
    for loader_name in loaders:
        dataset = ProteinPairDataset(csv_path, emb_index, ...)
        
        if "train" in loader_name:
            # Check if imbalanced sampling enabled
            stage = "pretrain" if "pretrain" in loader_name else "finetune"
            sampling_cfg = data_config[stage].get("sampling", {})
            
            if sampling_cfg.get("strategy") == "imbalanced":
                # Use custom batch sampler
                labels = dataset.pairs_df["label"].values  # Extract labels
                batch_sampler = ImbalancedBatchSampler(
                    labels=labels,
                    batch_size=stage_cfg["batch_size"],
                    pos_neg_ratio=sampling_cfg.get("pos_neg_ratio", 3.0),
                    shuffle=True,
                    seed=base_seed + rank,
                )
                result[loader_name] = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,  # Use batch_sampler, not sampler
                    **loader_kwargs
                )
            else:
                # Standard DistributedSampler
                # ... existing code ...
        else:
            # Val/test: always use standard sampler (no resampling)
            # ... existing code ...
```

**Key changes**:
- Add `sampling` subsection parsing in pretrain/finetune configs
- Switch between `batch_sampler` (for ImbalancedBatchSampler) and `sampler` (for DistributedSampler)
- Note: `batch_sampler` and `sampler` are mutually exclusive in DataLoader

---

### 4.2 Update `src/stages/finetune.py`

**Add DA integration in finetune loop**:

**Pseudocode**:
```python
def run_finetune(cfg, model, train_loader, val_loader, device, run_id, log_dir, checkpoint_dir, load_checkpoint_path):
    # ... existing setup (load checkpoint, build trainer, etc.) ...
    
    # Initialize DA (always enabled in finetune)
    ft_cfg = cfg["finetune_config"]
    da_cfg = ft_cfg.get("distribution_alignment", {})
    
    from src.finetune.distribution_alignment import DistributionAligner
    aligner = DistributionAligner(
        target_prior=da_cfg.get("target_prior", 0.5),
        search_metric=da_cfg.get("threshold_search_metric", "f1"),
        search_steps=da_cfg.get("search_steps", 100),
    )
    logger.info(f"DA: target_prior={da_cfg.get('target_prior', 0.5)}, metric={da_cfg.get('threshold_search_metric', 'f1')}")
    
    best_da_metric = -float('inf')
    best_epoch = -1
    
    for epoch in range(epochs):
        # Train one epoch with 1:3 sampling
        train_stats = trainer.train_one_epoch(train_loader)
        
        # DA calibration and validation (finetune always uses DA)
        model.eval()
        with torch.no_grad():
            da_result = aligner.calibrate_and_search(model, val_loader, device)
        
        # Log training and DA-calibrated validation metrics
        logger.info(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}")
        logger.info(f"DA: bias={da_result['bias']:.4f}, threshold={da_result['threshold']:.4f}")
        logger.info(f"DA metrics: F1={da_result['metrics']['f1']:.4f}, AUROC={da_result['metrics']['auroc']:.4f}, val_loss={da_result['loss']:.4f}")
        
        # Select best model by DA-calibrated metric
        current_metric = da_result['metrics'][da_cfg.get('threshold_search_metric', 'f1')]
        if current_metric > best_da_metric:
            best_da_metric = current_metric
            best_epoch = epoch
            
            # Save checkpoint with DA params
            checkpoint_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(
                model, epoch, checkpoint_path,
                extra={
                    'da_bias': da_result['bias'],
                    'da_threshold': da_result['threshold'],
                    'da_metrics': da_result['metrics'],
                }
            )
            logger.info(f"Saved best finetune checkpoint (DA F1={current_metric:.4f}): {checkpoint_path}")
        
        # Logging: append row to training_step.csv with DA metrics
        # Columns: Epoch, Epoch Time, Train Loss, Val Loss, Val AUROC, Val F1, DA Bias, DA Threshold, LR
        row = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train_loss': train_stats['loss'],
            'val_loss': da_result['loss'],  # Loss computed during DA pass
            'val_auroc': da_result['metrics']['auroc'],
            'val_f1': da_result['metrics']['f1'],
            'da_bias': da_result['bias'],
            'da_threshold': da_result['threshold'],
            'lr': train_stats['lr'],
        }
        append_row(log_dir / "training_step.csv", row)
        
        # Early stopping: check DA-calibrated metric
        early_stop_metric = da_result['metrics'][da_cfg.get('threshold_search_metric', 'f1')]
        if check_early_stop(metric_history, patience, mode='max'):
            logger.info(f"Early stopping at epoch {epoch}")
            break
```

**Key changes**:
- Instantiate `DistributionAligner` (always enabled in finetune)
- After each epoch, run DA calibration on full val set
- Save best model based on DA-calibrated metric (F1 by default)
- Log only DA-calibrated metrics (no separate raw validation)
- Checkpoint format includes `da_bias`, `da_threshold`, `da_metrics`
- Early stopping checks DA-calibrated metric

---

### 4.3 Update `src/utils/checkpoint.py`

**Extend checkpoint format**:

**Current format**:
```python
{
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'loss': loss,
}
```

**Extended format** (for finetune with DA):
```python
{
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'da_bias': float,           # NEW: DA logit bias
    'da_threshold': float,      # NEW: DA threshold
    'da_metrics': dict,         # NEW: DA-calibrated metrics (F1, AUROC, etc.)
}
```

**Modifications**:
- `save_checkpoint()`: Accept optional `extra: dict` param to store DA params
- `load_checkpoint()`: Return full checkpoint dict (caller extracts `da_bias`/`da_threshold` if present)

**No API changes needed** — existing signature already supports this via `extra` parameter (see 06mvp_roadmap.md).

---

### 4.4 Update `src/stages/evaluate.py`

**Apply DA at test time**:

**Pseudocode**:
```python
def run_evaluation(cfg, model, device, eval_run_id, log_dir, load_checkpoint_path):
    # Load checkpoint (may contain DA params)
    checkpoint = load_checkpoint(model, load_checkpoint_path, map_location=device)
    
    # Extract DA params if present
    da_bias = checkpoint.get('da_bias', 0.0)
    da_threshold = checkpoint.get('da_threshold', 0.5)
    has_da = 'da_bias' in checkpoint
    
    if has_da:
        logger.info(f"Loaded checkpoint with DA: bias={da_bias:.4f}, threshold={da_threshold:.4f}")
    else:
        logger.info("Checkpoint has no DA params; using raw predictions")
    
    # Build test loader(s)
    test_loaders = build_dataloaders(cfg, emb_index, {'eval'})
    
    # Run evaluation
    model.eval()
    with torch.no_grad():
        for loader_name, test_loader in test_loaders.items():
            # Collect logits and labels
            all_logits = []
            all_labels = []
            for batch in test_loader:
                batch = to_device(batch, device)
                outputs = model(batch)
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch['labels'].cpu())
            
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            
            # Apply DA if available
            if has_da:
                calibrated_logits = all_logits + da_bias
                calibrated_scores = torch.sigmoid(calibrated_logits)
                predictions = (calibrated_scores > da_threshold).long()
            else:
                calibrated_scores = torch.sigmoid(all_logits)
                predictions = (calibrated_scores > 0.5).long()
            
            # Compute metrics
            metrics = compute_metrics(
                labels=all_labels,
                scores=calibrated_scores,
                predictions=predictions,
                metrics_list=cfg['evaluate']['metrics']
            )
            
            # Log to evaluate.csv
            append_row(
                csv_path=log_dir / "evaluate.csv",
                row_dict=metrics,
                columns=cfg['evaluate']['metrics']
            )
            logger.info(f"Evaluation results ({loader_name}): {metrics}")
```

**Key changes**:
- Extract DA params from checkpoint (if present)
- Apply `logits + bias` before computing scores
- Use `da_threshold` instead of 0.5 for binary predictions
- Log whether DA was applied

---

## 5. Config Schema Updates

### 5.1 Update `configs/v3.yaml`

**Changes to `data_config`**:
```yaml
data_config:
  embeddings_path: "data/embedding/complete_TMP_embeddings.npz"  # NEW PATH
  embedding_dtype: "bf16"
  max_sequence_length: 1024

  pretrain:
    train_csv: "data/TMP_protein/processed/pair-no-leakage/split/pretrain_train.csv"  # NEW PATH
    valid_csv: "data/TMP_protein/processed/pair-no-leakage/split/pretrain_val.csv"    # NEW PATH
    sampling:                          # NEW SECTION
      strategy: "imbalanced"           # "imbalanced" | "uniform"
      pos_neg_ratio: 3.0               # Negatives per positive in each batch
      
  finetune:
    train_csv: "data/TMP_protein/processed/pair-no-leakage/split/finetune_train.csv"  # NEW PATH
    valid_csv: "data/TMP_protein/processed/pair-no-leakage/split/finetune_val.csv"    # NEW PATH
    sampling:                          # NEW SECTION
      strategy: "imbalanced"
      pos_neg_ratio: 3.0
    distribution_alignment:            # NEW SECTION (always enabled in finetune)
      target_prior: 0.5                # Target prior for balanced test
      threshold_search_metric: "f1"    # Metric for threshold optimization ("f1" | "balanced_accuracy" | "mcc")
      search_steps: 100                # Number of thresholds to try
      
  evaluate:
    test_balanced: "data/TMP_protein/processed/pair-no-leakage/test_balanced.csv"  # NEW PATH
```

**Changes to `pretrain_config` loss**:
```yaml
pretrain_config:
  # ... other settings ...
  
  loss:
    type: "bce_with_logits"
    # pos_weight: 3.0              # REMOVED - sampling handles balance
    label_smoothing: 0.03          # KEEP
    # use_class_weights: true      # REMOVED - no extra reweighting
    l1_lambda: 2.0e-6              # KEEP (orthogonal regularization)
```

**Changes to `finetune_config` loss**:
```yaml
finetune_config:
  # ... other settings ...
  
  loss:
    type: "bce_with_logits"
    # pos_weight: 2.0              # REMOVED
    label_smoothing: 0.01          # KEEP
    # use_class_weights: true      # REMOVED
    l1_lambda: 1.0e-6              # KEEP
```

**Summary of config changes**:
1. Update all CSV paths to `data/TMP_protein/processed/pair-no-leakage/split/*.csv`
2. Update embeddings path to `data/embedding/complete_TMP_embeddings.npz`
3. Add `sampling` section to pretrain/finetune with `strategy` and `pos_neg_ratio`
4. Add `distribution_alignment` section to finetune
5. Remove `pos_weight` and `use_class_weights` from loss configs
6. Keep `label_smoothing` and `l1_lambda` (orthogonal regularization)

---

## 6. Role Boundaries (Strict MVP)

| Module | Owns | Does NOT Own |
|--------|------|--------------|
| `samplers.py` | 1:3 batch index generation | Training loop, loss computation |
| `distribution_alignment.py` | Bias computation, threshold search, metric calculation | Training, checkpointing, logging |
| `data_io.py` | DataLoader construction, sampler selection | Training logic, validation |
| `stages/finetune.py` | Calling DA after validation, checkpoint decisions | Bias computation, metric math |
| `stages/evaluate.py` | Loading checkpoint, applying DA at test time | DA calibration (uses saved params) |
| `checkpoint.py` | Save/load checkpoint with optional extra fields | Metric comparison, DA computation |
| `run.py` | Mode selection, stage orchestration | Sampling logic, DA math |

**Key principles**:
- DA logic is **isolated** in `distribution_alignment.py`
- Stage runners (`finetune.py`, `evaluate.py`) **call** DA but don't implement it
- `run.py` orchestrates stages but **delegates** all computation
- No cross-contamination: samplers don't know about DA, DA doesn't know about sampling

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Test `ImbalancedBatchSampler`**:
- Verify pos:neg ratio maintained across batches
- Verify all positives sampled once per epoch
- Verify negatives sampled with replacement
- Verify DDP independence (each rank samples full dataset)

**Test `DistributionAligner`**:
- Verify bias computation shifts prior correctly
- Verify threshold search finds optimal threshold
- Verify metrics computed correctly
- Verify edge cases (all positive, all negative, etc.)

### 7.2 Integration Tests

**Test pretrain with 1:3 sampling**:
- Run 2 epochs on small dataset (~1000 samples)
- Verify batch composition (check labels in each batch)
- Verify no DA applied
- Verify checkpoint saved without DA params

**Test finetune with 1:3 sampling + DA**:
- Run 2 epochs on small dataset
- Verify DA calibration runs after each validation
- Verify checkpoint saved with DA params
- Verify best model selected by DA metric (not raw loss)

**Test evaluation with DA**:
- Load checkpoint from finetune test
- Verify DA bias + threshold applied to test predictions
- Verify metrics computed on calibrated predictions

### 7.3 Smoke Test (Full Pipeline)

**Run with test config** (`configs/test_e2e.yaml`):
- Small dataset (~5k pretrain, ~1k finetune, ~500 test)
- 3 pretrain epochs, 3 finetune epochs
- Verify:
  - Pretrain: 1:3 sampling works, no DA, checkpoint saved
  - Finetune: 1:3 sampling + DA works, best checkpoint has DA params
  - Eval: DA applied, metrics reasonable

**Verify outputs**:
- `logs/v3/pretrain/<run_id>/training_step.csv` has correct columns
- `logs/v3/finetune/<run_id>/training_step.csv` has DA metrics
- `logs/v3/evaluate/<run_id>/evaluate.csv` has final metrics
- Checkpoints at `models/v3/pretrain/<run_id>/best_model.pth` (no DA)
- Checkpoint at `models/v3/finetune/<run_id>/best_model.pth` (with DA)

---

## 8. Implementation Order

**Phase 1: Sampling (No DA)**
1. ✅ Update config paths in `configs/v3.yaml` (5 min)
2. Create `src/utils/samplers.py` with `ImbalancedBatchSampler` (45 min)
3. Update `src/utils/data_io.py` to use custom sampler (20 min)
4. Write unit tests for sampler (20 min)
5. Run pretrain smoke test to verify 1:3 sampling (10 min)

**Phase 2: Distribution Alignment**
6. Create `src/finetune/distribution_alignment.py` with `DistributionAligner` (60 min)
7. Write unit tests for DA (30 min)
8. Update `src/stages/finetune.py` to integrate DA (30 min)
9. Update `src/stages/evaluate.py` to apply DA at test time (15 min)
10. Run finetune smoke test to verify DA works (15 min)

**Phase 3: End-to-End Testing**
11. Run full pipeline smoke test with test config (20 min)
12. Verify all outputs (CSVs, logs, checkpoints) (10 min)
13. Fix any issues discovered (30 min buffer)

**Total estimated time**: ~4.5 hours implementation + 1 hour testing = **5.5 hours**

---

## 9. Open Questions / Future Extensions

### Resolved (with user clarification)
- ✅ DDP strategy: Option A (each rank samples independently)
- ✅ DA scope: Finetune only, not pretrain
- ✅ Best model selection: DA-calibrated F1, not raw loss
- ✅ Checkpoint format: Include da_bias, da_threshold, da_metrics
- ✅ Loss config: Remove pos_weight and use_class_weights
- ✅ Validation: Full val set (no sampling)
- ✅ Threshold metric: F1 as default (configurable)
- ✅ Target prior: 0.5 to match balanced test distribution

### Future Extensions (Out of Scope for MVP)
- **DDP Option B**: Partition positives across ranks (more complex, requires custom collate)
- **DA for pretrain**: Could apply if pretraining on imbalanced data with balanced validation
- **Multiple target priors**: Support different priors for different test distributions
- **Temperature scaling**: More sophisticated calibration than simple bias
- **Focal loss**: Alternative to sampling for handling imbalance (changes loss function)
- **Adaptive sampling**: Adjust pos:neg ratio during training based on loss/metrics

---

## 10. Success Criteria

**MVP is complete when**:
1. ✅ Pretrain runs with 1:3 sampling, no DA, saves checkpoint
2. ✅ Finetune runs with 1:3 sampling + DA, saves checkpoint with DA params
3. ✅ Evaluation loads checkpoint and applies DA to test predictions
4. ✅ All CSVs/logs written correctly with DA metrics
5. ✅ Unit tests pass for sampler and DA
6. ✅ Integration test passes (full pipeline smoke test)
7. ✅ No regression: existing models (without DA) still work

**Acceptance test**: Run full pipeline on TMP dataset
- Pretrain: ~30 epochs on ~917k pairs → saves best by AUROC
- Finetune: ~15 epochs on ~245k pairs with DA → saves best by DA-F1
- Evaluate: Load best and test on ~9.5k balanced pairs → AUROC > 0.75, F1 > 0.70

---

## 11. References

**Long-tail learning & Distribution Alignment**:
- DisAlign (2021): "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition"
- Classifier re-balancing: Train on imbalanced, calibrate for balanced test

**Batch sampling for imbalance**:
- PyTorch WeightedRandomSampler (sample-level reweighting)
- Custom BatchSampler for ratio control (this approach)

**Threshold search**:
- sklearn.metrics: precision_recall_curve, roc_curve
- Maximize F1 by searching thresholds on validation predictions

**Prior shift / label shift**:
- BBSE (Black-Box Shift Estimation)
- Saerens et al. (2002): "Adjusting the Outputs of a Classifier to New a Priori Probabilities"

---

## 12. Glossary

- **1:3 sampling**: Each batch contains 1 positive for every 3 negatives (1:3 pos:neg ratio)
- **DA (Distribution Alignment)**: Calibrate predictions to match target prior via bias + threshold
- **Prior shift**: Training data has different positive rate than test data
- **Bias**: Scalar added to logits to shift prediction distribution
- **Threshold search**: Find decision boundary that maximizes validation metric
- **Epoch (with sampling)**: One pass through all positives + sampled negatives
- **Target prior**: Desired positive rate for test distribution (0.5 for balanced)
- **Calibrated predictions**: Predictions after applying DA bias + threshold

---

## 13. Implementation Clarifications (Resolved)

### Q1: Embedding format compatibility
**Answer**: ✅ `.npz` format follows existing structure supported by `load_embeddings()`
- Keys: `['ids', 'embeddings', 'metadata', 'original_sequences', 'cleaned_sequences']`
- Embeddings shape: `(1, seq_len, 1536)` per protein
- Already handled by `_load_npz_embedding_dict()` in `data_io.py`
- **Action**: No changes needed to embedding loading logic

### Q2: Logging strategy for DA metrics
**Answer**: ✅ During finetune, **only log DA-calibrated metrics**, not raw
- Rationale: Simpler, avoids confusion about which metric to trust
- All validation metrics (AUROC, F1, recall) are DA-calibrated
- `training_step.csv` columns: `Epoch, Epoch Time, Train Loss, Val Loss, Val AUROC, Val F1, DA_Bias, DA_Threshold, Learning Rate`
- **Action**: Finetune stage computes only DA metrics, no raw validation pass

### Q3: Pretrain best model selection
**Answer**: ✅ Use `val_loss` (existing behavior, no DA in pretrain)
- **Action**: No change to pretrain checkpointing logic

### Q4: Early stopping with DA
**Answer**: ✅ Yes, early stopping checks DA-calibrated metric during finetune
- Monitor metric: DA-calibrated F1 (configurable via `finetune_config.monitor_metric`)
- **Action**: Early stopping in finetune uses DA metrics from `da_result['metrics']`

### Q5: Multi-GPU testing
**Answer**: ✅ Single-GPU testing is sufficient for MVP
- **Action**: Test on single GPU, DDP compatibility built-in via Option A sampling

---

**End of Roadmap 09**