# Top-to-bottom Implementation Plan for MVP (Compatible for V3 and TUnA Models)

This plan follows the design docs to build an MVP deep learning pipeline. It ensures compatibility for V3 and TUnA models via config-driven selection, with a single generic Trainer and Evaluator. Debugging happens in submodules without altering `run.py` once defined.

## 1. Key Responsibilities and Boundaries

- **run.py (Orchestrator)**: Parses config, sets up seed/device/DDP, builds model/data loaders, orchestrates pretrain/finetune/eval loops (calling Trainer/Evaluator/Logger), handles checkpointing/early stopping/logging; owns global flow but delegates all computation/logging to submodules.
- **Model Modules**: Define per-architecture nn.Module classes (e.g., `src/model/v3.py`, `src/model/tuna.py`) with `__init__` and `forward`; extract only relevant config keys via `utils/config.py` for construction; no training/eval logic.
- **Trainer (+ Strategy)**: Generic `Trainer` in `src/train/base.py` handles one-epoch training (forward/backward/optim step) and rebuilds optimizer/scheduler on demand; pluggable Strategy in `src/train/strategies.py` defines finetune policies (e.g., staged unfreeze) via hooks; no validation/logging/checkpointing.
- **Evaluator**: Generic class in `src/evaluate/base.py` computes losses/metrics over a dataloader in eval mode (returns dict of results); consumes metrics list from config; no model state changes or logging.
- **Logging Utils**: In `src/utils/logging.py`, appends structured rows to CSVs (e.g., `append_row(path, row_dict)`) and writes INFO lines to `log.log`; unifies pretrain/finetune/eval logging; no computation or decisions.

## 2. Concrete Implementation Plan

1. **Create `src/run.py` skeleton**: Define `main(cfg)` to parse full config, set seed/device/DDP, build model/loaders, and outline pretrain/finetune/eval loops with placeholders for Trainer/Evaluator calls; keep this file stable for the rest of implementation.
2. **Implement config parsing in `src/utils/config.py`**: ✅ 
   - **Public API** (call from `run.py`):
     - `load_config(path: str) -> TrackedConfig`: Load YAML config and wrap for tracking
     - `extract_keys(cfg: TrackedConfig, section: str) -> dict`: Extract section by path (e.g., `"run_config"`, `"model_config.v3"`), return flattened dict, mark as accessed
     - `enforce_used_keys(cfg: TrackedConfig, used_paths: List[str] = None) -> None`: Validate all keys consumed; raises `ValueError` if unused keys remain
   - **Usage Pattern**:
     ```python
     cfg = load_config("configs/v3.yaml")
     run_cfg = extract_keys(cfg, "run_config")          # → {"seed": 42, "mode": "full_pipeline", ...}
     model_params = extract_keys(cfg, "model_config.v3")  # → {"d_model": 384, ...}
     # ... extract other sections progressively
     enforce_used_keys(cfg)  # Raise if typos/unused params
     ```

3. **Add model selection in `src/run.py`**: In `build_model(cfg)`, use if-elif on `cfg.model_config.model` to call constructors like `V3(**extract_keys(cfg, "v3"))` from `src/model/v3.py` and `TUnA(**extract_keys(cfg, "tuna"))` from `src/model/tuna.py`.
4. **Model classes**
   - Create `src/model/v3.py` with `class V3(nn.Module)` that exposes:
     - Public constructor (kwargs provided by orchestrator; no config reads inside the model):
       - `input_dim: int`
       - `d_model: int`
       - `encoder_layers: int`
       - `cross_attn_layers: int`
       - `n_heads: int`
       - `mlp_head: dict` with:
         - `hidden_dims: list[int]` (required)
         - `dropout: float` (required)
         - `activation: str` = `"gelu"` (optional)
         - `norm: str` = `"layernorm"` (optional)
       - `regularization: dict` with:
         - `dropout: float` (required; encoder blocks)
         - `cross_attention_dropout: float` (optional; defaults to `dropout`)
         - `token_dropout: float` (optional; defaults to `0.0`)
     - Forward API:
       - `forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]`
       - Inputs: `emb_a`, `emb_b` (required), optional `len_a`, `len_b` (defaults to full lengths). Any `dist_*` keys are ignored.
       - Output: `{"logits": Tensor[B, 1]}` only (no aux/IO).
   - Create `src/model/tuna.py` with `class TUnA(nn.Module)` that exposes:
     - Public constructor (kwargs provided by orchestrator; no config reads inside the model):
       - `input_dim: int` (protein embedding dimension, e.g., 1536 for ESM3)
       - `d_model: int` (hidden dimension in transformer layers)
       - `intra_layers: int` (number of intra-protein encoder layers)
       - `inter_layers: int` (number of inter-protein encoder layers)
       - `n_heads: int` (number of attention heads, must divide `d_model`)
       - `ff_dim: int` (feedforward dimension, typically 2-4x `d_model`)
       - `dropout: float` (dropout probability in transformer layers)
       - `activation: str` 
       - Optional unused fields for config compatibility:
         - `spectral_norm: bool`
         - `gp_layer: dict` (Gaussian Process config, unused in MVP implementation)
     - Forward API:
       - `forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]`
       - Inputs: `emb_a`, `emb_b` (required), optional `len_a`, `len_b` (defaults to full lengths). Any other keys are ignored.
       - Output: `{"logits": Tensor[B, 1]}` only (no aux/IO).
     - Architecture: IntraEncoder (self-attention per protein) → InterEncoder (bidirectional AB+BA) → max pooling → linear head
     - Note: Original TUnA used GP layer for uncertainty; MVP uses simple linear head for logits.
5. **Implement data loading in `src/utils/data_io.py`**: ✅ Define `build_loader(csv_path: str, embeddings_path: str, batch_size: int, max_len: int, dtype: str, ddp: bool = False, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True, strip_cls_eos: bool = True) -> DataLoader` returning DataLoader with custom Dataset for protein pairs; call from `run.py` for train/val/test loaders.
6. **Define generic Trainer in `src/train/base.py`**: Implement `class Trainer` with public interfaces:
   ```python
   def __init__(self, model, device, optimizer_cfg, scheduler_cfg=None, amp_cfg=None, 
                strategy=None, grad_accum_steps=1, max_norm=None)
   def build_optimizer(self) -> torch.optim.Optimizer
   def build_scheduler(self, optimizer) -> Optional[lr_scheduler._LRScheduler]
   def rebuild_optimizer_and_scheduler(self) -> None
   def train_one_epoch(self, loader) -> Dict[str, float]  # Returns {"loss": float, "lr": float}
   ```
7. **Add Strategy base in `src/train/strategies.py`**: ✅
   - **Public API**:
     ```python
     class BaseStrategy:
         def on_train_begin(self, trainer): pass
         def on_epoch_begin(self, trainer, epoch_idx: int): pass
         def on_epoch_end(self, trainer, epoch_idx: int): pass
     
     class StagedUnfreeze(BaseStrategy):
         def __init__(self, schedule: List[Dict[str, Any]])
         # schedule entry: {"at_epoch": int, "freeze": [...], "unfreeze": [...], 
         #                  "optimizer_cfg": dict (optional), "scheduler_cfg": dict (optional)}
     ```
   - Substring pattern matching; freeze→unfreeze order; calls `trainer.rebuild_optimizer_and_scheduler()` when needed.
8. **Integrate Trainer into `run.py` loops**: In pretrain/finetune sections, instantiate Trainer with stage-specific config, call `train_one_epoch`, then Evaluator, then logging/checkpointing/early stopping.
9. **Implement Evaluator in `src/evaluate/base.py`**: ✅
   - **Public API**:
     ```python
     class Evaluator:
         def __init__(self, metrics_list: List[str], threshold: float = 0.5):
         def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
     ```
10. **Add checkpointing in `src/utils/checkpoint.py`**: ✅
   - `save_checkpoint(model, epoch, path, include_optim=False, optimizer=None, extra=None) -> Optional[str]`
   - `maybe_save_best(model, epoch, current_metric, best_so_far, mode, best_path, include_optim=False, optimizer=None, extra=None) -> Tuple[bool, float]`
   - `load_checkpoint(model, ckpt_path, map_location="cpu", strict=True, optimizer=None, load_optim=False, weights_only=False) -> Dict[str, Any]`
   - Role: stateless helpers for PyTorch save/load; DDP-safe (rank 0 writes); orchestrator (`run.py`) owns metric comparison and paths.
11. **Implement early stopping in `src/utils/early_stop.py`**: ✅
   - `check_early_stop(metrics: Sequence[float], patience: int, monitor: str, mode: str = "min", min_delta: float = 0.0) -> bool`
   - Role: pure stateless checker; returns True if training should stop based on metric history and patience threshold. Called by `run.py` in training loops after each validation epoch.
12. **Implement `src/utils/device.py` and `src/utils/distributed.py`**: ✅
   - **`src/utils/device.py` — Device Selection (Pure)**
     ```python
     def get_device(device_cfg: Union[str, dict], prefer_local_rank: bool = True) -> torch.device:
     ```
   - **`src/utils/distributed.py` — DDP Bootstrap/Cleanup (Pure)**
     ```python
     def init_if_enabled(ddp_cfg: dict, device: torch.device) -> bool:
     def barrier() -> None:
         """Synchronize all processes (no-op when not initialized)."""
     def cleanup() -> None:
         """Destroy process group (call once at end of main; no-op when not initialized)."""
     ```
13. **Unify logging in `src/utils/logging.py`**: `append_row(csv_path: Union[str, Path], row_dict: Mapping[str, object], columns: Optional[Sequence[str]] = None) -> None` for CSV metrics; called by `run.py` per epoch with training/validation stats per `logging_overview.md`.
14. **Handle any TODO marks in `run.py`**
15. Create `src/stages/` to house the three stage runners: `run_pretrain()`, `run_finetune()`, `run_evaluation()`.
16. **Test end-to-end**: Wire minimal config, run smoke test with small data/epochs, verify CSVs/logs without changing `run.py`.

## 3. Minimal CSV/Log Files Produced

- logs/<model>/pretrain/<run_id>/training_step.csv: Epoch,Epoch Time,Train Loss,Val Loss,Train <primary>,Train <secondary>,Val <primary>,Val <secondary>,Learning Rate
- logs/<model>/pretrain/<run_id>/log.log: INFO lines for epochs/validation/checkpoints
- logs/<model>/finetune/<run_id>/training_step.csv: Same schema as pretrain
- logs/<model>/finetune/<run_id>/log.log: INFO lines for epochs/validation/checkpoints
- logs/<model>/evaluate/<run_id>/evaluate.csv: Columns from evaluate.metrics (e.g., auroc,auprc,accuracy,...) + loss if computed
- logs/<model>/evaluate/<run_id>/log.log: INFO lines for evaluation