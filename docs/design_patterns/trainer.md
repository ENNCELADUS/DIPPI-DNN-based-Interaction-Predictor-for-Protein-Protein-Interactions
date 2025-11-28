# Trainer Design

The **Trainer** module is responsible for the core training loop execution. It is designed to be generic, handling one epoch of training at a time, while delegating policy decisions (like scheduling and unfreezing) to **Strategy** objects.

## Core Responsibilities

### The Trainer (`src/train/base.py`)

The `Trainer` class encapsulates the mechanics of updating the model weights.

**Does:**
*   **Execute `train_one_epoch(...)`**: Runs the standard forward-backward pass for a single epoch.
    *   Sets `model.train()`.
    *   Computes loss.
    *   Performs backpropagation (`loss.backward()`).
    *   Steps the optimizer (`optimizer.step()`) and then the scheduler (`scheduler.step()`).
*   **Manage Optimization State**: Builds and rebuilds the `Optimizer` and `Scheduler` on demand. This is critical for strategies that change trainable parameters (e.g., staged unfreezing).
*   **Handle Precision**: Wraps execution in `torch.autocast` and `torch.amp.GradScaler` when Mixed Precision (AMP) is enabled.
*   **Expose Hooks**: Provides access to `named_parameters` for strategies to manipulate gradients.

**Does NOT:**
*   **Validation**: Validation logic belongs to the `Evaluator`, called by the orchestrator.
*   **Logging**: Logging is handled by the orchestrator via `src/utils/logging.py`.
*   **Checkpointing**: File I/O for saving models is managed by the orchestrator.
*   **Global Config**: The Trainer receives only the specific configuration objects (optimizer cfg, scheduler cfg) it needs, not the entire global config.

### Strategies (`src/train/strategies.py`)

Strategies implement the "How" and "When" of training, particularly for complex finetuning protocols. They follow a callback-style pattern.

**Role:**
*   **Lifecycle Management**: Hooks into training events: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`.
*   **Parameter Control**: Decides which parameters are frozen (`requires_grad=False`) and which are trainable.
*   **Dynamic Optimization**: Triggers the Trainer to rebuild the optimizer and scheduler when the set of trainable parameters changes (e.g., during staged unfreezing).

**Example Strategy: `StagedUnfreeze`**
1.  **Start**: Freezes all layers except the head.
2.  **Epoch N**: Unfreezes the encoder.
3.  **Action**: Calls `trainer.rebuild_optimizer_and_scheduler()` to register the newly trainable parameters with the optimizer.

## Training Loop Logic

Inside `train_one_epoch`:

1.  **Zero Gradients**: `optimizer.zero_grad(set_to_none=True)` (Best practice for performance).
2.  **Forward Pass**: Compute model output and loss.
    *   *With AMP*: Inside `torch.autocast`.
3.  **Backward Pass**:
    *   *With AMP*: `scaler.scale(loss).backward()`, then `scaler.step(optimizer)` and `scaler.update()`.
    *   *Standard*: `loss.backward()`, then `optimizer.step()`.
4.  **Scheduler Step**: Called **after** the optimizer step.
5.  **Return**: A lightweight dictionary of statistics (e.g., `{"loss": avg_loss, "lr": current_lr}`).

## Architecture

```
src/train/
├── base.py          # Generic Trainer class
└── strategies.py    # Pluggable strategies (StagedUnfreeze, etc.)
```
