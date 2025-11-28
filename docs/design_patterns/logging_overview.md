# Logging and Artifacts

This document details the logging strategy and artifact structure for DIPPI. It serves as a reference for understanding where training metrics, evaluation results, and model checkpoints are stored.

## Directory Layout

All artifacts are stored under the `logs/` and `models/` directories, organized by model architecture and run ID.

### Logging (`logs/`)

*   **Pretrain Logs**: `logs/{model}/pretrain/<run_id>/`
*   **Finetune Logs**: `logs/{model}/finetune/<run_id>/`
*   **Evaluation Logs**: `logs/{model}/evaluate/<run_id>/`

**Note**: The `<run_id>` is either provided in the config or automatically generated (timestamped) by the orchestrator.

### Checkpoints (`models/`)

*   **Pretrain Models**: `models/{model}/pretrain/<run_id>/`
*   **Finetune Models**: `models/{model}/finetune/<run_id>/`

## Artifact Types

### 1. `log.log`
*   **Location**: Inside any run directory.
*   **Content**: Human-readable console output, including configuration summaries, training progress bars (text format), and system messages.

### 2. `training_step.csv`
*   **Location**: Pretrain and Finetune directories.
*   **Role**: structured time-series data for training curves.
*   **Schema**:
    *   `Epoch`: Integer epoch number.
    *   `Epoch Time`: Duration of the epoch in seconds.
    *   `Train Loss`: Average training loss.
    *   `Val Loss`: Average validation loss.
    *   `Train {Metric}`: Monitored training metrics (e.g., `Train AUROC`).
    *   `Val {Metric}`: Monitored validation metrics (e.g., `Val AUROC`).
    *   `Learning Rate`: Current learning rate.

### 3. `evaluate.csv`
*   **Location**: `logs/{model}/evaluate/<run_id>/`
*   **Role**: Final performance report for a model on a test set.
*   **Schema**: dynamic columns matching the metrics requested in `evaluate.metrics` config (e.g., `auroc`, `f1`, `accuracy`, `precision`, `recall`).

### 4. `best_model.pth`
*   **Location**: `models/{model}/{stage}/{run_id}/`
*   **Role**: The saved state dictionary of the model achieving the best performance on the monitored metric.

## Checkpoint Policy

The orchestrator controls when checkpoints are saved based on the `run_config.save_best_only` setting:
*   **`true`**: Only the single best checkpoint is kept (`best_model.pth`).
*   **`false`**: A checkpoint is saved at the end of every epoch (e.g., `checkpoint_epoch_01.pth`), in addition to `best_model.pth`.
