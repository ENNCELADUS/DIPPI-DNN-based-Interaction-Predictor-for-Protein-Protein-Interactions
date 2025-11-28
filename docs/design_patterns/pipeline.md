# Pipeline Architecture

The DIPPI pipeline is a centralized, config-driven orchestration system controlled by `run.py`-style entrypoints. It enforces strict separation of concerns between orchestration, training, evaluation, and model definition.

## Core Philosophy

1. **Centralized Orchestration**: The `run.py` module acts as the chief orchestrator. It manages the global state (configuration, seeds, devices) and drives the execution flow. Cross-module interactions are mediated by the orchestrator, not by direct calls between components.
2. **Config-Driven Execution**: All behaviors—model hyperparameters, training duration, optimization strategies, and data paths—are defined in a YAML configuration file.
3. **Stage-Based Workflow**: The pipeline explicitly supports three distinct stages: **Pretrain**, **Finetune**, and **Evaluate**.

## Pipeline Stages

### 1. Setup & Initialization

Before any training begins, the orchestrator performs the following:

*   **Config Loading**: Parses the YAML configuration using `src/utils/config.py`.
*   **Run ID Management**: Assigns unique IDs for pretrain, finetune, and eval runs. If not provided, timestamps are generated automatically. See [Logging Overview](logging_overview.md) for naming conventions.
*   **Device & Seeding**: Sets random seeds for reproducibility and initializes computation devices (CPU/GPU/DDP) via `src/utils/device.py` and `src/utils/distributed.py`.
*   **Data Loading**: Instantiates data loaders using `src/utils/data_io.py`.

### 2. Model Initialization

The orchestrator selects and instantiates the model architecture based on the `model_config` section.
*   The model class (e.g., `V3`, `TUnA`) is dynamically selected from `src/model/`.
*   Only the configuration parameters relevant to the specific architecture are passed to its constructor.

### 3. Pretrain Stage

**Role**: Train the model from scratch (or initial weights) on a pretext task or base dataset.

**Workflow**:
1.  **Trainer Instantiation**: A generic `Trainer` is created with the model and pretrain-specific optimizer/scheduler configs.
    *   **Strict Config**: The trainer receives only the configuration keys it actually uses.
2.  **Training Loop**:
    *   **Train Step**: The trainer executes `train_one_epoch(...)`. It returns training metrics (e.g., loss).
    *   **Validation Step**: The orchestrator calls the `Evaluator` to compute validation metrics (e.g., Val Loss, AUROC).
    *   **Logging**: Results are written to `log.log` and `training_step.csv` via `src/utils/logging.py`.
    *   **Checkpointing**: The orchestrator saves the best model (based on monitored metrics) or per-epoch snapshots.
    *   **Early Stopping**: Checked via `src/utils/early_stop.py`.

### 4. Finetune Stage

**Role**: Adapt a pretrained model to a specific downstream task, often with specialized strategies like freezing/unfreezing layers.

**Workflow**:
1.  **Trainer Re-instantiation**: A new `Trainer` instance is created for the `finetune` stage.
    *   **Checkpoint Loading**: If in `full_pipeline` mode, the best pretrain checkpoint is automatically loaded. Otherwise, an explicit path is required.
2.  **Strategy Initialization**: A `Strategy` object (e.g., `StagedUnfreeze`) is initialized to manage lifecycle hooks.
    *   **Freeze/Unfreeze**: The strategy applies freeze patterns (e.g., `encoder.*`) to `requires_grad`.
    *   **Optimizer/Scheduler Rebuild**: The strategy triggers the trainer to rebuild the optimizer and scheduler as parameters unfreeze.
3.  **Finetune Loop**:
    *   Identical structure to the Pretrain loop (Train -> Validate -> Log -> Checkpoint -> Early Stop).
    *   **Lifecycle Hooks**: The orchestrator invokes strategy hooks (e.g., `on_epoch_end`) to trigger staged unfreezing.

### 5. Evaluate Stage

**Role**: Assess the final model performance on test datasets.

**Workflow**:
1.  **Metric Selection**: Metrics are parsed from the `evaluate.metrics` config section.
2.  **Inference**: The `Evaluator` runs a validation-like pass in `eval` mode with `torch.no_grad()`.
3.  **Result Logging**: Final metrics are appended to `logs/{model}/evaluate/{run_id}/evaluate.csv`.

## Run Modes

The pipeline supports four strict execution modes defined in `run_config.mode`:

*   `pretrain_only`: Runs pretraining and saves the best model. No input checkpoint required.
*   `finetune_from_pretrain`: Loads a specific checkpoint (`load_checkpoint_path`) and runs finetuning.
*   `full_pipeline`: Runs pretraining, then automatically loads the resulting best model for finetuning.
*   `eval_only`: Loads a checkpoint and runs the evaluation protocol.
