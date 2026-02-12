# Pipeline Architecture

The DIPPI pipeline is a centralized, config-driven orchestration system controlled by `src/run.py` and launched via shell scripts (current HPC launcher: `scripts/v3.sh`). It enforces strict separation of concerns between orchestration, training, evaluation, and model definition.

## Core Philosophy

1. **Centralized Orchestration**: The `run.py` module acts as the chief orchestrator. It manages the global state (configuration, seeds, devices) and drives the execution flow. Cross-module interactions are mediated by the orchestrator, not by direct calls between components.
2. **Config-Driven Execution**: All behaviors—model hyperparameters, training duration, optimization strategies, and data paths—are defined in a YAML configuration file.
3. **Stage-Based Workflow**: The pipeline supports pretrain, finetune, and evaluate stages, selected by run mode.

## Pipeline Stages

### 1. Setup & Initialization

Before any training begins, the orchestrator performs the following:

*   **Config Loading**: Parses the YAML configuration using `src/utils/config.py`.
*   **Run ID Management**: Assigns unique IDs for pretrain, finetune, and eval runs. If not provided, timestamps are generated automatically. See [Logging Overview](logging_overview.md) for naming conventions.
*   **Device & Seeding**: Sets random seeds for reproducibility and initializes computation devices (CPU/GPU/DDP) via `src/utils/device.py` and `src/utils/distributed.py`.
*   **Data Loading**: Instantiates data loaders using `src/utils/data_io.py`.
*   **Stage Logging Bootstrap**: Creates stage loggers and artifact directories early so setup/runtime events are persisted in `log.log`.

### 2. Model Initialization

The orchestrator selects and instantiates the model architecture based on the `model_config` section.
*   The model class (e.g., `V3`, `TUnA`) is dynamically selected from `src/model/`.
*   Only the configuration parameters relevant to the specific architecture are passed to its constructor.

### 3. Train Stages (Pretrain / Finetune)

**Role**: Train the model from scratch (or initial weights) on a pretext task or base dataset.

**Workflow**:
1.  **Trainer Instantiation**: A generic `Trainer` is created with stage optimizer/scheduler config and centralized loss config (`training_config.loss`).
    *   **Strict Config**: The trainer receives only the configuration keys it actually uses.
2.  **Training Loop**:
    *   **Train Step**: The trainer executes `train_one_epoch(...)`. It returns training metrics (e.g., loss).
    *   **Validation Step**: The orchestrator calls the `Evaluator` to compute validation metrics (e.g., `val_loss`, `val_auprc`, `val_auroc`).
    *   **Logging**: Results are written to `log.log` and `training_step.csv` via `src/utils/logging.py`, including heartbeat progress controlled by `training_config.logging.heartbeat_every_n_steps`.
    *   **Checkpointing**: The orchestrator saves the best model (based on monitored metrics) or per-epoch snapshots.
    *   **Early Stopping**: Checked via `src/utils/early_stop.py`.
    *   **DDP Behavior**: rank 0 performs artifact writes; all ranks synchronize via barriers.

### 4. Evaluate Stage

**Role**: Assess the final model performance on test datasets.

**Workflow**:
1.  **Metric Selection**: Metrics are parsed from the `evaluate.metrics` config section.
2.  **Inference**: The `Evaluator` runs a validation-like pass in `eval` mode with `torch.no_grad()`.
3.  **Result Logging**: Final metrics are appended to `logs/{model}/evaluate/{run_id}/evaluate.csv`.

## Artifact Contracts

* `training_step.csv` strict header order:
  * `Epoch,Epoch Time,Train Loss,Val Loss,Val <metric>...,Learning Rate`
  * `Val <metric>` columns follow `training_config.logging.validation_metrics` order.
* `evaluate.csv` strict header order:
  * `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`
* No `test_` prefixes are used in persisted eval CSV columns.

## Run Modes

The pipeline supports three strict execution modes defined in `run_config.mode`:

*   `full_pipeline`: Runs pretraining, then automatically loads the resulting best model for finetuning.
*   `train_only`: Runs pretraining and saves the best model. No input checkpoint required.
*   `eval_only`: Loads a checkpoint and runs the evaluation protocol.

## Launcher Disposition

* Canonical HPC launcher: `scripts/v3.sh`.
* `scripts/run_pipeline.sh` is retired and should not be referenced for new runs.
* Core orchestration remains in `src/run.py`; launcher scripts only invoke it.
