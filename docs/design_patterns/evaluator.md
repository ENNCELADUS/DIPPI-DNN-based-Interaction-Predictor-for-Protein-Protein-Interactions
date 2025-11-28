# Evaluator and Metrics

The **Evaluator** module (`src/evaluate/base.py`) is responsible for assessing model performance. It provides a consistent interface for computing metrics during both training validation and final testing.

## Core Responsibilities

**Does:**
*   **Metric Calculation**: Computes performance metrics (e.g., AUROC, Accuracy, F1) based on model outputs.
*   **Inference Pass**: Runs a single pass over a data loader to collect logits/probabilities and labels.
*   **Stateless Reporting**: Returns a simple dictionary of results (e.g., `{"val_loss": 0.5, "val_auroc": 0.85}`).

**Does NOT:**
*   **State Management**: It does *not* change the model's training mode. The orchestrator is responsible for setting `model.eval()` and `torch.no_grad()`.
*   **Logging/I/O**: It does *not* write to files or logs. It simply returns data to the orchestrator.
*   **Configuration**: It does *not* parse the global config. It receives a specific list of metrics to compute.

## Configuration Schema

Metrics are defined in the `evaluate` section of the YAML config.

```yaml
evaluate:
  metrics: [
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "mcc",
  ]
```

## Usage Pattern

### 1. In the Orchestrator (Run Loop)

The orchestrator manages the evaluation context:

```python
# Inside run.py
model.eval()
with torch.no_grad():
    # Evaluator performs the pass and computes metrics
    results = evaluator.evaluate(model, val_loader, device)

# Orchestrator handles logging
logger.log_metrics(results)
```

### 2. Standalone Evaluation (Eval Only Mode)

When running in `eval_only` mode, the orchestrator:
1.  Loads the configuration and metrics list.
2.  Instantiates the Evaluator.
3.  Loads the checkpoint.
4.  Runs the evaluation pass.
5.  Writes the result to `logs/{model}/evaluate/{run_id}/evaluate.csv`.

## Output Format

The Evaluator returns a dictionary mapping metric names to values. The orchestrator ensures these are logged consistently.

*   **Console/Log**: `INFO | Validation metrics: val_auroc=0.842, val_recall=0.877`
*   **CSV**:
    *   `training_step.csv`: Appends validation metrics alongside training loss.
    *   `evaluate.csv`: Stores the final evaluation results.
