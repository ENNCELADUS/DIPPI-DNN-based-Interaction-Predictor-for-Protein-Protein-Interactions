# Model Architecture Standards

This document defines the architectural standards for implementing models in the DIPPI system. The goal is to maintain a modular, predictable, and clean codebase where new architectures can be added without modifying the core orchestration logic.

## Design Philosophy

1.  **Isolation**: Each model architecture resides in its own independent file within `src/model/`.
2.  **Standard Contract**: Models must adhere to the standard PyTorch `nn.Module` interface. No custom base classes or complex inheritance hierarchies are enforced.
3.  **Config Injection**: Models receive only the configuration parameters relevant to them, filtered by the orchestrator.

## Implementation Standards

### 1. File Structure
*   **Path**: `src/model/{model_name}.py`
*   **Content**: A single class inheriting from `nn.Module` (e.g., `class V3(nn.Module): ...`).
*   **Helper Modules**: Architecture-specific sub-modules (blocks, layers) should be defined within the same file or a private utility, keeping the public namespace clean.

### 2. The Contract (`nn.Module`)

Models must implement:

*   **`__init__(self, **kwargs)`**:
    *   Accepts configuration parameters as keyword arguments.
    *   Initializes all layers and sub-modules.
*   **`forward(self, **batch)`**:
    *   Accepts input data (typically a dictionary or named arguments matching the data loader output).
    *   Returns a dictionary containing at least the `loss` (during training) and `logits`/`probs` (for evaluation).

### 3. Configuration Handling

*   **Extraction**: The orchestrator (`run.py`) uses `src/utils/config.py` to extract model-specific parameters from the global config.
*   **Injection**: These parameters are passed to the model's `__init__`.
*   **Strictness**: Models should not be aware of the global config structure. They only know about their specific hyperparameters (e.g., `d_model`, `n_layers`, `dropout`).

## Example Implementation

```python
# src/model/v3.py
import torch.nn as nn

class V3(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, labels=None):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.BCEWithLogitsLoss()(logits, labels)
            
        return output
```

## Instantiation Logic

The orchestrator (`run.py`) handles model selection using simple branching, avoiding complex registries for the MVP phase:

```python
# Conceptual Orchestrator Logic
def build_model(cfg):
    name = cfg.model_config.model
    kwargs = extract_model_kwargs(cfg, name)
    
    if name == "v3":
        return V3(**kwargs)
    elif name == "tuna":
        return TUnA(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
```
