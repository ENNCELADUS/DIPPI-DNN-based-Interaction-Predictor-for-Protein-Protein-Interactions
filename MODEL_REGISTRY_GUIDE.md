# Model Registry System Guide

This guide explains how to organize and use a model registry when models are defined in different Python files.

## Overview

The model registry system allows you to:
- Register models across different files without manual imports
- Create models by name using a factory pattern
- Get metadata about available models
- Maintain clean separation of concerns

## File Structure

```
src/model/
├── __init__.py          # Main imports and exports
├── architectures.py     # Base model class and common utilities
├── registry.py          # Registry system implementation
├── v1.py               # Version 1 models
├── v2.py               # Version 2 models
└── ...                 # Additional model files
```

## How It Works

### 1. Registry System (`registry.py`)

The `ModelRegistry` class manages model registration and creation:

```python
from src.model.registry import register_model

@register_model('model_name', description='Model description', complexity='low')
class MyModel(BaseProteinModel):
    def __init__(self, **kwargs):
        # Model implementation
        pass
```

### 2. Model Registration

Each model file registers its models using the `@register_model` decorator:

**v1.py:**
```python
from .registry import register_model
from .architectures import BaseProteinModel

@register_model(
    'v1', 
    description='Simplified neural network for protein-protein interaction prediction',
    features=['average_pooling', 'mlp_layers'],
    complexity='low'
)
class SimplifiedProteinClassifier(BaseProteinModel):
    # Implementation
    pass
```

**v2.py:**
```python
from .registry import register_model
from .architectures import BaseProteinModel

@register_model('v2', description='Attention-based classifier', complexity='medium')
class AttentionProteinClassifier(BaseProteinModel):
    # Implementation
    pass

@register_model('multihead_attention', description='Multi-layer attention', complexity='high')
class MultiHeadAttentionProteinClassifier(BaseProteinModel):
    # Implementation
    pass
```

### 3. Module Initialization (`__init__.py`)

The `__init__.py` file ensures all models are registered by importing the model files:

```python
# Import registry functions
from .registry import (
    create_model,
    get_available_models,
    get_model_class,
    model_registry
)

# Import model files to trigger registration
from . import v1
from . import v2
# Import additional model files as needed
```

## Usage Examples

### Basic Usage

```python
from src.model import create_model, get_available_models

# List available models
print("Available models:", get_available_models())
# Output: ['v1', 'v2', 'multihead_attention', 'cross_attention']

# Create models by name
simple_model = create_model('v1', input_dim=960, hidden_dim=128)
attention_model = create_model('v2', input_dim=960, hidden_dim=256, num_heads=8)
```

### Advanced Usage

```python
from src.model import model_registry, get_model_class

# Get detailed model information
for model_name in get_available_models():
    info = model_registry.get_model_info(model_name)
    print(f"Model: {model_name}")
    print(f"  Description: {info['description']}")
    print(f"  Complexity: {info['complexity']}")
    print(f"  Features: {info['features']}")

# Get model class without instantiating
ModelClass = get_model_class('v2')
custom_model = ModelClass(input_dim=512, hidden_dim=64)
```

### Model Metadata

You can include rich metadata when registering models:

```python
@register_model(
    'advanced_model',
    description='An advanced model with sophisticated features',
    features=['cross_attention', 'residual_connections', 'layer_norm'],
    complexity='high',
    paper_reference='Smith et al. 2024',
    recommended_use_cases=['large_datasets', 'complex_interactions']
)
class AdvancedModel(BaseProteinModel):
    pass
```

## Benefits

### 1. **Scalability**
- Easy to add new models without modifying existing code
- Models can be organized in separate files
- No need to manually maintain import lists

### 2. **Discoverability**
- Automatic model discovery
- Rich metadata support
- Easy to query available models and their capabilities

### 3. **Factory Pattern**
- Create models by name programmatically
- Useful for configuration-driven model selection
- Easy to integrate with experiment frameworks

### 4. **Maintainability**
- Clean separation of concerns
- Each model file is self-contained
- Consistent interface across all models

## Adding New Models

To add a new model:

1. **Create a new model file** (e.g., `v3.py`)
2. **Define your model** inheriting from `BaseProteinModel`
3. **Register the model** using the `@register_model` decorator
4. **Import the file** in `__init__.py` to trigger registration

```python
# src/model/v3.py
from .registry import register_model
from .architectures import BaseProteinModel

@register_model('v3', description='Next generation model', complexity='very_high')
class NextGenModel(BaseProteinModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Model implementation
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        # Forward pass implementation
        pass
```

```python
# src/model/__init__.py
# Add this line to trigger registration
from . import v3
```

## Best Practices

1. **Use descriptive model names** that indicate their purpose or version
2. **Include rich metadata** to help users understand model capabilities
3. **Follow consistent naming conventions** across model files
4. **Document model parameters** in the constructor docstrings
5. **Use semantic versioning** for model names when appropriate (e.g., 'v1', 'v2')

## Integration with Training Scripts

The registry system integrates seamlessly with training and evaluation scripts:

```python
# config.yaml
model:
  name: "v2"
  params:
    input_dim: 960
    hidden_dim: 256
    num_heads: 8

# train.py
from src.model import create_model

def main(config):
    model = create_model(
        config['model']['name'], 
        **config['model']['params']
    )
    # Training logic
```

This approach provides a clean, scalable way to manage multiple model architectures across different files while maintaining ease of use and discoverability. 