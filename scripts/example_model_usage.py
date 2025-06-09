#!/usr/bin/env python3
"""
Example script demonstrating the model registry system.

This shows how to use the registry to create models from different files
without needing to manually import each model class.
"""

import torch
from src.model import (
    create_model, 
    get_available_models, 
    model_registry,
    get_model_class
)


def main():
    """Demonstrate the model registry functionality."""
    
    print("=== Model Registry Demo ===")
    
    # List all available models
    print(f"Available models: {get_available_models()}")
    
    # Get detailed information about all models
    print("\n=== Model Information ===")
    for model_name in get_available_models():
        info = model_registry.get_model_info(model_name)
        print(f"\nModel: {model_name}")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Features: {info.get('features', 'N/A')}")
        print(f"  Complexity: {info.get('complexity', 'N/A')}")
        print(f"  Class: {info.get('class_name', 'N/A')}")
    
    # Create models using the registry
    print("\n=== Creating Models ===")
    
    # Create a simplified model
    simple_model = create_model('simplified', input_dim=960, hidden_dim=128)
    print(f"Created simplified model: {simple_model.__class__.__name__}")
    print(f"Model info: {simple_model.get_model_info()}")
    
    # Create an attention model
    attention_model = create_model('attention', input_dim=960, hidden_dim=256, num_heads=4)
    print(f"\nCreated attention model: {attention_model.__class__.__name__}")
    
    # Test forward pass (with dummy data)
    print("\n=== Testing Forward Pass ===")
    batch_size = 2
    seq_len_a, seq_len_b = 100, 80
    input_dim = 960
    
    # Create dummy protein embeddings
    emb_a = torch.randn(batch_size, seq_len_a, input_dim)
    emb_b = torch.randn(batch_size, seq_len_b, input_dim)
    lengths_a = torch.tensor([seq_len_a, seq_len_a-20])
    lengths_b = torch.tensor([seq_len_b, seq_len_b-15])
    
    # Test the simplified model
    with torch.no_grad():
        output = simple_model(emb_a, emb_b, lengths_a, lengths_b)
        print(f"Simplified model output shape: {output.shape}")
        print(f"Sample output: {output[:2].flatten()}")
    
    # Get model class without instantiating
    print("\n=== Getting Model Classes ===")
    AttentionClass = get_model_class('attention')
    print(f"Retrieved class: {AttentionClass.__name__}")
    
    # You can also create custom instances
    custom_model = AttentionClass(input_dim=512, hidden_dim=128)
    print(f"Custom model created: {custom_model.__class__.__name__}")


if __name__ == "__main__":
    main() 