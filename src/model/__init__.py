"""
Model architectures and definitions for protein-protein interaction prediction.

This module contains neural network architectures, model utilities, and 
model-related functions for the DIPPI project.
"""

# Import model architectures and utilities
from .architectures import (
    BaseProteinModel,
    SimplifiedProteinClassifier,
    AttentionProteinClassifier,
    create_model,
    get_available_models,
    MODEL_REGISTRY
) 