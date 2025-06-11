"""
Model architectures and definitions for protein-protein interaction prediction.

This module contains neural network architectures, model utilities, and 
model-related functions for the DIPPI project.
"""

# Import base architecture
from .architectures import BaseProteinModel

# Import registry system
from .registry import (
    create_model,
    get_available_models,
    get_model_class,
    model_registry,
    register_model
)

# TODO:Import all model files to ensure registration
# This triggers the @register_model decorators
from . import v1
# NOTE: v2.py doesn't exist, skipping for now

# Backward compatibility - you can still import specific models if needed
from .v1 import SimplifiedProteinClassifier
# NOTE: v2 models not available, skipping for now

# Export the main interfaces
__all__ = [
    'BaseProteinModel',
    'create_model',
    'get_available_models', 
    'get_model_class',
    'model_registry',
    'register_model',
    # Individual model classes for backward compatibility
    'SimplifiedProteinClassifier',
] 