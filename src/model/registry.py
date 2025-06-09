"""
Model registry system for automatic model registration and discovery.

This module provides a clean way to register models across different files
without having to manually import them in a central location.
"""

from typing import Dict, Type, Optional, Any
import inspect


class ModelRegistry:
    """
    A registry class that allows models to register themselves automatically.
    """
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a model class.
        
        Args:
            name: The name to register the model under
            metadata: Optional metadata about the model
            
        Returns:
            The decorator function
        """
        def decorator(model_class: Type):
            if name in self._models:
                raise ValueError(f"Model '{name}' is already registered")
            
            self._models[name] = model_class
            self._model_metadata[name] = metadata or {}
            
            # Add registration info to the class
            model_class._registry_name = name
            model_class._registry_metadata = metadata or {}
            
            return model_class
        return decorator
    
    def create(self, name: str, **kwargs):
        """
        Create a model instance by name.
        
        Args:
            name: The registered model name
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            An instance of the requested model
        """
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        
        model_class = self._models[name]
        return model_class(**kwargs)
    
    def get_model_class(self, name: str) -> Type:
        """Get the model class by name without instantiating it."""
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        return self._models[name]
    
    def list_models(self) -> list:
        """Return a list of all registered model names."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a registered model."""
        if name not in self._models:
            raise ValueError(f"Unknown model '{name}'")
        
        model_class = self._models[name]
        metadata = self._model_metadata[name].copy()
        
        # Add class information
        metadata.update({
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'docstring': inspect.getdoc(model_class),
        })
        
        return metadata
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered models."""
        return {name: self.get_model_info(name) for name in self._models.keys()}


# Global registry instance
model_registry = ModelRegistry()

# Convenience decorator using the global registry
def register_model(name: str, **metadata):
    """
    Convenience decorator for registering models with the global registry.
    
    Usage:
        @register_model('my_model', description='A custom model')
        class MyModel(BaseProteinModel):
            pass
    """
    return model_registry.register(name, metadata)


# Factory functions using the global registry
def create_model(name: str, **kwargs):
    """Create a model instance using the global registry."""
    return model_registry.create(name, **kwargs)


def get_available_models():
    """Get list of available model names from the global registry."""
    return model_registry.list_models()


def get_model_class(name: str):
    """Get model class by name from the global registry."""
    return model_registry.get_model_class(name) 