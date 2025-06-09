"""
Neural network architectures for protein-protein interaction prediction.

This module contains clean model definitions without training or evaluation logic.
All models should inherit from a base model class for consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseProteinModel(nn.Module, ABC):
    """
    Abstract base class for all protein interaction models.
    Provides common interface and utilities.
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass for protein pair interaction prediction.
        
        Args:
            emb_a: (B, L_a, D) protein A embeddings
            emb_b: (B, L_b, D) protein B embeddings  
            lengths_a: (B,) actual lengths for protein A
            lengths_b: (B,) actual lengths for protein B
            
        Returns:
            logits: (B,) interaction prediction logits
        """
        pass
    
    def get_model_info(self):
        """Return model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': str(self)
        }
    
    def initialize_weights(self):
        """Initialize model weights using best practices."""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Weight initialization for different layer types."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)