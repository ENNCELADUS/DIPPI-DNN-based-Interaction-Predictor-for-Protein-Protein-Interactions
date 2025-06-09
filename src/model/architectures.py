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


class SimplifiedProteinClassifier(BaseProteinModel):
    """
    Simplified neural network for protein-protein interaction prediction.
    Uses average pooling for sequence aggregation and simple MLP layers.
    """
    
    def __init__(self, input_dim=960, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Protein sequence encoder
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction prediction head
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass for protein pair interaction prediction.
        
        Args:
            emb_a: (B, L_a, 960) protein A embeddings
            emb_b: (B, L_b, 960) protein B embeddings  
            lengths_a: (B,) actual lengths for protein A
            lengths_b: (B,) actual lengths for protein B
            
        Returns:
            logits: (B, 1) interaction prediction logits
        """
        device = emb_a.device
        
        # Create masks for variable length sequences
        mask_a = torch.arange(emb_a.size(1), device=device).unsqueeze(0) < lengths_a.unsqueeze(1)
        mask_b = torch.arange(emb_b.size(1), device=device).unsqueeze(0) < lengths_b.unsqueeze(1)
        
        # Average pooling with mask
        emb_a_avg = (emb_a * mask_a.unsqueeze(-1).float()).sum(dim=1) / lengths_a.unsqueeze(-1).float()
        emb_b_avg = (emb_b * mask_b.unsqueeze(-1).float()).sum(dim=1) / lengths_b.unsqueeze(-1).float()
        
        # Encode individual proteins
        enc_a = self.protein_encoder(emb_a_avg)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_avg)  # (B, hidden_dim)
        
        # Combine protein representations and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)      # (B, 1)
        
        return logits


class AttentionProteinClassifier(BaseProteinModel):
    """
    Attention-based protein-protein interaction classifier.
    Uses self-attention for sequence aggregation instead of simple pooling.
    """
    
    def __init__(self, input_dim=960, hidden_dim=256, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Project embeddings to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention for sequence aggregation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Protein sequence encoder
        self.protein_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction prediction head
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def _apply_attention_pooling(self, embeddings, lengths):
        """Apply self-attention based pooling for variable length sequences."""
        batch_size, max_len, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Create attention mask for padding
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Apply self-attention
        attended, _ = self.self_attention(
            embeddings, embeddings, embeddings, 
            key_padding_mask=mask
        )
        
        # Average pooling over attended representations (excluding padding)
        lengths_expanded = lengths.unsqueeze(-1).float()
        mask_expanded = (~mask).unsqueeze(-1).float()
        
        pooled = (attended * mask_expanded).sum(dim=1) / lengths_expanded
        
        return pooled
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass with attention-based sequence aggregation.
        
        Args:
            emb_a: (B, L_a, 960) protein A embeddings
            emb_b: (B, L_b, 960) protein B embeddings  
            lengths_a: (B,) actual lengths for protein A
            lengths_b: (B,) actual lengths for protein B
            
        Returns:
            logits: (B, 1) interaction prediction logits
        """
        # Project to hidden dimension
        emb_a_proj = self.input_projection(emb_a)  # (B, L_a, hidden_dim)
        emb_b_proj = self.input_projection(emb_b)  # (B, L_b, hidden_dim)
        
        # Apply attention-based pooling
        emb_a_pooled = self._apply_attention_pooling(emb_a_proj, lengths_a)  # (B, hidden_dim)
        emb_b_pooled = self._apply_attention_pooling(emb_b_proj, lengths_b)  # (B, hidden_dim)
        
        # Encode proteins
        enc_a = self.protein_encoder(emb_a_pooled)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_pooled)  # (B, hidden_dim)
        
        # Combine protein representations and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)      # (B, 1)
        
        return logits


# Model registry for easy instantiation
MODEL_REGISTRY = {
    'simplified': SimplifiedProteinClassifier,
    'attention': AttentionProteinClassifier,
}


def create_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def get_available_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys()) 