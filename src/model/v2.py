"""
Advanced model architectures (v2) for protein-protein interaction prediction.

This module contains more sophisticated model architectures including attention-based
models and other advanced techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .architectures import BaseProteinModel
from .registry import register_model


@register_model(
    'v2', 
    description='Attention-based protein-protein interaction classifier',
    features=['self_attention', 'multihead_attention'],
    complexity='medium'
)
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


@register_model(
    'multihead_attention', 
    description='Enhanced attention-based classifier with multiple attention layers',
    features=['multi_layer_attention', 'transformer_encoder'],
    complexity='high'
)
class MultiHeadAttentionProteinClassifier(BaseProteinModel):
    """
    Enhanced attention-based classifier with multiple attention layers.
    Uses stack of self-attention layers for more sophisticated sequence modeling.
    """
    
    def __init__(self, input_dim=960, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Project embeddings to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
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
    
    def _apply_multi_attention_pooling(self, embeddings, lengths):
        """Apply multiple self-attention layers for sequence aggregation."""
        batch_size, max_len, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Create attention mask for padding
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Apply stack of attention layers
        attended = embeddings
        for attention_layer in self.attention_layers:
            attended = attention_layer(attended, src_key_padding_mask=mask)
        
        # Average pooling over attended representations (excluding padding)
        lengths_expanded = lengths.unsqueeze(-1).float()
        mask_expanded = (~mask).unsqueeze(-1).float()
        
        pooled = (attended * mask_expanded).sum(dim=1) / lengths_expanded
        
        return pooled
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass with multi-layer attention sequence aggregation.
        
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
        
        # Apply multi-layer attention pooling
        emb_a_pooled = self._apply_multi_attention_pooling(emb_a_proj, lengths_a)  # (B, hidden_dim)
        emb_b_pooled = self._apply_multi_attention_pooling(emb_b_proj, lengths_b)  # (B, hidden_dim)
        
        # Encode proteins
        enc_a = self.protein_encoder(emb_a_pooled)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_pooled)  # (B, hidden_dim)
        
        # Combine protein representations and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)      # (B, 1)
        
        return logits


@register_model(
    'cross_attention', 
    description='Cross-attention based classifier for protein pair interaction modeling',
    features=['cross_attention', 'protein_pair_modeling'],
    complexity='high'
)
class CrossAttentionProteinClassifier(BaseProteinModel):
    """
    Cross-attention based classifier that models interactions between protein pairs.
    Uses cross-attention to explicitly model protein-protein interactions.
    """
    
    def __init__(self, input_dim=960, hidden_dim=256, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Project embeddings to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention for individual proteins
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between proteins
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Protein sequence encoders
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
    
    def _apply_self_attention_pooling(self, embeddings, lengths):
        """Apply self-attention pooling for individual proteins."""
        batch_size, max_len, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Create attention mask for padding
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Apply self-attention
        attended, _ = self.self_attention(
            embeddings, embeddings, embeddings, 
            key_padding_mask=mask
        )
        
        # Average pooling
        lengths_expanded = lengths.unsqueeze(-1).float()
        mask_expanded = (~mask).unsqueeze(-1).float()
        pooled = (attended * mask_expanded).sum(dim=1) / lengths_expanded
        
        return pooled
    
    def _apply_cross_attention_pooling(self, emb_a, emb_b, lengths_a, lengths_b):
        """Apply cross-attention between protein pairs."""
        device = emb_a.device
        
        # Create masks
        mask_a = torch.arange(emb_a.size(1), device=device).unsqueeze(0) >= lengths_a.unsqueeze(1)
        mask_b = torch.arange(emb_b.size(1), device=device).unsqueeze(0) >= lengths_b.unsqueeze(1)
        
        # Cross-attention: A attends to B
        attended_a, _ = self.cross_attention(emb_a, emb_b, emb_b, key_padding_mask=mask_b)
        
        # Cross-attention: B attends to A  
        attended_b, _ = self.cross_attention(emb_b, emb_a, emb_a, key_padding_mask=mask_a)
        
        # Average pooling
        lengths_a_expanded = lengths_a.unsqueeze(-1).float()
        lengths_b_expanded = lengths_b.unsqueeze(-1).float()
        mask_a_expanded = (~mask_a).unsqueeze(-1).float()
        mask_b_expanded = (~mask_b).unsqueeze(-1).float()
        
        pooled_a = (attended_a * mask_a_expanded).sum(dim=1) / lengths_a_expanded
        pooled_b = (attended_b * mask_b_expanded).sum(dim=1) / lengths_b_expanded
        
        return pooled_a, pooled_b
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass with cross-attention between protein pairs.
        
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
        
        # Apply self-attention first
        emb_a_self = self._apply_self_attention_pooling(emb_a_proj, lengths_a)
        emb_b_self = self._apply_self_attention_pooling(emb_b_proj, lengths_b)
        
        # Apply cross-attention
        emb_a_cross, emb_b_cross = self._apply_cross_attention_pooling(emb_a_proj, emb_b_proj, lengths_a, lengths_b)
        
        # Combine self and cross-attended representations
        emb_a_combined = emb_a_self + emb_a_cross
        emb_b_combined = emb_b_self + emb_b_cross
        
        # Encode proteins
        enc_a = self.protein_encoder(emb_a_combined)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_combined)  # (B, hidden_dim)
        
        # Combine protein representations and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)      # (B, 1)
        
        return logits