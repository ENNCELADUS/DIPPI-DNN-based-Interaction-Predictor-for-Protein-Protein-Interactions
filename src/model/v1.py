import torch
import torch.nn as nn
import torch.nn.functional as F
from .architectures import BaseProteinModel
from .registry import register_model

@register_model(
    'v1', 
    description='Simplified neural network for protein-protein interaction prediction',
    features=['average_pooling', 'mlp_layers'],
    complexity='low'
)
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