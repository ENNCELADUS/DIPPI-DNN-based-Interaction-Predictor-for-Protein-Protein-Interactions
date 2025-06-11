#!/usr/bin/env python3
"""
Standalone No-Adapter PPI Classifier
Direct connection from v2 MAE (512-dim) to downstream components without adapter layer
Contains all necessary classes without external dependencies from v2 or v5 modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# LEAN V2 ENCODER - Only the parts we actually use
# ============================================================================

class V2Encoder(nn.Module):
    """Lean version of v2 MAE encoder containing only the components we actually use"""
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 num_layers=4,
                 nhead=16,
                 ff_dim=2048,
                 max_len=1502):
        super().__init__()
        
        # Only the components we actually use
        self.embed = nn.Linear(input_dim, embed_dim)  # (B,L,960) → (B,L,512)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))  # (1,1502,512)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, lengths):
        """
        Forward pass without masking or reconstruction
        
        Args:
            x: (B, L, 960) - protein embeddings
            lengths: (B,) - sequence lengths
            
        Returns:
            encoder_output: (B, L, 512) - encoded representations
        """
        B, L, _ = x.shape
        device = x.device
        
        # Embed and add positional encoding
        x_emb = self.embed(x) + self.pos_embed[:, :L]  # (B, L, 512)
        
        # Create padding mask
        pad_mask = torch.arange(L, device=device).expand(B, L) >= lengths.unsqueeze(1)
        
        # Pass through transformer encoder
        encoder_output = self.encoder(x_emb, src_key_padding_mask=pad_mask)  # (B, L, 512)
        
        return encoder_output

# ============================================================================
# COMPATIBILITY LAYER - For loading pretrained v2 MAE weights
# ============================================================================

class TransformerMAE(nn.Module):
    """
    Minimal compatibility wrapper for loading pretrained v2 MAE weights.
    Contains only what's needed for weight loading, then we extract the useful parts.
    """
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 mask_ratio=0.5,  # Ignored - kept for compatibility
                 num_layers=4,
                 nhead=16,
                 ff_dim=2048,
                 max_len=1502):
        super().__init__()
        
        # Core components (what we actually use)
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Unused components (kept only for weight loading compatibility)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.compress_head = nn.Linear(embed_dim, input_dim)
        
        # Note: We never call forward() on this class - it's just for weight loading
    
    def extract_encoder(self) -> V2Encoder:
        """Extract only the encoder components we need"""
        encoder = V2Encoder(
            input_dim=960,
            embed_dim=512,
            num_layers=4,
            nhead=16,
            ff_dim=2048,
            max_len=1502
        )
        
        # Copy only the useful weights
        encoder.embed.load_state_dict(self.embed.state_dict())
        encoder.pos_embed.data = self.pos_embed.data.clone()
        encoder.encoder.load_state_dict(self.encoder.state_dict())
        
        return encoder

# ============================================================================
# InteractionCrossAttention and InteractionMLPHead
# ============================================================================

class InteractionCrossAttention(nn.Module):
    """Cross-attention module for protein-protein interactions - updated for 512-dim v2 MAE"""
    def __init__(self, d_model=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Learnable CLS_int token
        self.cls_int = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_int, std=0.02)
    
    def forward(self, tok_a, tok_b, lengths_a=None, lengths_b=None):
        """
        tok_a: (B, La, 512) - encoded tokens from protein A (v2 MAE output)
        tok_b: (B, Lb, 512) - encoded tokens from protein B (v2 MAE output)
        lengths_a: (B,) - actual lengths of protein A sequences (optional)
        lengths_b: (B,) - actual lengths of protein B sequences (optional)
        Returns: (B, 512) - interaction vector
        """
        B = tok_a.shape[0]
        La, Lb = tok_a.shape[1], tok_b.shape[1]
        device = tok_a.device
        
        # Pad sequences to equal length within the batch for efficient attention
        max_len_a = La
        max_len_b = Lb
        
        # If sequences are already different lengths, pad the shorter ones
        if La != Lb:
            max_len = max(La, Lb)
            if La < max_len:
                pad_a = torch.zeros(B, max_len - La, self.d_model, device=device, dtype=tok_a.dtype)
                tok_a = torch.cat([tok_a, pad_a], dim=1)
                max_len_a = max_len
            if Lb < max_len:
                pad_b = torch.zeros(B, max_len - Lb, self.d_model, device=device, dtype=tok_b.dtype)
                tok_b = torch.cat([tok_b, pad_b], dim=1)
                max_len_b = max_len
        
        # Concatenate protein tokens
        protein_tokens = torch.cat([tok_a, tok_b], dim=1)  # (B, La+Lb, 512)
        
        # Add learnable CLS_int token
        cls_int = self.cls_int.expand(B, -1, -1)  # (B, 1, 512)
        
        # Combine: [CLS_int] + [protein_A_tokens] + [protein_B_tokens]
        combined = torch.cat([cls_int, protein_tokens], dim=1)  # (B, La+Lb+1, 512)
        
        # Create key_padding_mask if lengths are provided
        key_padding_mask = None
        if lengths_a is not None and lengths_b is not None:
            # Create mask for protein A: True = padding (should be ignored)
            mask_a = torch.arange(max_len_a, device=device).expand(B, max_len_a) >= lengths_a.unsqueeze(1)
            
            # Create mask for protein B: True = padding (should be ignored)  
            mask_b = torch.arange(max_len_b, device=device).expand(B, max_len_b) >= lengths_b.unsqueeze(1)
            
            # Concatenate masks for protein tokens
            protein_mask = torch.cat([mask_a, mask_b], dim=1)  # (B, La+Lb)
            
            # Add False for CLS token (CLS should never be masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B, 1)
            key_padding_mask = torch.cat([cls_mask, protein_mask], dim=1)  # (B, La+Lb+1)
        
        # Pass through cross-attention layers with padding mask
        for layer in self.cross_attn_layers:
            combined = layer(combined, src_key_padding_mask=key_padding_mask)
        
        # Extract and normalize the CLS_int token
        interaction_vector = self.norm(combined[:, 0])  # (B, 512)
        
        return interaction_vector

class InteractionMLPHead(nn.Module):
    """MLP head for final classification - updated for 512-dim v2 MAE"""
    def __init__(self, input_dim=512, hidden_dim1=256, hidden_dim2=64, output_dim=1, dropout=0.2):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        x: (B, 768) -> (B, 1)
        """
        return self.layers(x)

# ============================================================================
# V5.2 SPECIFIC CLASSES
# ============================================================================

class FrozenV2Encoder(nn.Module):
    """
    Loads pretrained v2 MAE and extracts only the encoder components we need.
    Much more memory efficient than loading the full MAE.
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
        
        # Step 1: Load full MAE temporarily for weight extraction
        temp_mae = TransformerMAE(
            input_dim=960, 
            embed_dim=512, 
            mask_ratio=0.0,  # Ignored
            num_layers=4, 
            nhead=16, 
            ff_dim=2048, 
            max_len=1502
        )
        
        # Step 2: Load pretrained weights into temporary MAE
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        temp_mae.load_state_dict(state_dict, strict=False)
        
        # Step 3: Extract only the lean encoder (discards decoder, compress_head, mask_token)
        self.encoder = temp_mae.extract_encoder()
        
        # Step 4: Freeze and cleanup
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        del temp_mae  # Free memory from unused components
        
        print(f"✅ Loaded lean v2 encoder from {ckpt_path} (decoder/compress_head discarded)")

    @torch.no_grad()
    def forward(self, x, lengths):
        """
        Forward pass using lean v2 encoder
        
        Args:
            x: (B, L, 960) - protein embeddings
            lengths: (B,) - sequence lengths
            
        Returns:
            encoder_output: (B, L, 512) - v2 encoder output
        """
        return self.encoder(x, lengths)

class PPIClassifier_NoAdapter(nn.Module):
    """PPI Classifier using pretrained v2 MAE without adapter (direct 512-dim connection)"""
    def __init__(self, v2_mae_path: str):
        super().__init__()
        
        # Pretrained v2 MAE encoder (frozen)
        self.encoder = FrozenV2Encoder(v2_mae_path)
        
        # Downstream components (trainable) - now using 512-dim directly
        self.cross_attn = InteractionCrossAttention(d_model=512, n_heads=8, n_layers=2, dropout=0.1)
        self.mlp_head = InteractionMLPHead(input_dim=512, hidden_dim1=256, hidden_dim2=64)
        
        print("🔧 No-Adapter Architecture:")
        print("  - Encoder: Pretrained v2 MAE (frozen, 512-dim)")
        print("  - Cross-attention: Trainable components (512-dim)")
        print("  - MLP head: Trainable components (512-dim)")
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass for PPI classification
        
        Args:
            emb_a: (B, L, 960) - protein A embeddings
            emb_b: (B, L, 960) - protein B embeddings
            lengths_a: (B,) - protein A sequence lengths
            lengths_b: (B,) - protein B sequence lengths
            
        Returns:
            logits: (B,) - classification logits
        """
        # Encode through pretrained v2 MAE (no adapter)
        tok_a = self.encoder(emb_a, lengths_a)  # (B, La, 512)
        tok_b = self.encoder(emb_b, lengths_b)  # (B, Lb, 512)
        
        # Cross-attention for interaction with padding mask
        z_int = self.cross_attn(tok_a, tok_b, lengths_a, lengths_b)  # (B, 512)
        
        # Final classification
        logits = self.mlp_head(z_int)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
    
    def get_interaction_embeddings(self, emb_a, emb_b, lengths_a, lengths_b):
        """Get interaction embeddings without final classification"""
        with torch.no_grad():
            tok_a = self.encoder(emb_a, lengths_a)
            tok_b = self.encoder(emb_b, lengths_b)
            z_int = self.cross_attn(tok_a, tok_b, lengths_a, lengths_b)
        return z_int

def create_ppi_classifier_no_adapter(v2_mae_path: str) -> PPIClassifier_NoAdapter:
    """
    Create PPI classifier using pretrained v2 MAE (no adapter)
    
    Args:
        v2_mae_path: Path to pretrained v2 MAE checkpoint
    
    Returns:
        PPIClassifier_NoAdapter model with direct 512-dim connection
    """
    model = PPIClassifier_NoAdapter(v2_mae_path)
    return model

# Backward compatibility alias
def create_ppi_classifier_v52(v2_mae_path: str) -> PPIClassifier_NoAdapter:
    """Backward compatibility alias"""
    return create_ppi_classifier_no_adapter(v2_mae_path)

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # Test the No-Adapter architecture
    print("🧪 Testing No-Adapter PPI Classifier architecture...")
    
    # TODO: Replace with your actual v2 MAE checkpoint path
    v2_mae_path = "models/mae_best_20250528-174157.pth"
    
    try:
        # Create model
        model = create_ppi_classifier_no_adapter(v2_mae_path)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        emb_a = torch.randn(batch_size, seq_len, 960)
        emb_b = torch.randn(batch_size, seq_len, 960)
        lengths_a = torch.tensor([80, 100])
        lengths_b = torch.tensor([90, 100])
        
        print(f"\n🔄 Testing forward pass:")
        print(f"  Input shapes: emb_a={emb_a.shape}, emb_b={emb_b.shape}")
        
        # Forward pass
        logits = model(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Output logits shape: {logits.shape}")
        
        # Test interaction embeddings
        interaction_emb = model.get_interaction_embeddings(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Interaction embeddings shape: {interaction_emb.shape}")
        
        print("\n✅ No-Adapter architecture test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure to set the correct v2_mae_path") 