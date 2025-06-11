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
# COPIED FROM v2.py - TransformerMAE class
# ============================================================================

class TransformerMAE(nn.Module):
    """Masked Autoencoder with Transformer - copied from v2.py"""
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 mask_ratio=0.5,
                 num_layers=4,
                 nhead=16,
                 ff_dim=2048,
                 max_len=1502):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.max_len = max_len

        # ---- embed & mask token & pos embed ----
        self.embed = nn.Linear(input_dim, embed_dim)  # (B,L,960)-->(B, L, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1,1,512)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))  # (1,1502,512)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- decoder head (MLP) ----
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim)
        )

        # ---- compression head: (embed_dim -> input_dim) ----
        self.compress_head = nn.Linear(embed_dim, input_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: Tensor (B, L, 960)
        lengths: Tensor (B,)  # 每个样本的有效长度（非 padding 部分）
        return:
          - recon: Tensor (B, L, 960)   # 重建整个序列
          - compressed: Tensor (B, 960) # 池化后的压缩向量
          - mask_bool: Tensor (B, L)    # True 表示该位置被掩码
        """
        device = x.device
        B, L, _ = x.shape

        # 1) 依据 lengths 构造 padding mask（True 表示该位置为 padding）
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        mask_pad = arange >= lengths.unsqueeze(1)                          # (B, L)

        # 2) 计算每个样本需要掩码的数量（非 padding 区域的 mask_ratio）
        len_per_sample = lengths
        num_mask_per_sample = (len_per_sample.float() * self.mask_ratio).long()  # (B,)

        # 3) 为所有样本生成随机噪声，并屏蔽 padding 为 +inf，排序后取前 num_mask_i
        noise = torch.rand(B, L, device=device)
        noise = noise.masked_fill(mask_pad, float("inf"))  # padding 位置永远不会被选中
        sorted_indices = torch.argsort(noise, dim=1)        # (B, L)

        # 4) 生成 boolean 掩码矩阵 mask_bool
        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)  # False = 不 mask
        for i in range(B):
            k = num_mask_per_sample[i].item()
            mask_bool[i, sorted_indices[i, :k]] = True

        # 5) 替换成 mask_token
        x_emb = self.embed(x)  # (B, L, E)
        x_emb = x_emb.masked_scatter(mask_bool.unsqueeze(-1), 
                                   self.mask_token.expand(B, L, -1)[mask_bool.unsqueeze(-1).expand(-1, -1, x_emb.size(-1))])

        # 6) 位置编码 & Transformer 编码器
        x_emb = x_emb + self.pos_embed[:, :L]
        enc_out = self.encoder(x_emb, src_key_padding_mask=mask_pad)

        # 7) 解码器重建
        recon = self.decoder(enc_out)  # (B, L, 960)

        # 8) 生成压缩向量
        compressed = self.compress_head(enc_out.mean(dim=1))  # (B, 960)

        return recon, compressed, mask_bool

# ============================================================================
# COPIED FROM v5.py - InteractionCrossAttention and InteractionMLPHead
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
    
    def forward(self, tok_a, tok_b):
        """
        tok_a: (B, La, 512) - encoded tokens from protein A (v2 MAE output)
        tok_b: (B, Lb, 512) - encoded tokens from protein B (v2 MAE output)
        Returns: (B, 512) - interaction vector
        """
        B = tok_a.shape[0]
        
        # Concatenate protein tokens directly (v2 MAE has no CLS tokens)
        protein_tokens = torch.cat([tok_a, tok_b], dim=1)  # (B, La+Lb, 512)
        
        # Add learnable CLS_int token
        cls_int = self.cls_int.expand(B, -1, -1)  # (B, 1, 512)
        
        # Combine: [CLS_int] + [protein_A_tokens] + [protein_B_tokens]
        combined = torch.cat([cls_int, protein_tokens], dim=1)  # (B, La+Lb+1, 512)
        
        # Pass through cross-attention layers
        # (optional) you can pass a key_padding_mask here if you padded to equal length
        for layer in self.cross_attn_layers:
            combined = layer(combined)
        
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
    Wraps the pretrained v2 MAE *without* any dimension or length adaptation.
    Returns encoder tokens of shape (B, L, 512).
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.core = TransformerMAE(
            input_dim=960, 
            embed_dim=512, 
            mask_ratio=0.0,  # No masking during inference
            num_layers=4, 
            nhead=16, 
            ff_dim=2048, 
            max_len=1502
        )
        
        # Load pretrained weights
        self.load_pretrained_weights(ckpt_path)
        
        # Freeze all parameters
        self.core.eval()
        for param in self.core.parameters():
            param.requires_grad = False
            
        print(f"✅ Loaded and froze pretrained v2 MAE from {ckpt_path}")
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained v2 MAE weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.core.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            print(f"❌ Error loading v2 MAE weights: {e}")
            raise e

    @torch.no_grad()
    def forward(self, x, lengths):
        """
        Forward pass using pretrained v2 MAE encoder only
        
        Args:
            x: (B, L, 960) - protein embeddings
            lengths: (B,) - sequence lengths
            
        Returns:
            encoder_output: (B, L, 512) - v2 MAE encoder output
        """
        B, L, _ = x.shape
        device = x.device
        
        # Embed and add positional encoding
        pos_added = self.core.embed(x) + self.core.pos_embed[:, :L]  # (B, L, 512)
        
        # Create padding mask
        pad_mask = torch.arange(L, device=device).expand(B, L) >= lengths.unsqueeze(1)
        
        # Pass through transformer encoder
        encoder_output = self.core.encoder(pos_added, src_key_padding_mask=pad_mask)  # (B, L, 512)
        
        return encoder_output

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
        
        # Cross-attention for interaction
        z_int = self.cross_attn(tok_a, tok_b)  # (B, 512)
        
        # Final classification
        logits = self.mlp_head(z_int)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
    
    def get_interaction_embeddings(self, emb_a, emb_b, lengths_a, lengths_b):
        """Get interaction embeddings without final classification"""
        with torch.no_grad():
            tok_a = self.encoder(emb_a, lengths_a)
            tok_b = self.encoder(emb_b, lengths_b)
            z_int = self.cross_attn(tok_a, tok_b)
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