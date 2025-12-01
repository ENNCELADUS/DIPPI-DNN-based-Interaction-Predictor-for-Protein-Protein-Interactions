from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .v2 import SiameseEncoder
from .v3 import MLPHead, _build_padding_mask


class AttentionPooling(nn.Module):
    """
    Pool a sequence to a single vector via a learned query.

    A learnable query vector attends to all positions in the input sequence
    using multi-head attention, producing a fixed-size representation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional padding mask of shape (batch, seq_len), True = masked

        Returns:
            Pooled tensor of shape (batch, d_model)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        x_norm = self.norm(x)
        out, _ = self.attn(query, x_norm, x_norm, key_padding_mask=mask)
        return out.squeeze(1)


class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention layer without CLS token.

    Performs A→B and B→A cross-attention to exchange information
    between two protein sequences.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_a = nn.Dropout(dropout)
        self.drop_b = nn.Dropout(dropout)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        mask_b: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_a: Protein A representations (batch, seq_a, d_model)
            h_b: Protein B representations (batch, seq_b, d_model)
            mask_a: Padding mask for A (batch, seq_a), True = masked
            mask_b: Padding mask for B (batch, seq_b), True = masked

        Returns:
            Updated (h_a, h_b) after cross-attention
        """
        # A attends to B
        a_norm = self.norm_a(h_a)
        attn_a, _ = self.attn_a_to_b(a_norm, h_b, h_b, key_padding_mask=mask_b)
        h_a = h_a + self.drop_a(attn_a)

        # B attends to A
        b_norm = self.norm_b(h_b)
        attn_b, _ = self.attn_b_to_a(b_norm, h_a, h_a, key_padding_mask=mask_a)
        h_b = h_b + self.drop_b(attn_b)

        return h_a, h_b


class InteractionCrossAttention(nn.Module):
    """
    Stacked bidirectional cross-attention without CLS token.

    Exchanges information between two protein sequences through
    multiple layers of bidirectional cross-attention.
    """

    def __init__(
        self, d_model: int, n_heads: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_a: Protein A representations (batch, seq_a, d_model)
            h_b: Protein B representations (batch, seq_b, d_model)
            lengths_a: Actual lengths for A (batch,)
            lengths_b: Actual lengths for B (batch,)

        Returns:
            (h_a', h_b') after cross-attention layers
        """
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        max_len_a = h_a.size(1)
        max_len_b = h_b.size(1)
        mask_a = _build_padding_mask(lengths_a, max_len_a)
        mask_b = _build_padding_mask(lengths_b, max_len_b)

        for layer in self.layers:
            h_a, h_b = layer(h_a, h_b, mask_a, mask_b)

        return h_a, h_b, mask_a, mask_b


class V4(nn.Module):
    """
    V4 PPI Classifier - Ablation model for V3.

    Architecture:
    1. SiameseEncoder (V2-style): Linear projection + dropout + norm (no self-attention)
    2. Bidirectional cross-attention (A→B, B→A) WITHOUT CLS token
    3. Attention pooling with learned query to get v_a, v_b
    4. Combine: product = v_a * v_b, diff = |v_a - v_b|, concat → [batch, 2*d_model]
    5. MLP classification head
    """

    name: str = "v4"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim: int = int(model_config["input_dim"])
        self.d_model: int = int(model_config["d_model"])
        # encoder_layers is optional and ignored in V4 (kept for config compatibility)
        self.encoder_layers: int = int(model_config.get("encoder_layers", 0))
        self.cross_attn_layers: int = int(model_config["cross_attn_layers"])
        self.n_heads: int = int(model_config["n_heads"])

        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        if not mlp_cfg:
            raise ValueError("mlp_head configuration is required for V4")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError(
                "mlp_head.hidden_dims and mlp_head.dropout must be provided"
            )
        self.mlp_hidden_dims = list(mlp_cfg["hidden_dims"])
        self.mlp_dropout = float(mlp_cfg["dropout"])
        self.mlp_activation = mlp_cfg.get("activation", "gelu")
        self.mlp_norm = mlp_cfg.get("norm", "layernorm")

        reg_cfg: Dict[str, Any] = model_config.get("regularization", {})
        if "dropout" not in reg_cfg:
            raise ValueError("regularization.dropout must be provided for V4")
        self.encoder_dropout = float(reg_cfg["dropout"])
        self.cross_attention_dropout = float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout)
        )
        self.token_dropout = float(reg_cfg.get("token_dropout", 0.0))

        # V4 ablation: encoder has no transformer layers (V2-style)
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
        )

        # Cross-attention (bidirectional, no CLS token)
        self.cross_attention = InteractionCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
        )

        # Attention pooling for each protein
        self.pool_a = AttentionPooling(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.cross_attention_dropout,
        )
        self.pool_b = AttentionPooling(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.cross_attention_dropout,
        )

        # MLP head takes 2*d_model input (product + diff concatenation)
        self.output_head = MLPHead(
            input_dim=2 * self.d_model,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=1,
            dropout=self.mlp_dropout,
            activation=self.mlp_activation,
            norm=self.mlp_norm,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "emb_a" not in batch or "emb_b" not in batch:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = batch["emb_a"]
        emb_b = batch["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError(
                "Input embeddings must be shaped (batch, seq_len, embedding_dim)"
            )
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = batch.get("len_a")
        lengths_b = batch.get("len_b")
        if lengths_a is None:
            lengths_a = torch.full(
                (emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long
            )
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full(
                (emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long
            )
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        # Encode (no self-attention, just projection)
        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)

        # Cross-attention (bidirectional, no CLS)
        h_a, h_b, mask_a, mask_b = self.cross_attention(
            encoded_a, encoded_b, lengths_a, lengths_b
        )

        # Attention pooling
        v_a = self.pool_a(h_a, mask_a)  # [batch, d_model]
        v_b = self.pool_b(h_b, mask_b)  # [batch, d_model]

        # Combine with product and absolute difference
        product = v_a * v_b
        diff = torch.abs(v_a - v_b)
        combined = torch.cat([product, diff], dim=-1)  # [batch, 2*d_model]

        # Classification
        logits = self.output_head(combined)

        # Compute loss if labels are provided (training mode)
        output = {"logits": logits}
        if "label" in batch:
            labels = batch["label"].float()
            # Normalize logits shape: (N, 1) → (N,) and (N, 1) labels → (N,)
            logits_for_loss = (
                logits.squeeze(-1)
                if logits.dim() > 1 and logits.size(-1) == 1
                else logits
            )
            labels_for_loss = (
                labels.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1
                else labels
            )
            # Compute BCE loss with logits
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, labels_for_loss
            )
            output["loss"] = loss

        return output
