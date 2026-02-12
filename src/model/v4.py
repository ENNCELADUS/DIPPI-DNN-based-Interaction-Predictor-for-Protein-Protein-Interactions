from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.model.v3 import SiameseEncoder, _build_padding_mask


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


def _build_activation(name: str) -> nn.Module:
    name_norm = str(name).lower()
    if name_norm == "gelu":
        return nn.GELU()
    if name_norm == "relu":
        return nn.ReLU()
    if name_norm in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class CrossAttentionLayer(nn.Module):
    """
    Shared-weight bidirectional cross-attention block with FFN.

    Each direction applies:
      x = x + Dropout(CrossAttn(LN(x)))
      x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)

    def _attend(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_norm = self.norm_attn(query)
        attn_out, _ = self.attn(
            query_norm, key_value, key_value, key_padding_mask=key_padding_mask
        )
        return query + self.drop_attn(attn_out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_ffn(self.ffn(self.norm_ffn(x)))

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
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)

        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        return h_a, h_b


class InteractionCrossAttention(nn.Module):
    """
    Stacked shared-weight bidirectional cross-attention with FFN.

    Exchanges information between two protein sequences through
    multiple shared-weight cross-attention blocks.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm, activation, and dropout."""

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act = _build_activation(activation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class ResidualMLPHead(nn.Module):
    """Residual MLP head with configurable hidden dims and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("mlp_head.hidden_dims must be non-empty for V4")
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    input_dim=input_dim,
                    hidden_dim=int(hidden_dim),
                    dropout=dropout,
                    activation=activation,
                )
                for hidden_dim in hidden_dims
            ]
        )
        self.output = nn.Linear(input_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class V4(nn.Module):
    """
    V4 PPI Classifier - Ablation model for V3.

    Architecture:
    1. SiameseEncoder (V2-style): Linear projection + dropout + norm (no self-attention)
    2. Shared-weight bidirectional cross-attention with FFN
    3. Attention pooling with learned query to get v_a, v_b
    4. Combine: [v_a, v_b, |v_a - v_b|, v_a * v_b] → LayerNorm
    5. Residual MLP head with dropout
    """

    name: str = "v4"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim: int = int(model_config["input_dim"])
        self.d_model: int = int(model_config["d_model"])
        self.encoder_layers: int = int(model_config["encoder_layers"])
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
        if str(self.mlp_norm).lower() not in {"layernorm", "ln"}:
            raise ValueError("mlp_head.norm must be 'layernorm' for V4")

        reg_cfg: Dict[str, Any] = model_config.get("regularization", {})
        if "dropout" not in reg_cfg:
            raise ValueError("regularization.dropout must be provided for V4")
        self.encoder_dropout = float(reg_cfg["dropout"])
        self.cross_attention_dropout = float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout)
        )
        self.token_dropout = float(reg_cfg.get("token_dropout", 0.0))
        self.stochastic_depth = float(reg_cfg.get("stochastic_depth", 0.0))

        # V4: Use v3-style encoder with transformer layers
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
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

        pair_dim = 4 * self.d_model
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.output_head = ResidualMLPHead(
            input_dim=pair_dim,
            hidden_dims=self.mlp_hidden_dims,
            dropout=self.mlp_dropout,
            activation=self.mlp_activation,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        merged_batch: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged_batch.update(batch)
        merged_batch.update(kwargs)

        if "emb_a" not in merged_batch or "emb_b" not in merged_batch:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = merged_batch["emb_a"]
        emb_b = merged_batch["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError(
                "Input embeddings must be shaped (batch, seq_len, embedding_dim)"
            )
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = merged_batch.get("len_a")
        lengths_b = merged_batch.get("len_b")
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

        # Cross-attention (bidirectional, shared weights, with FFN)
        h_a, h_b, mask_a, mask_b = self.cross_attention(
            encoded_a, encoded_b, lengths_a, lengths_b
        )

        # Attention pooling
        v_a = self.pool_a(h_a, mask_a)  # [batch, d_model]
        v_b = self.pool_b(h_b, mask_b)  # [batch, d_model]

        # Combine with full pair features
        product = v_a * v_b
        diff = torch.abs(v_a - v_b)
        combined = torch.cat([v_a, v_b, diff, product], dim=-1)  # [batch, 4*d_model]
        combined = self.pair_norm(combined)

        # Classification
        logits = self.output_head(combined)

        # Compute loss if labels are provided (training mode)
        output = {"logits": logits}
        if "label" in merged_batch:
            labels = merged_batch["label"].float()
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
