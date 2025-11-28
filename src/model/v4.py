from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .v3 import MLPHead, SiameseEncoder, _build_padding_mask


class CrossAttentionLayerNoCLS(nn.Module):
    """Bidirectional cross-attention block without CLS pooling."""

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
        a_norm = self.norm_a(h_a)
        attn_a, _ = self.attn_a_to_b(a_norm, h_b, h_b, key_padding_mask=mask_b)
        h_a = h_a + self.drop_a(attn_a)

        b_norm = self.norm_b(h_b)
        attn_b, _ = self.attn_b_to_a(b_norm, h_a, h_a, key_padding_mask=mask_a)
        h_b = h_b + self.drop_b(attn_b)

        return h_a, h_b


def _masked_mean_pool(
    x: torch.Tensor, padding_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if padding_mask is None:
        return x.mean(dim=1)
    valid_mask = (~padding_mask).unsqueeze(-1)
    valid_mask = valid_mask.type_as(x)
    summed = (x * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class InteractionCrossAttentionBiPool(nn.Module):
    """Cross-attention stack that returns pooled pair features instead of CLS."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionLayerNoCLS(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Protein pair batches must share batch dimension")

        max_len_a = h_a.size(1)
        max_len_b = h_b.size(1)
        mask_a = _build_padding_mask(lengths_a, max_len_a)
        mask_b = _build_padding_mask(lengths_b, max_len_b)

        for layer in self.layers:
            h_a, h_b = layer(h_a, h_b, mask_a, mask_b)

        rep_a_mean = _masked_mean_pool(h_a, mask_a)
        rep_b_mean = _masked_mean_pool(h_b, mask_b)
        interaction_repr = torch.cat(
            [
                rep_a_mean,
                rep_b_mean,
                torch.abs(rep_a_mean - rep_b_mean),
                rep_a_mean * rep_b_mean,
            ],
            dim=-1,
        )
        return interaction_repr


class V4(nn.Module):
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

        self.input_dim = int(model_config["input_dim"])
        self.d_model = int(model_config["d_model"])
        self.encoder_layers = int(model_config["encoder_layers"])
        self.cross_attn_layers = int(model_config["cross_attn_layers"])
        self.n_heads = int(model_config["n_heads"])

        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        if not mlp_cfg or "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError(
                "mlp_head configuration with hidden_dims and dropout is required for V4"
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
        self.stochastic_depth = float(reg_cfg.get("stochastic_depth", 0.0))

        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
        )
        self.cross_attention = InteractionCrossAttentionBiPool(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
        )
        self.output_head = MLPHead(
            input_dim=4 * self.d_model,
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

        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)
        interaction_repr = self.cross_attention(
            encoded_a, encoded_b, lengths_a, lengths_b
        )
        logits = self.output_head(interaction_repr)

        output = {"logits": logits}
        if "label" in batch:
            labels = batch["label"].float()
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
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, labels_for_loss
            )
            output["loss"] = loss

        return output
