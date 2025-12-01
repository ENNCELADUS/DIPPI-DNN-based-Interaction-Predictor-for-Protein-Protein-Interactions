from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _build_padding_mask(lengths: torch.Tensor, max_len: int) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    if lengths.dim() != 1:
        raise ValueError("lengths must be a 1D tensor of shape (batch_size,)")
    return torch.arange(max_len, device=lengths.device).expand(
        lengths.size(0), max_len
    ) >= lengths.unsqueeze(1)


class SiameseEncoder(nn.Module):
    """Ablation encoder: linear projection + norm + activation (no transformer layers)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float,
        token_dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.token_dropout = (
            nn.Dropout(token_dropout) if token_dropout > 0.0 else nn.Identity()
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError(
                "embeddings must be of shape (batch_size, seq_len, embedding_dim)"
            )
        projected = self.input_projection(embeddings)
        projected = self.token_dropout(projected)
        projected = self.output_norm(projected)
        return self.activation(projected)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.norm_cls_attn = nn.LayerNorm(d_model)
        self.norm_cls_ffn = nn.LayerNorm(d_model)
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
        self.attn_cls = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_cls = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_a = nn.Dropout(dropout)
        self.drop_b = nn.Dropout(dropout)
        self.drop_cls_attn = nn.Dropout(dropout)
        self.drop_cls_ffn = nn.Dropout(dropout)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        mask_b: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_norm = self.norm_a(h_a)
        attn_a, _ = self.attn_a_to_b(a_norm, h_b, h_b, key_padding_mask=mask_b)
        h_a = h_a + self.drop_a(attn_a)

        b_norm = self.norm_b(h_b)
        attn_b, _ = self.attn_b_to_a(b_norm, h_a, h_a, key_padding_mask=mask_a)
        h_b = h_b + self.drop_b(attn_b)

        combined = torch.cat([h_a, h_b], dim=1)
        if mask_a is not None and mask_b is not None:
            combined_mask = torch.cat([mask_a, mask_b], dim=1)
        else:
            combined_mask = None

        cls_norm = self.norm_cls_attn(cls_token)
        attn_cls, _ = self.attn_cls(
            cls_norm, combined, combined, key_padding_mask=combined_mask
        )
        cls_token = cls_token + self.drop_cls_attn(attn_cls)

        cls_ffn_norm = self.norm_cls_ffn(cls_token)
        cls_token = cls_token + self.drop_cls_ffn(self.ff_cls(cls_ffn_norm))

        return h_a, h_b, cls_token


class InteractionCrossAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
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
    ) -> torch.Tensor:
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        batch_size = h_a.size(0)
        max_len_a = h_a.size(1)
        max_len_b = h_b.size(1)
        mask_a = _build_padding_mask(lengths_a, max_len_a)
        mask_b = _build_padding_mask(lengths_b, max_len_b)

        cls_token = self.cls_token.repeat(batch_size, 1, 1)

        for layer in self.layers:
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)

        return cls_token.squeeze(1)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        activation: str,
        norm: str,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        activation_map: Dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}' for MLPHead")

        def build_norm(dim: int) -> nn.Module:
            if norm == "layernorm":
                return nn.LayerNorm(dim)
            if norm == "batchnorm":
                return nn.BatchNorm1d(dim)
            if norm == "none":
                return nn.Identity()
            raise ValueError(f"Unsupported norm '{norm}' for MLPHead")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(build_norm(hidden_dim))
            layers.append(activation_map[activation])
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class V2(nn.Module):
    """
    V2 PPI Classifier - Ablation model without encoder transformer layers.

    This model tests whether cross-attention alone (without self-attention in the encoder)
    is sufficient for PPI prediction. Architecture:
    1. Siamese encoder: Linear projection + dropout + norm only (no transformers)
    2. Cross-attention interaction pooling with CLS token (same as V3)
    3. MLP classification head (same as V3)
    """

    name: str = "v2"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            # encoder_layers NOT required for V2 (no transformer layers)
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim: int = int(model_config["input_dim"])
        # encoder_layers is optional and ignored in V2 (kept for config compatibility)
        self.encoder_layers: int = int(model_config.get("encoder_layers", 0))
        self.cross_attn_layers: int = int(model_config["cross_attn_layers"])
        self.n_heads: int = int(model_config["n_heads"])

        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        if not mlp_cfg:
            raise ValueError("mlp_head configuration is required for V2")
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
            raise ValueError("regularization.dropout must be provided for V2")
        self.encoder_dropout = float(reg_cfg["dropout"])
        self.cross_attention_dropout = float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout)
        )
        self.token_dropout = float(reg_cfg.get("token_dropout", 0.0))

        # Optional toggles retained as placeholders for compatibility
        self._unused_geometry_cfg = model_config.get("geometry", None)
        self._unused_inference_cfg = model_config.get("inference", None)
        self._unused_spectral_norm = model_config.get("spectral_norm", False)
        self._unused_mc_dropout_eval = model_config.get("use_mc_dropout_eval", False)
        self._unused_mc_samples = model_config.get("mc_dropout_samples", 0)

        # V2 ablation: encoder has no transformer layers, only projection + norm
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
        )
        self.cross_attention = InteractionCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
        )
        self.output_head = MLPHead(
            input_dim=self.d_model,
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
        cls_representation = self.cross_attention(
            encoded_a, encoded_b, lengths_a, lengths_b
        )
        logits = self.output_head(cls_representation)

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
