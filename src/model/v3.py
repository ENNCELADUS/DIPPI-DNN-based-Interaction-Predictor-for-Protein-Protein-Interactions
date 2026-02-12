"""V3 model definition for protein-protein interaction classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth per-sample residual drop-path layer."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(max(0.0, min(1.0, drop_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic path dropping.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after stochastic depth.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def _build_padding_mask(lengths: torch.Tensor | None, max_len: int) -> torch.Tensor | None:
    """Create a padding mask from sequence lengths.

    Args:
        lengths: Batch sequence lengths.
        max_len: Padded sequence length.

    Returns:
        Boolean padding mask of shape ``(batch, max_len)``.
    """
    if lengths is None:
        return None
    if lengths.dim() != 1:
        raise ValueError("lengths must be a 1D tensor of shape (batch_size,)")
    return torch.arange(max_len, device=lengths.device).expand(
        lengths.size(0), max_len
    ) >= lengths.unsqueeze(1)


def _to_int(value: object, field_name: str) -> int:
    """Convert config value to integer."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be int-compatible") from error
    raise ValueError(f"{field_name} must be int-compatible")


def _to_float(value: object, field_name: str) -> float:
    """Convert config value to float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be float-compatible") from error
    raise ValueError(f"{field_name} must be float-compatible")


def _to_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Validate config value as key-value mapping."""
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be a mapping")


class SiameseEncoder(nn.Module):
    """Shared transformer encoder for each protein sequence."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        token_dropout: float,
        stochastic_depth: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.token_dropout = nn.Dropout(token_dropout) if token_dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        )
        # Stochastic depth: linear schedule across depth, identity when rate == 0
        sd_max = float(max(0.0, min(1.0, stochastic_depth)))
        if sd_max > 0.0 and n_layers > 0:
            rates = [sd_max * float(i + 1) / float(n_layers) for i in range(n_layers)]
            self.drop_paths = nn.ModuleList([DropPath(r) for r in rates])
        else:
            self.drop_paths = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode token embeddings.

        Args:
            embeddings: Input embedding tensor ``(batch, seq_len, input_dim)``.
            lengths: Sequence lengths tensor ``(batch,)``.

        Returns:
            Encoded embedding tensor ``(batch, seq_len, d_model)``.
        """
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be of shape (batch_size, seq_len, embedding_dim)")
        projected = self.input_projection(embeddings)
        projected = self.token_dropout(projected)
        max_len = projected.size(1)
        padding_mask = _build_padding_mask(lengths, max_len)
        for idx, layer in enumerate(self.layers):
            residual_in = projected
            y = layer(projected, src_key_padding_mask=padding_mask)
            # Apply layer-level stochastic depth to the residual delta f(x) = y - x
            delta = y - residual_in
            delta = self.drop_paths[idx](delta)  # Identity in eval or when rate==0
            projected = residual_in + delta
        return cast(torch.Tensor, self.output_norm(projected))


class CrossAttentionLayer(nn.Module):
    """Shared-weight bidirectional cross-attention with FFN and CLS pooling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_cls_attn = nn.LayerNorm(d_model)
        self.norm_cls_ffn = nn.LayerNorm(d_model)
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
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.drop_cls_attn = nn.Dropout(dropout)
        self.drop_cls_ffn = nn.Dropout(dropout)

    def _attend(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply attention sub-layer with residual connection."""
        query_norm = self.norm_attn(query)
        attn_out, _ = self.attn(query_norm, key_value, key_value, key_padding_mask=key_padding_mask)
        return query + cast(torch.Tensor, self.drop_attn(attn_out))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward sub-layer with residual connection."""
        return x + cast(torch.Tensor, self.drop_ffn(self.ffn(self.norm_ffn(x))))

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one bidirectional cross-attention block.

        Args:
            h_a: Protein A hidden states.
            h_b: Protein B hidden states.
            cls_token: CLS token state.
            mask_a: Padding mask for sequence A.
            mask_b: Padding mask for sequence B.

        Returns:
            Updated ``(h_a, h_b, cls_token)`` tuple.
        """
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)

        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        combined = torch.cat([h_a, h_b], dim=1)
        if mask_a is not None and mask_b is not None:
            combined_mask = torch.cat([mask_a, mask_b], dim=1)
        else:
            combined_mask = None

        cls_norm = self.norm_cls_attn(cls_token)
        attn_cls, _ = self.attn_cls(cls_norm, combined, combined, key_padding_mask=combined_mask)
        cls_token = cls_token + self.drop_cls_attn(attn_cls)

        cls_ffn_norm = self.norm_cls_ffn(cls_token)
        cls_token = cls_token + self.drop_cls_ffn(self.ff_cls(cls_ffn_norm))

        return h_a, h_b, cls_token


class InteractionCrossAttention(nn.Module):
    """Stacked cross-attention encoder with CLS pooling."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
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
        """Pool pair representation from encoded proteins.

        Args:
            h_a: Protein A hidden states.
            h_b: Protein B hidden states.
            lengths_a: Sequence lengths for A.
            lengths_b: Sequence lengths for B.

        Returns:
            CLS representation of shape ``(batch, d_model)``.
        """
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
    """Configurable MLP head for binary interaction logits."""

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

        activation_map: dict[str, nn.Module] = {
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
        """Initialize linear layers with Xavier uniform weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output logits.

        Args:
            x: Input feature tensor.

        Returns:
            Output logit tensor.
        """
        return cast(torch.Tensor, self.layers(x))


class V3(nn.Module):
    """V3 model for protein-protein interaction classification."""

    name: str = "v3"

    def __init__(self, **model_config: object) -> None:
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

        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model = _to_int(model_config["d_model"], "model_config.d_model")
        self.encoder_layers = _to_int(model_config["encoder_layers"], "model_config.encoder_layers")
        self.cross_attn_layers = _to_int(
            model_config["cross_attn_layers"],
            "model_config.cross_attn_layers",
        )
        self.n_heads = _to_int(model_config["n_heads"], "model_config.n_heads")

        mlp_cfg_raw = model_config.get("mlp_head")
        if not isinstance(mlp_cfg_raw, dict) or not mlp_cfg_raw:
            raise ValueError("mlp_head configuration is required for V3")
        mlp_cfg = _to_mapping(mlp_cfg_raw, "model_config.mlp_head")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError("mlp_head.hidden_dims and mlp_head.dropout must be provided")
        hidden_dims_raw = mlp_cfg["hidden_dims"]
        if not isinstance(hidden_dims_raw, list) or not hidden_dims_raw:
            raise ValueError("mlp_head.hidden_dims must be a non-empty list")
        self.mlp_hidden_dims = [
            _to_int(value, "model_config.mlp_head.hidden_dims") for value in hidden_dims_raw
        ]
        self.mlp_dropout = _to_float(mlp_cfg["dropout"], "model_config.mlp_head.dropout")
        self.mlp_activation = str(mlp_cfg.get("activation", "gelu"))
        self.mlp_norm = str(mlp_cfg.get("norm", "layernorm"))

        reg_cfg_raw = model_config.get("regularization")
        if not isinstance(reg_cfg_raw, dict) or "dropout" not in reg_cfg_raw:
            raise ValueError("regularization.dropout must be provided for V3")
        reg_cfg = _to_mapping(reg_cfg_raw, "model_config.regularization")
        self.encoder_dropout = _to_float(reg_cfg["dropout"], "model_config.regularization.dropout")
        self.cross_attention_dropout = _to_float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout),
            "model_config.regularization.cross_attention_dropout",
        )
        self.token_dropout = _to_float(
            reg_cfg.get("token_dropout", 0.0),
            "model_config.regularization.token_dropout",
        )
        self.stochastic_depth = _to_float(
            reg_cfg.get("stochastic_depth", 0.0),
            "model_config.regularization.stochastic_depth",
        )

        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
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

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a model forward pass.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            Output dictionary containing ``logits`` and optional ``loss``.
        """
        merged_batch: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged_batch.update(batch)
        merged_batch.update(kwargs)

        if "emb_a" not in merged_batch or "emb_b" not in merged_batch:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = merged_batch["emb_a"]
        emb_b = merged_batch["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = merged_batch.get("len_a")
        lengths_b = merged_batch.get("len_b")
        if lengths_a is None:
            lengths_a = torch.full((emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long)
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full((emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long)
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)
        cls_representation = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.output_head(cls_representation)

        # Compute loss if labels are provided (training mode)
        output = {"logits": logits}
        if "label" in merged_batch:
            labels = merged_batch["label"].float()
            # Normalize logits shape: (N, 1) → (N,) and (N, 1) labels → (N,)
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            # Compute BCE loss with logits
            loss = nn.functional.binary_cross_entropy_with_logits(logits_for_loss, labels_for_loss)
            output["loss"] = loss

        return output
