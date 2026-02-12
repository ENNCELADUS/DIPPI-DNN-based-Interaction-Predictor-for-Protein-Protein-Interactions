"""
V5 PPI Classifier - Contact Map Modeling Ablation

This model tests contact map-based interaction modeling:
1. SiameseEncoder: Linear projection + norm (no transformer layers, same as V2)
2. BidirectionalCrossAttention: Residue-level info exchange between proteins
3. InteractionMapBuilder: Projects to pair space and builds enriched 2D grid [B, C, L_A, L_B]
4. ContactMapCNN: ResNet-style CNN for local pattern extraction (multiple blocks)
5. Aggregation: Global max/mean pool on CNN features → MLP head
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.model.v3 import MLPHead, SiameseEncoder, _build_padding_mask


class BidirectionalCrossAttentionLayer(nn.Module):
    """
    Single layer of bidirectional cross-attention between two protein sequences.

    Pre-LN architecture with residual connections and FFN:
    - A attends to B, updates H_A (shared weights)
    - B attends to A, updates H_B (shared weights)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Shared cross-attention: used for both A->B and B->A
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
            h_a: [B, L_A, D_h] - Protein A representations
            h_b: [B, L_B, D_h] - Protein B representations
            mask_a: [B, L_A] - Padding mask for A (True = padded)
            mask_b: [B, L_B] - Padding mask for B (True = padded)

        Returns:
            Updated (h_a, h_b) with same shapes
        """
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)

        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        return h_a, h_b


class BidirectionalCrossAttention(nn.Module):
    """
    Stack of bidirectional cross-attention layers.

    Unlike V2/V3's InteractionCrossAttention, this does NOT use a CLS token.
    It only updates the residue representations H_A and H_B.
    """

    def __init__(
        self, d_model: int, n_heads: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BidirectionalCrossAttentionLayer(
                d_model=d_model, n_heads=n_heads, dropout=dropout
            )
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
            h_a: [B, L_A, D_h]
            h_b: [B, L_B, D_h]
            lengths_a: [B] - Actual lengths of sequences in A
            lengths_b: [B] - Actual lengths of sequences in B

        Returns:
            (h_a, h_b) with updated representations
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

        return h_a, h_b


class InteractionMapBuilder(nn.Module):
    """
    Builds 2D interaction map from residue representations.

    Steps:
    1. Project H_A, H_B from D_h to D_p (pair dimension)
    2. Broadcast and combine to form [B, C, L_A, L_B] grid
    """

    def __init__(
        self,
        d_model: int,
        pair_dim: int,
        include_pair_features: bool,
        similarity: str,
        eps: float,
    ) -> None:
        super().__init__()
        self.proj_a = nn.Linear(d_model, pair_dim)
        self.proj_b = nn.Linear(d_model, pair_dim)
        self.activation = nn.GELU()
        self.include_pair_features = include_pair_features
        self.similarity = str(similarity).lower()
        self.eps = float(eps)
        if self.similarity not in {"none", "cosine", "dot"}:
            raise ValueError(
                "interaction_map.similarity must be one of: 'none', 'cosine', 'dot'"
            )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_a: [B, L_A, D_h]
            h_b: [B, L_B, D_h]
            mask_a: Optional padding mask for A
            mask_b: Optional padding mask for B

        Returns:
            M_in: [B, C, L_A, L_B] - Channel-first format for CNN
        """
        # Project to pair dimension: [B, L, D_h] -> [B, L, D_p]
        z_a = self.activation(self.proj_a(h_a))  # [B, L_A, D_p]
        z_b = self.activation(self.proj_b(h_b))  # [B, L_B, D_p]

        # Broadcast and expand
        # z_a: [B, L_A, D_p] -> [B, L_A, 1, D_p] -> [B, L_A, L_B, D_p]
        # z_b: [B, L_B, D_p] -> [B, 1, L_B, D_p] -> [B, L_A, L_B, D_p]
        L_A = z_a.size(1)
        L_B = z_b.size(1)

        z_a_exp = z_a.unsqueeze(2).expand(-1, -1, L_B, -1)  # [B, L_A, L_B, D_p]
        z_b_exp = z_b.unsqueeze(1).expand(-1, L_A, -1, -1)  # [B, L_A, L_B, D_p]

        features = [z_a_exp, z_b_exp]
        if self.include_pair_features:
            diff = torch.abs(z_a_exp - z_b_exp)
            prod = z_a_exp * z_b_exp
            features.extend([diff, prod])

        if self.similarity != "none":
            if self.similarity == "cosine":
                norm_a = torch.sqrt(
                    torch.sum(z_a_exp * z_a_exp, dim=-1, keepdim=True) + self.eps
                )
                norm_b = torch.sqrt(
                    torch.sum(z_b_exp * z_b_exp, dim=-1, keepdim=True) + self.eps
                )
                sim = torch.sum(z_a_exp * z_b_exp, dim=-1, keepdim=True) / (
                    norm_a * norm_b
                )
            else:
                scale = float(z_a_exp.size(-1)) ** 0.5
                sim = torch.sum(z_a_exp * z_b_exp, dim=-1, keepdim=True) / (
                    scale + self.eps
                )
            features.append(sim)

        # Concatenate along feature dimension
        M_raw = torch.cat(features, dim=-1)  # [B, L_A, L_B, C]

        # Permute to channel-first for CNN: [B, 2*D_p, L_A, L_B]
        M_in = M_raw.permute(0, 3, 1, 2).contiguous()

        return M_in


class ResidualBlock(nn.Module):
    """
    Standard ResNet-style residual block with two 3x3 convolutions.

    Path A (main): Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    Path B (skip): Identity
    Output: ReLU(Path_A + Path_B)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ContactMapCNN(nn.Module):
    """
    ResNet-style CNN for contact map feature extraction.

    Architecture:
    1. Feature Fusion: 1x1 Conv (C -> D_c) + BN + ReLU
    2. Spatial Residual Blocks: 3x3 Conv residual blocks
    3. Output kept at D_c channels for downstream pooling
    """

    def __init__(self, in_channels: int, cnn_dim: int, num_blocks: int, dropout: float):
        """
        Args:
            in_channels: Input channels (C)
            cnn_dim: CNN channel dimension (D_c)
            num_blocks: Number of residual blocks
            dropout: Dropout rate after each residual block
        """
        super().__init__()

        # Step 3.1: Feature Fusion (1x1 Conv)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True),
        )

        if num_blocks <= 0:
            raise ValueError("cnn_blocks must be >= 1")
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(cnn_dim) for _ in range(int(num_blocks))]
        )
        self.drop = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2*D_p, L_A, L_B]

        Returns:
            F_res: [B, D_c, L_A, L_B]
        """
        x = self.fusion(x)
        for block in self.res_blocks:
            x = block(x)
            x = self.drop(x)
        return x


class V5(nn.Module):
    """
    V5 PPI Classifier - Contact Map Modeling Ablation.

    Architecture:
    1. Siamese encoder: Linear projection + transformer + dropout
    2. Bidirectional cross-attention: Residue-level info exchange
    3. Interaction map builder: 2D grid with enriched pair features
    4. Contact map CNN: ResNet-style feature extraction (multiple blocks)
    5. Global pooling (max/mean) + MLP head for classification
    """

    name: str = "v5"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
            "pair_dim",
            "cnn_dim",
        ]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim: int = int(model_config["input_dim"])
        self.d_model: int = int(model_config["d_model"])
        self.encoder_layers: int = int(model_config["encoder_layers"])
        self.cross_attn_layers: int = int(model_config["cross_attn_layers"])
        self.n_heads: int = int(model_config["n_heads"])
        self.pair_dim: int = int(model_config["pair_dim"])
        self.cnn_dim: int = int(model_config["cnn_dim"])

        # MLP head config
        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        if not mlp_cfg:
            raise ValueError("mlp_head configuration is required for V5")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError(
                "mlp_head.hidden_dims and mlp_head.dropout must be provided"
            )
        self.mlp_hidden_dims = list(mlp_cfg["hidden_dims"])
        self.mlp_dropout = float(mlp_cfg["dropout"])
        self.mlp_activation = mlp_cfg.get("activation", "gelu")
        self.mlp_norm = mlp_cfg.get("norm", "layernorm")

        # Regularization config
        reg_cfg: Dict[str, Any] = model_config.get("regularization", {})
        if "dropout" not in reg_cfg:
            raise ValueError("regularization.dropout must be provided for V5")
        self.encoder_dropout = float(reg_cfg["dropout"])
        self.cross_attention_dropout = float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout)
        )
        self.token_dropout = float(reg_cfg.get("token_dropout", 0.0))
        self.stochastic_depth = float(reg_cfg.get("stochastic_depth", 0.0))
        self.cnn_dropout = float(reg_cfg.get("cnn_dropout", 0.0))

        map_cfg: Dict[str, Any] = model_config.get("interaction_map", {})
        self.map_include_pair_features = bool(
            map_cfg.get("include_pair_features", True)
        )
        self.map_similarity = str(map_cfg.get("similarity", "cosine")).lower()
        self.map_eps = float(map_cfg.get("eps", 1.0e-8))

        cnn_blocks = int(model_config.get("cnn_blocks", 2))

        pool_cfg: Dict[str, Any] = model_config.get("pooling", {})
        self.pooling = str(pool_cfg.get("mode", "max_mean")).lower()
        if self.pooling not in {"max", "mean", "max_mean"}:
            raise ValueError("pooling.mode must be one of: 'max', 'mean', 'max_mean'")

        # Build modules - V5: Use v3-style encoder with transformer layers
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
        )

        self.cross_attention = BidirectionalCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
        )

        map_channels = 2 * self.pair_dim
        if self.map_include_pair_features:
            map_channels += 2 * self.pair_dim
        if self.map_similarity != "none":
            map_channels += 1

        self.map_builder = InteractionMapBuilder(
            d_model=self.d_model,
            pair_dim=self.pair_dim,
            include_pair_features=self.map_include_pair_features,
            similarity=self.map_similarity,
            eps=self.map_eps,
        )

        self.contact_cnn = ContactMapCNN(
            in_channels=map_channels,
            cnn_dim=self.cnn_dim,
            num_blocks=cnn_blocks,
            dropout=self.cnn_dropout,
        )

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_mean_pool = nn.AdaptiveAvgPool2d((1, 1))
        head_input_dim = self.cnn_dim * (2 if self.pooling == "max_mean" else 1)

        # MLP head: input is pooled_dim (after pooling)
        self.output_head = MLPHead(
            input_dim=head_input_dim,
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

        # 1. Encode both proteins (shared weights)
        encoded_a = self.encoder(emb_a, lengths_a)  # [B, L_A, D_h]
        encoded_b = self.encoder(emb_b, lengths_b)  # [B, L_B, D_h]

        # 2. Bidirectional cross-attention
        h_a, h_b = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)

        # 3. Build interaction map
        interaction_map = self.map_builder(h_a, h_b)  # [B, 2*D_p, L_A, L_B]

        # 4. Contact map CNN
        features = self.contact_cnn(interaction_map)  # [B, D_c, L_A, L_B]

        # 5. Global pooling
        if self.pooling == "max_mean":
            pooled_max = self.global_max_pool(features)
            pooled_mean = self.global_mean_pool(features)
            pooled = torch.cat([pooled_max, pooled_mean], dim=1)
        elif self.pooling == "mean":
            pooled = self.global_mean_pool(features)
        else:
            pooled = self.global_max_pool(features)
        pooled = pooled.flatten(1)

        # 6. MLP head
        logits = self.output_head(pooled)  # [B, 1]

        # Compute loss if labels are provided
        output = {"logits": logits}
        if "label" in merged_batch:
            labels = merged_batch["label"].float()
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
