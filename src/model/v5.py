"""
V5 PPI Classifier - Contact Map Modeling Ablation

This model tests contact map-based interaction modeling:
1. SiameseEncoder: Linear projection + norm (no transformer layers, same as V2)
2. BidirectionalCrossAttention: Residue-level info exchange between proteins
3. InteractionMapBuilder: Projects to pair space and builds 2D grid [B, 2*D_p, L_A, L_B]
4. ContactMapCNN: ResNet-style CNN for local pattern extraction
5. Aggregation: Global max pool on CNN features → MLP head
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .v2 import SiameseEncoder, MLPHead, _build_padding_mask


class BidirectionalCrossAttentionLayer(nn.Module):
    """
    Single layer of bidirectional cross-attention between two protein sequences.

    Pre-LN architecture with residual connections:
    - A attends to B, updates H_A
    - B attends to A, updates H_B
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

        # Cross-attention: A to B (Q from A, K/V from B)
        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Cross-attention: B to A (Q from B, K/V from A)
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
            h_a: [B, L_A, D_h] - Protein A representations
            h_b: [B, L_B, D_h] - Protein B representations
            mask_a: [B, L_A] - Padding mask for A (True = padded)
            mask_b: [B, L_B] - Padding mask for B (True = padded)

        Returns:
            Updated (h_a, h_b) with same shapes
        """
        # A attends to B: Q=A, K=V=B
        a_norm = self.norm_a(h_a)
        attn_a, _ = self.attn_a_to_b(a_norm, h_b, h_b, key_padding_mask=mask_b)
        h_a = h_a + self.drop_a(attn_a)

        # B attends to A: Q=B, K=V=A (use updated h_a)
        b_norm = self.norm_b(h_b)
        attn_b, _ = self.attn_b_to_a(b_norm, h_a, h_a, key_padding_mask=mask_a)
        h_b = h_b + self.drop_b(attn_b)

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
    2. Broadcast and concatenate to form [B, 2*D_p, L_A, L_B] grid
    """

    def __init__(self, d_model: int, pair_dim: int) -> None:
        super().__init__()
        self.proj_a = nn.Linear(d_model, pair_dim)
        self.proj_b = nn.Linear(d_model, pair_dim)
        self.activation = nn.GELU()

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
            M_in: [B, 2*D_p, L_A, L_B] - Channel-first format for CNN
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

        # Concatenate along feature dimension
        M_raw = torch.cat([z_a_exp, z_b_exp], dim=-1)  # [B, L_A, L_B, 2*D_p]

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
    1. Feature Fusion: 1x1 Conv (2*D_p -> D_c) + BN + ReLU
    2. Spatial Residual Block: 3x3 Conv residual block
    3. Output kept at D_c channels for downstream pooling
    """

    def __init__(self, in_channels: int, cnn_dim: int) -> None:
        """
        Args:
            in_channels: Input channels (2 * pair_dim)
            cnn_dim: CNN channel dimension (D_c)
        """
        super().__init__()

        # Step 3.1: Feature Fusion (1x1 Conv)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True),
        )

        # Step 3.2: Spatial Residual Block
        self.res_block = ResidualBlock(cnn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2*D_p, L_A, L_B]

        Returns:
            F_res: [B, D_c, L_A, L_B]
        """
        x = self.fusion(x)
        x = self.res_block(x)
        return x


class V5(nn.Module):
    """
    V5 PPI Classifier - Contact Map Modeling Ablation.

    Architecture:
    1. Siamese encoder: Linear projection + dropout + norm (no transformers, same as V2)
    2. Bidirectional cross-attention: Residue-level info exchange
    3. Interaction map builder: 2D grid construction [B, 2*D_p, L_A, L_B]
    4. Contact map CNN: ResNet-style feature extraction
    5. Global max pooling + MLP head for classification
    """

    name: str = "v5"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
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

        # Build modules
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
        )

        self.cross_attention = BidirectionalCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
        )

        self.map_builder = InteractionMapBuilder(
            d_model=self.d_model,
            pair_dim=self.pair_dim,
        )

        self.contact_cnn = ContactMapCNN(
            in_channels=2 * self.pair_dim,
            cnn_dim=self.cnn_dim,
        )

        # Global max pooling
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # MLP head: input is cnn_dim (after pooling)
        self.output_head = MLPHead(
            input_dim=self.cnn_dim,
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

        # 1. Encode both proteins (shared weights)
        encoded_a = self.encoder(emb_a, lengths_a)  # [B, L_A, D_h]
        encoded_b = self.encoder(emb_b, lengths_b)  # [B, L_B, D_h]

        # 2. Bidirectional cross-attention
        h_a, h_b = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)

        # 3. Build interaction map
        interaction_map = self.map_builder(h_a, h_b)  # [B, 2*D_p, L_A, L_B]

        # 4. Contact map CNN
        features = self.contact_cnn(interaction_map)  # [B, D_c, L_A, L_B]

        # 5. Global max pooling
        pooled = self.global_pool(features)  # [B, D_c, 1, 1]
        pooled = pooled.flatten(1)  # [B, D_c]

        # 6. MLP head
        logits = self.output_head(pooled)  # [B, 1]

        # Compute loss if labels are provided
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
