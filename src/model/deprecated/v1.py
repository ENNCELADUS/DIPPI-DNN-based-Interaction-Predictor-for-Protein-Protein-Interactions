from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class V1(nn.Module):
    """
    V1 PPI Classifier - Simple baseline with average pooling + MLP.

    Architecture:
    1. Average pool embeddings (length-aware)
    2. Shared protein encoder (Linear + LayerNorm + ReLU + Dropout)
    3. Concatenate encoded features
    4. 3-layer MLP classification head
    """

    name: str = "v1"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()

        # Parse config
        self.input_dim: int = int(model_config.get("input_dim", 1536))

        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        self.hidden_dim: int = int(mlp_cfg.get("hidden_dim", 256))

        reg_cfg: Dict[str, Any] = model_config.get("regularization", {})
        self.dropout: float = float(reg_cfg.get("dropout", 0.2))

        # Protein encoder (shared for both proteins)
        self.protein_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Interaction head (3-layer MLP)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self._initialize_weights()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        emb_a = batch["emb_a"]
        emb_b = batch["emb_b"]

        device = emb_a.device

        # Handle lengths (default to full sequence)
        lengths_a = batch.get("len_a")
        if lengths_a is None:
            lengths_a = torch.full(
                (emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long
            )
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)

        lengths_b = batch.get("len_b")
        if lengths_b is None:
            lengths_b = torch.full(
                (emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long
            )
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        # Average pool
        pooled_a = self._average_pool(emb_a, lengths_a)
        pooled_b = self._average_pool(emb_b, lengths_b)

        # Encode
        enc_a = self.protein_encoder(pooled_a)
        enc_b = self.protein_encoder(pooled_b)

        # Interaction prediction
        combined = torch.cat([enc_a, enc_b], dim=-1)
        logits = self.interaction_head(combined)

        # Output
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

    def _average_pool(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Length-aware average pooling."""
        mask = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0)
        mask = mask < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        summed = (embeddings * mask).sum(dim=1)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(-1).float()
        return summed / lengths_clamped

    def _initialize_weights(self) -> None:
        """Initialize linear layers with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
