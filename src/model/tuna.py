"""TUnA (Transformer with Uncertainty-Aware predictions) model architecture.

This module provides a clean nn.Module implementation following MVP boundaries:
- No config parsing (kwargs provided by orchestrator)
- No logging, checkpointing, or metrics
- No training/testing logic (handled by Trainer/Evaluator)

Architecture adapted from the original TUnA paper implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Multi-head self-attention with spectral normalization."""

    def __init__(
        self, hid_dim: int, n_heads: int, dropout: float, device: torch.device
    ) -> None:
        super().__init__()
        if hid_dim % n_heads != 0:
            raise ValueError(
                f"hid_dim ({hid_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.w_q = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_k = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_v = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.fc = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = query.shape[0]

        # [batch, seq_len, hid_dim] -> [batch, seq_len, hid_dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # [batch, seq_len, hid_dim] -> [batch, n_heads, seq_len, head_dim]
        Q = Q.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention: [batch, n_heads, seq_len_Q, seq_len_K]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # [batch, n_heads, seq_len_Q, head_dim]
        x = torch.matmul(attention, V)

        # [batch, seq_len_Q, hid_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.hid_dim)

        return self.fc(x)


class Feedforward(nn.Module):
    """Position-wise feedforward network with spectral normalization."""

    def __init__(
        self, hid_dim: int, ff_dim: int, dropout: float, activation_fn: str
    ) -> None:
        super().__init__()
        self.fc_1 = spectral_norm(nn.Linear(hid_dim, ff_dim))
        self.fc_2 = spectral_norm(nn.Linear(ff_dim, hid_dim))
        self.do = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation_fn)

    def _get_activation_fn(self, activation_fn: str) -> nn.Module:
        """Return the corresponding activation function."""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "mish": nn.Mish(),
        }
        if activation_fn not in activation_map:
            raise ValueError(
                f"Activation function '{activation_fn}' not supported. Choose from {list(activation_map.keys())}"
            )
        return activation_map[activation_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, hid_dim]
        x = self.do(self.activation(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm architecture."""

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_fn: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ff = Feedforward(hid_dim, ff_dim, dropout, activation_fn)

    def forward(
        self, trg: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm with residual connections
        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, mask)))
        trg = self.ln2(trg + self.do2(self.ff(trg)))
        return trg


class IntraEncoder(nn.Module):
    """Intra-protein encoder: processes individual protein representations."""

    def __init__(
        self,
        prot_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_fn: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.ft = spectral_norm(nn.Linear(prot_dim, hid_dim))
        self.n_layers = n_layers
        self.layer = nn.ModuleList(
            EncoderLayer(hid_dim, n_heads, ff_dim, dropout, activation_fn, device)
            for _ in range(n_layers)
        )

    def forward(
        self, trg: torch.Tensor, trg_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # [batch, seq_len, prot_dim] -> [batch, seq_len, hid_dim]
        trg = self.ft(trg)
        for layer in self.layer:
            trg = layer(trg, trg_mask)
        return trg


class InterEncoder(nn.Module):
    """Inter-protein encoder: processes concatenated protein pair representations."""

    def __init__(
        self,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_fn: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.layer = nn.ModuleList(
            EncoderLayer(hid_dim, n_heads, ff_dim, dropout, activation_fn, device)
            for _ in range(n_layers)
        )

    def forward(
        self,
        enc_protA: torch.Tensor,
        enc_protB: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate along sequence dimension: [batch, lenA+lenB, hid_dim]
        combined_trg_src = torch.cat([enc_protA, enc_protB], dim=1)

        for layer in self.layer:
            combined_trg_src = layer(combined_trg_src, combined_mask)

        # Global average pooling with mask: [batch, hid_dim]
        # Extract 2D mask [batch, lenA+lenB] from 4D mask [batch, 1, lenA+lenB, lenA+lenB]
        combined_mask_2d = combined_mask[:, 0, :, 0]
        label = torch.sum(
            combined_trg_src * combined_mask_2d[:, :, None], dim=1
        ) / combined_mask_2d.sum(dim=1, keepdims=True)

        return label


class TUnA(nn.Module):
    """TUnA model for protein-protein interaction prediction.

    Architecture:
    1. IntraEncoder: processes each protein separately with self-attention
    2. InterEncoder: processes concatenated protein pairs (bidirectional AB and BA)
    3. Max pooling over AB and BA representations
    4. Linear output head for binary classification logits

    Public interface (orchestrator contract):
    - Constructor: accepts kwargs from orchestrator (no config parsing)
    - forward(batch: Dict[str, Any]) -> Dict[str, Tensor]
      - Input: {"emb_a": [B, L_a, D], "emb_b": [B, L_b, D], "len_a": [B], "len_b": [B]}
      - Output: {"logits": [B, 1]}
    """

    name: str = "tuna"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()

        # Validate required fields
        required_fields = [
            "input_dim",
            "d_model",
            "intra_layers",
            "inter_layers",
            "n_heads",
            "ff_dim",
            "dropout",
            "activation",
        ]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        # Extract and store config
        self.input_dim: int = int(model_config["input_dim"])
        self.d_model: int = int(model_config["d_model"])
        self.intra_layers: int = int(model_config["intra_layers"])
        self.inter_layers: int = int(model_config["inter_layers"])
        self.n_heads: int = int(model_config["n_heads"])
        self.ff_dim: int = int(model_config["ff_dim"])
        self.dropout: float = float(model_config["dropout"])
        self.activation: str = str(model_config["activation"])

        # Device handling (will be set by orchestrator via .to(device))
        self.device = torch.device("cpu")

        # Optional unused fields for config compatibility
        self._unused_spectral_norm = model_config.get("spectral_norm", True)
        self._unused_gp_layer = model_config.get("gp_layer", None)

        # Build architecture
        self.intra_encoder = IntraEncoder(
            prot_dim=self.input_dim,
            hid_dim=self.d_model,
            n_layers=self.intra_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            activation_fn=self.activation,
            device=self.device,
        )

        self.inter_encoder = InterEncoder(
            hid_dim=self.d_model,
            n_layers=self.inter_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            activation_fn=self.activation,
            device=self.device,
        )

        # Simple linear output (replaces GP layer for MVP)
        self.output_head = spectral_norm(nn.Linear(self.d_model, 1))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform for layers with dim > 1."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to(self, *args, **kwargs):
        """Override to() to update device tracking."""
        self = super().to(*args, **kwargs)
        # Extract device from args/kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        # Update device in submodules
        for module in self.modules():
            if isinstance(module, (SelfAttention,)):
                module.scale = module.scale.to(self.device)
        return self

    def _make_masks(
        self, prot_lens: torch.Tensor, protein_max_len: int
    ) -> torch.Tensor:
        """Create 4D attention masks [batch, 1, max_len, max_len] from sequence lengths."""
        N = len(prot_lens)
        mask = torch.zeros((N, protein_max_len, protein_max_len), device=self.device)

        for i, lens in enumerate(prot_lens):
            mask[i, :lens, :lens] = 1

        # Expand to 4D: [batch, 1, max_len, max_len]
        return mask.unsqueeze(1)

    def _combine_masks(self, maskA: torch.Tensor, maskB: torch.Tensor) -> torch.Tensor:
        """Combine two masks into a block-diagonal concatenated mask."""
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(
            maskA.size(0), 1, lenA + lenB, lenA + lenB, device=self.device
        )
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for PPI prediction.

        Args:
            batch: Dictionary containing:
                - "emb_a": Tensor[B, L_a, input_dim] - Protein A embeddings
                - "emb_b": Tensor[B, L_b, input_dim] - Protein B embeddings
                - "len_a": Tensor[B] - Sequence lengths for A (optional, defaults to L_a)
                - "len_b": Tensor[B] - Sequence lengths for B (optional, defaults to L_b)

        Returns:
            Dictionary containing:
                - "logits": Tensor[B, 1] - Interaction prediction logits
        """
        # Validate inputs
        if "emb_a" not in batch or "emb_b" not in batch:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = batch["emb_a"]
        emb_b = batch["emb_b"]

        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError(
                "Input embeddings must be shaped (batch, seq_len, embedding_dim)"
            )
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError(
                f"Input embedding dimension must match model input_dim ({self.input_dim})"
            )
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        # Extract lengths (default to full sequence if not provided)
        device = emb_a.device
        self.device = device  # Update device tracking

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

        # Create masks
        batch_protA_max_length = emb_a.size(1)
        batch_protB_max_length = emb_b.size(1)

        protA_mask = self._make_masks(lengths_a, batch_protA_max_length)
        protB_mask = self._make_masks(lengths_b, batch_protB_max_length)

        # Intra-protein encoding
        enc_protA = self.intra_encoder(emb_a, protA_mask)  # [B, L_a, d_model]
        enc_protB = self.intra_encoder(emb_b, protB_mask)  # [B, L_b, d_model]

        # Inter-protein encoding (bidirectional: AB and BA)
        combined_mask_AB = self._combine_masks(protA_mask, protB_mask)
        combined_mask_BA = self._combine_masks(protB_mask, protA_mask)

        AB_interaction = self.inter_encoder(
            enc_protA, enc_protB, combined_mask_AB
        )  # [B, d_model]
        BA_interaction = self.inter_encoder(
            enc_protB, enc_protA, combined_mask_BA
        )  # [B, d_model]

        # Max pooling over AB and BA: [B, d_model]
        ppi_feature_vector, _ = torch.max(
            torch.stack([AB_interaction, BA_interaction], dim=-1), dim=-1
        )

        # Output logits: [B, 1]
        logits = self.output_head(ppi_feature_vector)

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
            loss = F.binary_cross_entropy_with_logits(logits_for_loss, labels_for_loss)
            output["loss"] = loss

        return output
