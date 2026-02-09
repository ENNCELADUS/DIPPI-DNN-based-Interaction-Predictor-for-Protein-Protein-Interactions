"""V6 PPI Classifier - ESM3 backbone with LoRA + token-level cross-attention head."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.model.v3 import _build_padding_mask


class LoRALinear(nn.Module):
    """LoRA wrapper for a Linear layer (trainable low-rank adapters only)."""

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank r must be non-negative")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Freeze base weights
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if self.r > 0:
            self.lora_a = nn.Linear(self.base.in_features, self.r, bias=False)
            self.lora_b = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.r > 0 and self.lora_a is not None and self.lora_b is not None:
            result = result + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return result


class MLPHead(nn.Module):
    """Fixed 4d -> 2d -> d -> 1 MLP with GELU, LayerNorm, and dropout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        hidden_2d = 2 * d_model
        self.layers = nn.Sequential(
            nn.Linear(4 * d_model, hidden_2d),
            nn.LayerNorm(hidden_2d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2d, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _resolve_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint did not contain a state_dict-like object")
    return state


def _get_parent_module(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    last_n_layers: int,
    target_modules: Optional[Iterable[str]],
    r: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    """Inject LoRA into selected Linear layers in the last N transformer blocks."""
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "blocks"):
        raise ValueError("ESM3 model missing expected transformer.blocks attribute")

    blocks = model.transformer.blocks
    total_blocks = len(blocks)
    if total_blocks == 0:
        raise ValueError("ESM3 transformer.blocks is empty")

    n_layers = min(max(int(last_n_layers), 1), total_blocks)
    start_idx = total_blocks - n_layers
    target_prefixes = [
        f"transformer.blocks.{i}." for i in range(start_idx, total_blocks)
    ]

    if target_modules is None:
        target_list: List[str] = []
    elif isinstance(target_modules, str):
        target_list = [target_modules]
    else:
        target_list = [str(item) for item in target_modules]

    matched: List[str] = []
    candidate_names: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.startswith(prefix) for prefix in target_prefixes):
            continue
        if target_list and not any(substr in name for substr in target_list):
            continue
        candidate_names.append(name)

    for name in candidate_names:
        parent, attr = _get_parent_module(model, name)
        existing = getattr(parent, attr)
        if isinstance(existing, LoRALinear):
            continue
        setattr(parent, attr, LoRALinear(existing, r=r, alpha=alpha, dropout=dropout))
        matched.append(name)

    if not matched:
        warnings.warn(
            "No target Linear layers matched for LoRA injection. "
            "Check lora.target_modules and last_n_layers.",
            stacklevel=2,
        )

    return matched


class SharedCrossAttentionLayer(nn.Module):
    """Shared-weight bidirectional cross-attention with FFN."""

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)

        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        return h_a, h_b


class CLSPooling(nn.Module):
    """Cross-attention pooling with learnable CLS tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.cls_a = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_b = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_a, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_b, mean=0.0, std=0.02)

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)

    def _pool(
        self,
        cls_token: torch.Tensor,
        combined: torch.Tensor,
        combined_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cls_norm = self.norm_attn(cls_token)
        attn_out, _ = self.attn(
            cls_norm, combined, combined, key_padding_mask=combined_mask
        )
        cls_token = cls_token + self.drop_attn(attn_out)
        cls_token = cls_token + self.drop_ffn(self.ffn(self.norm_ffn(cls_token)))
        return cls_token

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        mask_b: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_a.size(0)
        cls_a = self.cls_a.expand(batch_size, -1, -1)
        cls_b = self.cls_b.expand(batch_size, -1, -1)

        combined_a = torch.cat([h_a, h_b], dim=1)
        combined_b = torch.cat([h_b, h_a], dim=1)

        if mask_a is not None and mask_b is not None:
            mask_ab = torch.cat([mask_a, mask_b], dim=1)
            mask_ba = torch.cat([mask_b, mask_a], dim=1)
        else:
            mask_ab = None
            mask_ba = None

        cls_a = self._pool(cls_a, combined_a, mask_ab)
        cls_b = self._pool(cls_b, combined_b, mask_ba)

        return cls_a.squeeze(1), cls_b.squeeze(1)


class V6(nn.Module):
    """V6 PPI classifier with ESM3 backbone and LoRA adaptation."""

    name: str = "v6"

    def __init__(self, **model_config: Any) -> None:
        super().__init__()

        required_fields = ["d_model", "cross_attn_layers", "n_heads", "mlp_head"]
        missing = [field for field in required_fields if field not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim = int(model_config.get("input_dim", 1536))
        self.d_model = int(model_config["d_model"])
        self.cross_attn_layers = int(model_config["cross_attn_layers"])
        self.n_heads = int(model_config["n_heads"])

        esm_cfg: Dict[str, Any] = model_config.get("esm3", {})
        self.esm3_model_name = esm_cfg.get("model_name", "esm3-open")
        self.esm3_checkpoint_path = Path(
            esm_cfg.get("checkpoint_path", "models/esm3/esm3_sm_open_v1_full.pth")
        )
        self.strip_cls_eos = bool(esm_cfg.get("strip_cls_eos", True))
        embed_batch_size = int(esm_cfg.get("embed_batch_size", 0))
        self.esm3_embed_batch_size = embed_batch_size if embed_batch_size > 0 else None
        self.combine_pairs = bool(esm_cfg.get("combine_pairs", True))

        lora_cfg: Dict[str, Any] = model_config.get("lora", {})
        self.lora_last_n_layers = int(lora_cfg.get("last_n_layers", 8))
        self.lora_target_modules = lora_cfg.get(
            "target_modules", ["layernorm_qkv", "out_proj", "ffn"]
        )
        self.lora_r = int(lora_cfg.get("r", 8))
        self.lora_alpha = int(lora_cfg.get("alpha", 16))
        self.lora_dropout = float(lora_cfg.get("dropout", 0.05))

        reg_cfg: Dict[str, Any] = model_config.get("regularization", {})
        self.dropout = float(reg_cfg.get("dropout", 0.1))
        self.cross_attention_dropout = float(
            reg_cfg.get("cross_attention_dropout", self.dropout)
        )
        self.projection_dropout = float(reg_cfg.get("projection_dropout", self.dropout))

        mlp_cfg: Dict[str, Any] = model_config.get("mlp_head", {})
        if not mlp_cfg:
            raise ValueError("mlp_head configuration is required for V6")
        if "dropout" not in mlp_cfg:
            raise ValueError("mlp_head.dropout must be provided for V6")
        self.mlp_dropout = float(mlp_cfg["dropout"])

        self.esm3 = self._load_esm3()
        self._esm_protein_cls, self._logits_config = self._load_esm_sdk()
        self._apply_lora_to_esm3()
        self._esm3_batch_encode_supported: Optional[bool] = None

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.projection_dropout),
        )

        self.cross_layers = nn.ModuleList(
            SharedCrossAttentionLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.cross_attention_dropout,
            )
            for _ in range(self.cross_attn_layers)
        )

        self.cls_pool = CLSPooling(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.cross_attention_dropout,
        )

        self.output_head = MLPHead(d_model=self.d_model, dropout=self.mlp_dropout)

    def _load_esm3(self) -> nn.Module:
        try:
            from esm.models.esm3 import ESM3
        except ImportError as exc:
            raise ImportError(
                "ESM3 is not installed. Please run: conda activate esm"
            ) from exc

        model = ESM3.from_pretrained(self.esm3_model_name)

        if not self.esm3_checkpoint_path.exists():
            raise FileNotFoundError(
                f"ESM3 checkpoint not found: {self.esm3_checkpoint_path}"
            )

        state_dict = _resolve_state_dict(self.esm3_checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                "ESM3 checkpoint mismatch. "
                f"Missing keys: {missing}, unexpected keys: {unexpected}"
            )

        return model

    def _apply_lora_to_esm3(self) -> None:
        for param in self.esm3.parameters():
            param.requires_grad = False

        if self.lora_r <= 0:
            return

        apply_lora(
            self.esm3,
            last_n_layers=self.lora_last_n_layers,
            target_modules=self.lora_target_modules,
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )

    def _load_esm_sdk(self) -> Tuple[Type[Any], Any]:
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise ImportError(
                "ESM3 SDK is not installed. Please run: conda activate esm"
            ) from exc

        return ESMProtein, LogitsConfig(sequence=True, return_embeddings=True)

    def _embed_chunk(
        self, sequences: Sequence[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if any(not isinstance(seq, str) for seq in sequences):
            raise TypeError("Sequences must be raw strings")
        if not sequences:
            raise ValueError("Sequences must be non-empty")

        if self._esm3_batch_encode_supported is False:
            return self._embed_chunk_serial(sequences)

        proteins = [self._esm_protein_cls(sequence=seq) for seq in sequences]
        try:
            protein_tensor = self.esm3.encode(proteins)
        except Exception as exc:
            try:
                from attr.exceptions import NotAnAttrsClassError
            except ImportError:
                NotAnAttrsClassError = None  # type: ignore[assignment]

            if NotAnAttrsClassError is not None and isinstance(
                exc, NotAnAttrsClassError
            ):
                self._esm3_batch_encode_supported = False
                warnings.warn(
                    "ESM3 encode does not accept batched inputs; falling back to "
                    "per-sequence encoding. Expect slower throughput.",
                    RuntimeWarning,
                )
                return self._embed_chunk_serial(sequences)
            raise
        else:
            self._esm3_batch_encode_supported = True

        output = self.esm3.logits(protein_tensor, self._logits_config)
        embeddings = output.embeddings
        if embeddings is None:
            raise ValueError("ESM3 logits did not return embeddings")

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        seq_lengths = torch.tensor(
            [len(seq) for seq in sequences],
            device=embeddings.device,
            dtype=torch.long,
        )

        if self.strip_cls_eos:
            stripped: List[torch.Tensor] = []
            for idx, seq_len in enumerate(seq_lengths.tolist()):
                if seq_len <= 0:
                    stripped.append(embeddings[idx, :0])
                else:
                    stripped.append(embeddings[idx, 1 : 1 + seq_len])
            return stripped, seq_lengths

        kept: List[torch.Tensor] = []
        for idx, seq_len in enumerate(seq_lengths.tolist()):
            keep_len = seq_len + 2
            kept.append(embeddings[idx, :keep_len])
        return kept, seq_lengths + 2

    def _embed_chunk_serial(
        self, sequences: Sequence[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        embeddings_list: List[torch.Tensor] = []
        lengths_list: List[int] = []

        for seq in sequences:
            protein = self._esm_protein_cls(sequence=seq)
            protein_tensor = self.esm3.encode(protein)
            output = self.esm3.logits(protein_tensor, self._logits_config)
            embeddings = output.embeddings
            if embeddings is None:
                raise ValueError("ESM3 logits did not return embeddings")

            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)

            if embeddings.size(0) != 1:
                raise ValueError("ESM3 serial encode returned a batch size != 1")

            embedding = embeddings.squeeze(0)
            seq_len = len(seq)
            if self.strip_cls_eos:
                embedding = embedding[1 : 1 + seq_len] if seq_len > 0 else embedding[:0]
                lengths_list.append(seq_len)
            else:
                keep_len = seq_len + 2
                embedding = embedding[:keep_len]
                lengths_list.append(keep_len)

            embeddings_list.append(embedding)

        device = embeddings_list[0].device
        lengths = torch.tensor(lengths_list, device=device, dtype=torch.long)
        return embeddings_list, lengths

    def _embed_batch(
        self, sequences: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("Sequences must be non-empty")
        batch_size = self.esm3_embed_batch_size or len(sequences)
        if batch_size <= 0:
            batch_size = len(sequences)

        chunk_embeddings: List[torch.Tensor] = []
        chunk_lengths: List[torch.Tensor] = []
        for start in range(0, len(sequences), batch_size):
            end = start + batch_size
            emb, lengths = self._embed_chunk(sequences[start:end])
            chunk_embeddings.extend(emb)
            chunk_lengths.append(lengths)

        padded = pad_sequence(chunk_embeddings, batch_first=True)
        lengths = torch.cat(chunk_lengths, dim=0)
        return padded, lengths

    @staticmethod
    def _pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, (a - b).abs(), a * b], dim=-1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "seq_a" not in batch or "seq_b" not in batch:
            raise KeyError("Batch must contain 'seq_a' and 'seq_b' raw sequences")

        seq_a = batch["seq_a"]
        seq_b = batch["seq_b"]

        if isinstance(seq_a, str) or isinstance(seq_b, str):
            raise TypeError("seq_a and seq_b must be sequences of strings")
        if not isinstance(seq_a, (list, tuple)) or not isinstance(seq_b, (list, tuple)):
            raise TypeError("seq_a and seq_b must be lists or tuples of strings")
        if any(not isinstance(item, str) for item in seq_a) or any(
            not isinstance(item, str) for item in seq_b
        ):
            raise TypeError("seq_a and seq_b must contain only strings")

        if len(seq_a) != len(seq_b):
            raise ValueError("Protein pair batches must have matching batch dimension")

        if self.combine_pairs:
            combined = list(seq_a) + list(seq_b)
            combined_emb, combined_lengths = self._embed_batch(combined)
            batch_size = len(seq_a)
            emb_a = combined_emb[:batch_size]
            emb_b = combined_emb[batch_size:]
            lengths_a = combined_lengths[:batch_size]
            lengths_b = combined_lengths[batch_size:]
        else:
            emb_a, lengths_a = self._embed_batch(seq_a)
            emb_b, lengths_b = self._embed_batch(seq_b)

        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("ESM3 embedding dimension does not match input_dim")

        h_a = self.projection(emb_a)
        h_b = self.projection(emb_b)

        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))

        for layer in self.cross_layers:
            h_a, h_b = layer(h_a, h_b, mask_a, mask_b)

        cls_a, cls_b = self.cls_pool(h_a, h_b, mask_a, mask_b)
        pair_features = self._pair_features(cls_a, cls_b)
        logits = self.output_head(pair_features)

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
