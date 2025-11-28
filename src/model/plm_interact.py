"""
Lightweight PLM-interact wrapper for DIPPI runner.

This wraps a Hugging Face ESM masked LM with a binary classifier head
matching the original PLM-interact inference code. It consumes raw
tokenized sequence pairs and returns logits (and optional loss).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM


class PLMInteractModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        embedding_size: int,
        classifier_dropout: float = 0.1,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(
            base_model_path, use_safetensors=False
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(embedding_size, 1)
        self.pos_weight = (
            torch.tensor([pos_weight], dtype=torch.float) if pos_weight else None
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Tokenized inputs and optional 'label' tensor.
        Returns:
            dict with 'logits' and, if labels provided, 'loss'.
        """
        labels = batch.get("label")
        model_inputs = {k: v for k, v in batch.items() if k not in {"label", "labels"}}

        outputs = self.esm_mask.base_model(**model_inputs, return_dict=True)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls = F.relu(cls)
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            labels = labels.float()
            loss_fn = (
                nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
                if self.pos_weight is not None
                else nn.BCEWithLogitsLoss()
            )
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result
