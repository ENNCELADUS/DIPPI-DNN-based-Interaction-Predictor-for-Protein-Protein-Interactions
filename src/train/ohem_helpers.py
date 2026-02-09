"""Helper functions for Trainer OHEM batch preparation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def compute_mining_scores(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize_logits: Callable[[torch.Tensor], torch.Tensor],
    normalize_labels: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute per-sample OHEM mining scores from logits and labels."""
    normalized_logits = normalize_logits(logits)
    normalized_labels = normalize_labels(labels)
    return F.binary_cross_entropy_with_logits(
        normalized_logits,
        normalized_labels,
        reduction="none",
    ).view(-1)


def index_batch(batch: Dict[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
    """Slice a heterogeneous batch dictionary with tensor indices."""
    if indices.numel() == 0:
        return {}
    index_list = indices.detach().cpu().tolist()
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value.index_select(0, indices)
        elif isinstance(value, list):
            sliced[key] = [value[i] for i in index_list]
        elif isinstance(value, tuple):
            sliced[key] = tuple(value[i] for i in index_list)
        elif isinstance(value, np.ndarray):
            sliced[key] = value[index_list]
        else:
            sliced[key] = value
    return sliced


def prepare_ohem_batch(
    batch: Dict[str, Any],
    *,
    model: torch.nn.Module,
    move_batch_to_device: Callable[[Dict[str, Any]], Dict[str, Any]],
    amp_context: Callable[[], Any],
    compute_scores: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    index_batch_fn: Callable[[Dict[str, Any], torch.Tensor], Dict[str, Any]],
) -> Dict[str, Any]:
    """Select hard examples from OHEM candidate pool and build training batch."""
    pool = batch.get("pool")
    if pool is None:
        raise RuntimeError("OHEM batch is missing the candidate pool.")

    target_batch_size = int(batch.get("ohem_batch_size", 0)) or int(
        batch.get("batch_size", 0)
    )
    if target_batch_size <= 0:
        target_batch_size = int(pool.get("label", torch.empty(0)).size(0))

    cap_protein = int(batch.get("cap_protein", 0))
    protein_a = pool.get("protein_a")
    protein_b = pool.get("protein_b")
    if protein_a is None or protein_b is None:
        raise ValueError("OHEM mining requires protein IDs in the pool batch.")

    was_training = model.training
    model.eval()
    pool_device = move_batch_to_device(pool)

    with torch.no_grad():
        with amp_context():
            outputs = model(pool_device)
            if not isinstance(outputs, dict) or "logits" not in outputs:
                raise ValueError(
                    "OHEM mining requires model outputs to include 'logits'."
                )
            logits = outputs["logits"]
            labels = pool_device["label"]
            scores = compute_scores(logits, labels)

    if was_training:
        model.train()

    if scores.numel() == 0:
        raise RuntimeError("OHEM mining produced no candidate scores.")

    sorted_idx = torch.argsort(scores, descending=True).tolist()
    selected: List[int] = []

    if cap_protein > 0:
        counts: Dict[str, int] = {}
        for idx in sorted_idx:
            protein_left = protein_a[idx]
            protein_right = protein_b[idx]
            if counts.get(protein_left, 0) >= cap_protein:
                continue
            if counts.get(protein_right, 0) >= cap_protein:
                continue
            selected.append(idx)
            counts[protein_left] = counts.get(protein_left, 0) + 1
            counts[protein_right] = counts.get(protein_right, 0) + 1
            if len(selected) >= target_batch_size:
                break

    if len(selected) < target_batch_size:
        selected_set = set(selected)
        for idx in sorted_idx:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= target_batch_size:
                break

    if not selected:
        raise RuntimeError("OHEM selection produced an empty training batch.")

    selected_tensor = torch.tensor(
        selected, device=pool_device["label"].device, dtype=torch.long
    )
    train_batch = index_batch_fn(pool_device, selected_tensor)
    train_batch["_ohem_unweighted"] = True
    return train_batch
