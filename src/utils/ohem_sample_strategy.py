"""Online hard example mining selection logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def select_ohem_indices(
    losses: torch.Tensor,
    keep_ratio: float = 0.5,
    min_keep: int = 1,
) -> torch.Tensor:
    """Select hardest-sample indices from per-sample losses.

    Args:
        losses: One-dimensional loss tensor with shape ``(batch_size,)``.
        keep_ratio: Ratio of hardest examples to keep in ``(0, 1]``.
        min_keep: Minimum number of selected samples.

    Returns:
        Tensor of indices sorted by descending loss.
    """
    if losses.dim() != 1:
        raise ValueError("losses must be a 1D tensor")
    if losses.numel() == 0:
        raise ValueError("losses must be non-empty")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if min_keep <= 0:
        raise ValueError("min_keep must be positive")

    keep_count = max(min_keep, int(math.ceil(losses.numel() * keep_ratio)))
    keep_count = min(keep_count, int(losses.numel()))
    detached = losses.detach()
    return torch.topk(detached, k=keep_count, largest=True, sorted=True).indices


@dataclass(frozen=True)
class OHEMSampleStrategy:
    """Configuration for OHEM sample selection.

    Attributes:
        target_batch_size: Number of hard samples selected per optimization step.
        cap_protein: Maximum count for one protein in selected pairs.
        warmup_epochs: Number of initial epochs that bypass OHEM.
    """

    target_batch_size: int = 1
    cap_protein: int = 4
    warmup_epochs: int = 0

    def _select_with_cap(
        self,
        sorted_indices: torch.Tensor,
        protein_a_ids: torch.Tensor,
        protein_b_ids: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        selected: list[int] = []
        remaining: list[int] = []
        counts: dict[int, int] = {}
        sorted_list = [int(index) for index in sorted_indices.detach().cpu().tolist()]
        protein_a_list = [int(protein_id) for protein_id in protein_a_ids.detach().cpu().tolist()]
        protein_b_list = [int(protein_id) for protein_id in protein_b_ids.detach().cpu().tolist()]

        for sample_index in sorted_list:
            protein_a = protein_a_list[sample_index]
            protein_b = protein_b_list[sample_index]
            count_a = counts.get(protein_a, 0)
            count_b = counts.get(protein_b, 0)
            if protein_a == protein_b:
                if count_a + 2 > self.cap_protein:
                    remaining.append(sample_index)
                    continue
                counts[protein_a] = count_a + 2
            else:
                if count_a + 1 > self.cap_protein or count_b + 1 > self.cap_protein:
                    remaining.append(sample_index)
                    continue
                counts[protein_a] = count_a + 1
                counts[protein_b] = count_b + 1
            selected.append(sample_index)
            if len(selected) >= keep_count:
                break

        if len(selected) < keep_count:
            for sample_index in remaining:
                selected.append(sample_index)
                if len(selected) >= keep_count:
                    break

        return torch.tensor(selected, device=sorted_indices.device, dtype=torch.long)

    def select(
        self,
        losses: torch.Tensor,
        epoch_index: int = 0,
        protein_a_ids: torch.Tensor | None = None,
        protein_b_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select hard-sample indices.

        Args:
            losses: Per-sample loss tensor.
            epoch_index: Zero-based epoch index.
            protein_a_ids: Optional protein ids for first protein in each pair.
            protein_b_ids: Optional protein ids for second protein in each pair.

        Returns:
            Indices of selected hard samples.
        """
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive")
        if self.cap_protein <= 0:
            raise ValueError("cap_protein must be positive")
        if epoch_index < self.warmup_epochs:
            return torch.arange(losses.numel(), device=losses.device)
        if losses.dim() != 1:
            raise ValueError("losses must be a 1D tensor")
        if losses.numel() == 0:
            raise ValueError("losses must be non-empty")

        keep_count = min(int(losses.numel()), int(self.target_batch_size))
        sorted_indices = torch.argsort(losses.detach(), descending=True)

        if protein_a_ids is None or protein_b_ids is None:
            return sorted_indices[:keep_count]
        if protein_a_ids.dim() != 1 or protein_b_ids.dim() != 1:
            raise ValueError("protein ids must be 1D tensors")
        if protein_a_ids.numel() != losses.numel() or protein_b_ids.numel() != losses.numel():
            raise ValueError("protein ids must align with losses")

        return self._select_with_cap(
            sorted_indices=sorted_indices,
            protein_a_ids=protein_a_ids,
            protein_b_ids=protein_b_ids,
            keep_count=keep_count,
        )
