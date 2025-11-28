"""
Custom samplers used throughout DIPPI data loading.

Currently provides an ImbalancedBatchSampler that enforces a target
positive-to-negative ratio (default 1:3) while ensuring each positive
sample is seen exactly once per epoch.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional, Sequence


class ImbalancedBatchSampler:
    """
    Batch sampler that maintains a target positive-to-negative ratio.

    Each epoch iterates through all positive indices once (without replacement)
    and samples negatives with replacement to match the requested ratio.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        pos_neg_ratio: float = 3.0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            labels: Binary labels (0 for negative, 1 for positive) for the dataset.
            batch_size: Desired batch size used to derive per-class counts.
            pos_neg_ratio: Number of negatives per positive in each batch.
            shuffle: Whether to shuffle positive indices every epoch.
            drop_last: Drop the final batch if it has fewer positives than expected.
            seed: Optional random seed for reproducibility.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pos_neg_ratio <= 0:
            raise ValueError("pos_neg_ratio must be positive")
        if len(labels) == 0:
            raise ValueError("labels must be non-empty")

        try:
            processed_labels = [int(label) for label in labels]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"labels must be convertible to integers: {exc}") from exc

        if any(label not in (0, 1) for label in processed_labels):
            raise ValueError("labels must be binary (0 or 1)")

        self.pos_indices = [idx for idx, label in enumerate(processed_labels) if label]
        self.neg_indices = [
            idx for idx, label in enumerate(processed_labels) if not label
        ]

        if not self.pos_indices:
            raise ValueError(
                "ImbalancedBatchSampler requires at least one positive sample"
            )
        if not self.neg_indices:
            raise ValueError(
                "ImbalancedBatchSampler requires at least one negative sample"
            )

        self.batch_size = batch_size
        self.pos_neg_ratio = float(pos_neg_ratio)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = random.Random(seed)

        self.pos_per_batch = self._compute_pos_per_batch()
        self.neg_per_batch = self._compute_neg_per_batch(self.pos_per_batch)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices with the configured ratio."""
        pos_indices = list(self.pos_indices)
        if self.shuffle:
            self._rng.shuffle(pos_indices)

        batch_size = self.pos_per_batch

        for start in range(0, len(pos_indices), batch_size):
            pos_batch = pos_indices[start : start + batch_size]
            if len(pos_batch) < batch_size and self.drop_last:
                break

            neg_count = self._negatives_for_batch(len(pos_batch))
            neg_batch = (
                self._rng.choices(self.neg_indices, k=neg_count)
                if neg_count > 0
                else []
            )

            batch = pos_batch + neg_batch
            self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Return number of batches per epoch based on positive coverage."""
        full_batches, remainder = divmod(len(self.pos_indices), self.pos_per_batch)
        if self.drop_last:
            return full_batches
        return full_batches + (1 if remainder else 0)

    def _compute_pos_per_batch(self) -> int:
        """Derive the number of positives per batch from batch_size and ratio."""
        denom = 1.0 + self.pos_neg_ratio
        raw = self.batch_size / denom
        pos_per_batch = max(1, int(math.floor(raw)))
        return pos_per_batch

    def _compute_neg_per_batch(self, pos_per_batch: int) -> int:
        """Compute negatives per batch based on the ratio."""
        neg_per = int(round(pos_per_batch * self.pos_neg_ratio))
        return max(0, neg_per)

    def _negatives_for_batch(self, positive_count: int) -> int:
        """Scale negatives for partial batches at epoch end."""
        if positive_count <= 0:
            return 0
        if positive_count == self.pos_per_batch:
            return self.neg_per_batch
        scaled = int(math.ceil(positive_count * self.pos_neg_ratio))
        return max(0, scaled)


__all__ = ["ImbalancedBatchSampler"]
