"""
Custom samplers used throughout DIPPI data loading.

Provides batch samplers used for class-imbalanced training, including
staged hard-negative sampling for finetune.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional, Sequence


class ImbalancedBatchSampler:
    """
    Batch sampler that maintains a target positive-to-negative ratio.

    Each epoch iterates through all positive indices once (without replacement)
    and samples negatives once per epoch (with replacement) to match the
    requested ratio.
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

        # Precompute batches of positives and required negatives for the epoch.
        batches: list[list[int]] = []
        neg_requirements: list[int] = []
        batch_size = self.pos_per_batch

        for start in range(0, len(pos_indices), batch_size):
            pos_batch = pos_indices[start : start + batch_size]
            if len(pos_batch) < batch_size and self.drop_last:
                break
            batches.append(pos_batch)
            neg_requirements.append(self._negatives_for_batch(len(pos_batch)))

        total_negs_needed = sum(neg_requirements)
        neg_pool = (
            self._rng.choices(self.neg_indices, k=total_negs_needed)
            if total_negs_needed > 0
            else []
        )

        neg_offset = 0
        for pos_batch, neg_count in zip(batches, neg_requirements):
            neg_batch = neg_pool[neg_offset : neg_offset + neg_count]
            neg_offset += neg_count

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


class StagedHardNegativeBatchSampler:
    """
    Batch sampler with epoch-aware staged hard-negative sampling.

    Each epoch uses all positive indices once (without replacement), then
    samples negatives with replacement to maintain the target ratio. During
    warmup epochs, negatives are sampled uniformly at random. After warmup,
    a configurable fraction of negatives is sampled from a hard pool derived
    from per-sample hard scores.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        pos_neg_ratio: float = 16.0,
        hard_scores: Optional[Sequence[float]] = None,
        warmup_epochs: int = 2,
        hard_ratio: float = 0.7,
        hard_score_top_fraction: Optional[float] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pos_neg_ratio <= 0:
            raise ValueError("pos_neg_ratio must be positive")
        if len(labels) == 0:
            raise ValueError("labels must be non-empty")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if not (0.0 <= hard_ratio <= 1.0):
            raise ValueError("hard_ratio must be in [0, 1]")
        if hard_score_top_fraction is not None and not (
            0.0 < hard_score_top_fraction <= 1.0
        ):
            raise ValueError("hard_score_top_fraction must be in (0, 1]")

        try:
            processed_labels = [int(label) for label in labels]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"labels must be convertible to integers: {exc}") from exc

        if any(label not in (0, 1) for label in processed_labels):
            raise ValueError("labels must be binary (0 or 1)")

        self.labels = processed_labels
        self.pos_indices = [idx for idx, label in enumerate(processed_labels) if label]
        self.neg_indices = [
            idx for idx, label in enumerate(processed_labels) if not label
        ]

        if not self.pos_indices:
            raise ValueError(
                "StagedHardNegativeBatchSampler requires at least one positive sample"
            )
        if not self.neg_indices:
            raise ValueError(
                "StagedHardNegativeBatchSampler requires at least one negative sample"
            )

        self.batch_size = batch_size
        self.pos_neg_ratio = float(pos_neg_ratio)
        self.warmup_epochs = int(warmup_epochs)
        self.hard_ratio = float(hard_ratio)
        self.hard_score_top_fraction = hard_score_top_fraction
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = random.Random(seed)
        self._epoch = 0

        self.pos_per_batch = self._compute_pos_per_batch()
        self.neg_per_batch = self._compute_neg_per_batch(self.pos_per_batch)

        self._hard_indices: List[int] = []
        self._hard_weights: List[float] = []
        self._hard_available = False
        self._prepare_hard_pool(hard_scores)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (0-based)."""
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self._epoch = epoch
        if self.seed is not None:
            self._rng = random.Random(self.seed + epoch)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices with staged hard-negative sampling."""
        pos_indices = list(self.pos_indices)
        if self.shuffle:
            self._rng.shuffle(pos_indices)

        hard_ratio = self._current_hard_ratio()

        batches: list[list[int]] = []
        neg_requirements: list[int] = []
        batch_size = self.pos_per_batch

        for start in range(0, len(pos_indices), batch_size):
            pos_batch = pos_indices[start : start + batch_size]
            if len(pos_batch) < batch_size and self.drop_last:
                break
            batches.append(pos_batch)
            neg_requirements.append(self._negatives_for_batch(len(pos_batch)))

        for pos_batch, neg_count in zip(batches, neg_requirements):
            hard_count = int(round(neg_count * hard_ratio))
            random_count = max(0, neg_count - hard_count)

            hard_batch = (
                self._sample_hard_negatives(hard_count) if hard_count > 0 else []
            )
            random_batch = (
                self._rng.choices(self.neg_indices, k=random_count)
                if random_count > 0
                else []
            )

            batch = pos_batch + hard_batch + random_batch
            self._rng.shuffle(batch)
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        """Return number of batches per epoch based on positive coverage."""
        full_batches, remainder = divmod(len(self.pos_indices), self.pos_per_batch)
        if self.drop_last:
            return full_batches
        return full_batches + (1 if remainder else 0)

    def _compute_pos_per_batch(self) -> int:
        denom = 1.0 + self.pos_neg_ratio
        raw = self.batch_size / denom
        return max(1, int(math.floor(raw)))

    def _compute_neg_per_batch(self, pos_per_batch: int) -> int:
        neg_per = int(round(pos_per_batch * self.pos_neg_ratio))
        return max(0, neg_per)

    def _negatives_for_batch(self, positive_count: int) -> int:
        if positive_count <= 0:
            return 0
        if positive_count == self.pos_per_batch:
            return self.neg_per_batch
        scaled = int(math.ceil(positive_count * self.pos_neg_ratio))
        return max(0, scaled)

    def _current_hard_ratio(self) -> float:
        if not self._hard_available or self._epoch < self.warmup_epochs:
            return 0.0
        return self.hard_ratio

    def _prepare_hard_pool(self, hard_scores: Optional[Sequence[float]]) -> None:
        if not hard_scores:
            return

        if len(hard_scores) == len(self.labels):
            neg_scores = [hard_scores[idx] for idx in self.neg_indices]
        elif len(hard_scores) == len(self.neg_indices):
            neg_scores = list(hard_scores)
        else:
            raise ValueError("hard_scores must match dataset length or negative count.")

        scored: list[tuple[int, float]] = []
        for neg_idx, score in zip(self.neg_indices, neg_scores):
            if score is None:
                continue
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                continue
            if math.isnan(score_val):
                continue
            scored.append((neg_idx, score_val))

        if not scored:
            return

        if self.hard_score_top_fraction is not None:
            scored.sort(key=lambda item: item[1], reverse=True)
            top_k = max(1, int(math.ceil(len(scored) * self.hard_score_top_fraction)))
            scored = scored[:top_k]

        indices, scores = zip(*scored)
        weights = self._normalize_weights(list(scores))

        self._hard_indices = list(indices)
        self._hard_weights = weights
        self._hard_available = True

    @staticmethod
    def _normalize_weights(scores: List[float]) -> List[float]:
        min_score = min(scores)
        shifted = [score - min_score for score in scores]
        max_shifted = max(shifted)
        if max_shifted <= 0:
            return [1.0 for _ in scores]
        return [score + 1e-6 for score in shifted]

    def _sample_hard_negatives(self, count: int) -> List[int]:
        if count <= 0 or not self._hard_available:
            return []
        return self._rng.choices(
            self._hard_indices, weights=self._hard_weights, k=count
        )


__all__ = ["ImbalancedBatchSampler", "StagedHardNegativeBatchSampler"]
