"""
Unit tests for src/utils/samplers.py.
"""

import math

import pytest

from src.utils.samplers import ImbalancedBatchSampler, OnlineHardNegativeBatchSampler


class TestImbalancedBatchSampler:
    """Tests for the ImbalancedBatchSampler."""

    def test_sampler_covers_all_positives_once(self):
        """All positive indices should appear exactly once per epoch."""
        labels = [1] * 10 + [0] * 60
        sampler = ImbalancedBatchSampler(
            labels=labels,
            batch_size=16,
            pos_neg_ratio=3.0,
            shuffle=False,
            drop_last=False,
            seed=123,
        )

        batches = list(iter(sampler))
        expected_batches = math.ceil(
            len([1 for x in labels if x == 1]) / sampler.pos_per_batch
        )
        assert len(batches) == expected_batches

        seen = set()
        for batch in batches:
            positives = [idx for idx in batch if idx < 10]
            seen.update(positives)
            # All non-negative samples must come from remaining indices
            assert all(idx >= 10 for idx in batch if idx not in positives)

        assert seen == set(range(10))

    def test_sampler_respects_ratio_for_full_batches(self):
        """Full batches should use configured pos/neg counts."""
        labels = [1] * 12 + [0] * 24
        sampler = ImbalancedBatchSampler(
            labels=labels,
            batch_size=20,
            pos_neg_ratio=2.0,
            shuffle=False,
            drop_last=True,
            seed=7,
        )

        batches = list(iter(sampler))
        # With drop_last=True and 12 positives, expect floor(12 / pos_per_batch) batches
        assert len(batches) == len(labels[:12]) // sampler.pos_per_batch

        for batch in batches:
            pos_count = sum(1 for idx in batch if idx < 12)
            neg_count = len(batch) - pos_count
            assert pos_count == sampler.pos_per_batch
            assert neg_count == sampler.neg_per_batch

    def test_sampler_drop_last_discards_partial_batch(self):
        """drop_last should remove final partial positive chunk."""
        labels = [1] * 5 + [0] * 20
        sampler = ImbalancedBatchSampler(
            labels=labels,
            batch_size=12,
            pos_neg_ratio=3.0,
            shuffle=False,
            drop_last=True,
        )

        # pos_per_batch will be 3 so floor(5/3)=1 batch when drop_last=True
        batches = list(iter(sampler))
        assert len(batches) == 1

    def test_sampler_seed_reproducibility(self):
        """Same seed should yield identical batches even when shuffling."""
        labels = [1] * 8 + [0] * 40
        sampler_a = ImbalancedBatchSampler(
            labels=labels,
            batch_size=16,
            pos_neg_ratio=3.0,
            shuffle=True,
            seed=99,
        )
        sampler_b = ImbalancedBatchSampler(
            labels=labels,
            batch_size=16,
            pos_neg_ratio=3.0,
            shuffle=True,
            seed=99,
        )

        assert list(iter(sampler_a)) == list(iter(sampler_b))

    def test_sampler_requires_both_classes(self):
        """Sampler should raise if positives or negatives are missing."""
        with pytest.raises(ValueError, match="requires at least one positive"):
            ImbalancedBatchSampler(labels=[0, 0, 0], batch_size=8)

        with pytest.raises(ValueError, match="requires at least one negative"):
            ImbalancedBatchSampler(labels=[1, 1, 1], batch_size=8)


class TestOnlineHardNegativeBatchSampler:
    """Tests for OnlineHardNegativeBatchSampler."""

    def test_ohem_roles_and_counts(self):
        labels = [1] * 10 + [0] * 50
        sampler = OnlineHardNegativeBatchSampler(
            labels=labels,
            batch_size=8,
            pos_neg_ratio=3.0,
            warmup_epochs=0,
            hard_ratio=0.5,
            shuffle=False,
            seed=123,
        )

        batch = next(iter(sampler))
        assert all(isinstance(item, tuple) for item in batch)

        roles = [item[1] for item in batch]
        hard_counts = {item[2] for item in batch}
        assert len(hard_counts) == 1

        pos_count = roles.count("pos")
        cand_count = roles.count("neg_candidate")
        def_count = roles.count("neg_default")

        assert pos_count == sampler.pos_per_batch
        assert cand_count == 3 * int(round(sampler.neg_per_batch * 0.5))
        expected_hard = int(round(sampler.neg_per_batch * 0.5))
        assert def_count == sampler.neg_per_batch - expected_hard

    def test_warmup_uses_standard_indices(self):
        labels = [1] * 6 + [0] * 24
        sampler = OnlineHardNegativeBatchSampler(
            labels=labels,
            batch_size=8,
            pos_neg_ratio=3.0,
            warmup_epochs=1,
            hard_ratio=0.7,
            shuffle=False,
            seed=1,
        )

        batch = next(iter(sampler))
        assert all(isinstance(item, int) for item in batch)
