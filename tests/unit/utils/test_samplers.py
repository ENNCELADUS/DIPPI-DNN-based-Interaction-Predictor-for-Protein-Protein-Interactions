"""
Unit tests for src/utils/samplers.py.
"""

import math

import pytest

from src.utils.samplers import ImbalancedBatchSampler, StagedOHEMBatchSampler


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


class TestStagedOHEMBatchSampler:
    """Tests for StagedOHEMBatchSampler."""

    def test_warmup_uses_standard_indices(self):
        labels = [1] * 6 + [0] * 24
        sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=8,
            warmup_pos_neg_ratio=3.0,
            warmup_epochs=1,
            pool_multiplier=4,
            cap_protein=2,
            shuffle=False,
            seed=1,
        )

        sampler.set_epoch(0)
        batch = next(iter(sampler))
        assert all(isinstance(item, int) for item in batch)
        assert len(batch) == sampler.pos_per_batch + sampler.neg_per_batch

    def test_mining_pool_roles_and_counts(self):
        labels = [1] * 10 + [0] * 30
        sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=8,
            warmup_pos_neg_ratio=3.0,
            warmup_epochs=0,
            pool_multiplier=4,
            cap_protein=2,
            shuffle=False,
            seed=123,
        )

        batch = next(iter(sampler))
        assert all(isinstance(item, tuple) for item in batch)
        assert len(batch) == sampler.mining_batch_size

        roles = {item[1] for item in batch}
        assert roles == {"ohem_pool"}

        batch_sizes = {item[2] for item in batch}
        cap_values = {item[3] for item in batch}
        assert batch_sizes == {sampler.batch_size}
        assert cap_values == {sampler.cap_protein}

        pos_count = sum(1 for idx, *_ in batch if idx < 10)
        neg_count = len(batch) - pos_count
        assert pos_count == 8
        assert neg_count == 24

    def test_warmup_len_matches_iteration_with_partial_tail(self):
        labels = [1] * 5 + [0] * 40
        sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=8,
            warmup_pos_neg_ratio=3.0,
            warmup_epochs=1,
            pool_multiplier=4,
            cap_protein=2,
            shuffle=False,
            drop_last=False,
            seed=3,
        )

        sampler.set_epoch(0)
        expected_batches = len(sampler)
        batches = list(iter(sampler))

        assert len(batches) == expected_batches
        assert expected_batches == 3
