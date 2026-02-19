"""Unit tests for custom data samplers."""

from __future__ import annotations

from src.utils.data_samplers import StagedOHEMBatchSampler


def _labels(pos_count: int, neg_count: int) -> list[int]:
    return [1] * pos_count + [0] * neg_count


def test_staged_ohem_distributed_warmup_steps_match_across_ranks() -> None:
    labels = _labels(pos_count=11, neg_count=700)
    world_size = 4
    samplers = [
        StagedOHEMBatchSampler(
            labels=labels,
            batch_size=16,
            warmup_pos_neg_ratio=42.0,
            warmup_epochs=2,
            pool_multiplier=8,
            cap_protein=2,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            drop_last=True,
            seed=7,
        )
        for rank in range(world_size)
    ]

    for sampler in samplers:
        sampler.set_epoch(0)
    expected_steps = [len(sampler) for sampler in samplers]

    for sampler in samplers:
        sampler.set_epoch(0)
    observed_steps = [sum(1 for _ in sampler) for sampler in samplers]

    assert len(set(expected_steps)) == 1
    assert observed_steps == expected_steps


def test_staged_ohem_distributed_mining_steps_match_across_ranks() -> None:
    labels = _labels(pos_count=11, neg_count=700)
    world_size = 4
    samplers = [
        StagedOHEMBatchSampler(
            labels=labels,
            batch_size=16,
            warmup_pos_neg_ratio=42.0,
            warmup_epochs=0,
            pool_multiplier=10,
            cap_protein=2,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            drop_last=True,
            seed=11,
        )
        for rank in range(world_size)
    ]

    for sampler in samplers:
        sampler.set_epoch(0)
    expected_steps = [len(sampler) for sampler in samplers]

    for sampler in samplers:
        sampler.set_epoch(0)
    rank_batches = [list(sampler) for sampler in samplers]
    observed_steps = [len(batches) for batches in rank_batches]

    assert len(set(expected_steps)) == 1
    assert observed_steps == expected_steps
    if observed_steps[0] > 0:
        first_batch_sizes = [len(batches[0]) for batches in rank_batches]
        assert len(set(first_batch_sizes)) == 1
