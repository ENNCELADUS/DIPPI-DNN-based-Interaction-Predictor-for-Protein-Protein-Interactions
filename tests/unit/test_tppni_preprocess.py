"""Unit tests for topology-driven preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocess.tppni import (
    CandidatePairs,
    CleaningConfig,
    build_positive_graph,
    build_split_dataset,
    clean_pairs,
    filter_zero_l3_candidates,
    fit_configuration_model_weights,
    sample_candidate_pairs,
    select_bottom_configuration_model_nonedges,
)


def _pair_frame(rows: list[tuple[object, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["uniprotID_A", "uniprotID_B", "isInteraction"])


def test_clean_pairs_filters_bad_rows_and_canonicalizes_pairs() -> None:
    frame = _pair_frame(
        [
            (" P2 ", "P1", 1),
            ("P1", "P2", 1),
            ("P3", "P3", 1),
            ("", "P4", 0),
            (None, "P5", 0),
            ("P6", "P7", 0),
        ]
    )

    cleaned, stats = clean_pairs(frame, CleaningConfig())

    assert cleaned.to_dict(orient="records") == [
        {"uniprotID_A": "P1", "uniprotID_B": "P2", "isInteraction": 1},
        {"uniprotID_A": "P6", "uniprotID_B": "P7", "isInteraction": 0},
    ]
    assert stats.dropped_missing == 2
    assert stats.dropped_self_loops == 1
    assert stats.deduplicated_within_label == 1


def test_clean_pairs_raises_when_canonical_pair_has_conflicting_labels() -> None:
    frame = _pair_frame([("P1", "P2", 1), ("P2", "P1", 0)])

    with pytest.raises(ValueError, match="conflicting labels"):
        clean_pairs(frame, CleaningConfig())


def test_select_bottom_configuration_model_nonedges_excludes_positives_and_self_pairs() -> (
    None
):
    positives = _pair_frame(
        [
            ("A", "B", 1),
            ("B", "C", 1),
            ("B", "D", 1),
            ("C", "D", 1),
            ("D", "E", 1),
        ]
    )
    graph = build_positive_graph(positives)

    candidates = select_bottom_configuration_model_nonedges(
        graph,
        candidate_limit=3,
        block_size=2,
    )
    candidate_frame = candidates.to_frame(graph)
    candidate_pairs = {
        (row["uniprotID_A"], row["uniprotID_B"])
        for row in candidate_frame.to_dict(orient="records")
    }

    assert ("A", "B") not in candidate_pairs
    assert all(a != b for a, b in candidate_pairs)
    assert ("A", "E") in candidate_pairs


def test_fit_configuration_model_weights_matches_expected_degrees_on_toy_graph() -> (
    None
):
    positives = _pair_frame(
        [
            ("A", "B", 1),
            ("B", "C", 1),
            ("C", "D", 1),
            ("D", "E", 1),
        ]
    )
    graph = build_positive_graph(positives)

    weights = fit_configuration_model_weights(
        graph,
        block_size=2,
        max_iterations=200,
        tolerance=1e-6,
    )
    probability_matrix = np.multiply.outer(weights, weights)
    probability_matrix = probability_matrix / (1.0 + probability_matrix)
    np.fill_diagonal(probability_matrix, 0.0)

    assert np.allclose(
        probability_matrix.sum(axis=1),
        graph.degrees.astype(np.float64),
        atol=5e-3,
    )


def test_filter_zero_l3_candidates_removes_pairs_with_l3_paths() -> None:
    positives = _pair_frame([("A", "B", 1), ("B", "C", 1), ("C", "D", 1)])
    graph = build_positive_graph(positives)
    candidate_frame = pd.DataFrame(
        [
            {"uniprotID_A": "A", "uniprotID_B": "D"},
            {"uniprotID_A": "A", "uniprotID_B": "C"},
        ]
    )
    candidates = CandidatePairs.from_frame(
        graph, candidate_frame, scores=np.array([0.1, 0.2])
    )

    filtered = filter_zero_l3_candidates(graph, candidates)

    assert filtered.to_frame(graph)["uniprotID_A"].tolist() == ["A"]
    assert filtered.to_frame(graph)["uniprotID_B"].tolist() == ["C"]


def test_sample_candidate_pairs_is_deterministic_and_fails_when_pool_too_small() -> (
    None
):
    positives = _pair_frame([("A", "B", 1), ("C", "D", 1), ("E", "F", 1)])
    graph = build_positive_graph(positives)
    candidate_frame = pd.DataFrame(
        [
            {"uniprotID_A": "A", "uniprotID_B": "C"},
            {"uniprotID_A": "A", "uniprotID_B": "D"},
            {"uniprotID_A": "B", "uniprotID_B": "C"},
            {"uniprotID_A": "B", "uniprotID_B": "D"},
        ]
    )
    candidates = CandidatePairs.from_frame(
        graph,
        candidate_frame,
        scores=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    sample_a = sample_candidate_pairs(candidates, target_count=2, seed=7)
    sample_b = sample_candidate_pairs(candidates, target_count=2, seed=7)

    assert sample_a.to_frame(graph).equals(sample_b.to_frame(graph))
    with pytest.raises(ValueError, match="Not enough TPPNI candidates"):
        sample_candidate_pairs(candidates, target_count=5, seed=7)


def test_build_split_dataset_uses_all_nonedge_tppni_negatives_for_two_disjoint_edges() -> (
    None
):
    positives = _pair_frame([("A", "B", 1), ("C", "D", 1)])

    dataset = build_split_dataset(
        positive_pairs=positives,
        candidate_limit=10,
        seed=11,
        block_size=2,
    )

    assert int(dataset["isInteraction"].sum()) == 2
    assert len(dataset) - int(dataset["isInteraction"].sum()) == 4


def test_build_split_dataset_uses_full_tppni_pool_for_split() -> None:
    positives = _pair_frame([("A", "B", 1), ("C", "D", 1), ("E", "F", 1)])

    dataset = build_split_dataset(
        positive_pairs=positives,
        candidate_limit=100,
        seed=13,
        block_size=2,
    )

    assert int(dataset["isInteraction"].sum()) == 3
    assert len(dataset) - int(dataset["isInteraction"].sum()) == 12
