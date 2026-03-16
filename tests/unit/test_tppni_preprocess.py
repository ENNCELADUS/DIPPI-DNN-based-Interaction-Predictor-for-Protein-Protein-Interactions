"""Unit tests for topology-driven preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocess.tppni import (
    CandidatePairs,
    CleaningConfig,
    ProteinSplit,
    build_global_tppni_pool,
    build_positive_graph,
    build_stage_datasets_from_pools,
    clean_pairs,
    filter_zero_l3_candidates,
    fit_configuration_model_weights,
    induce_pairs_for_protein_split,
    sample_candidate_pairs,
    select_bottom_configuration_model_nonedges,
    split_proteins_inductively,
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


def test_build_global_tppni_pool_preserves_sorted_scm_scores() -> None:
    positives = _pair_frame([("A", "B", 1), ("C", "D", 1), ("E", "F", 1)])

    graph, candidates = build_global_tppni_pool(
        positive_pairs=positives,
        candidate_limit=100,
        block_size=2,
    )

    candidate_frame = candidates.to_frame(graph)
    assert len(candidate_frame) == 12
    assert candidate_frame["score"].is_monotonic_increasing
    assert {("A", "B"), ("C", "D"), ("E", "F")}.isdisjoint(
        set(zip(candidate_frame["uniprotID_A"], candidate_frame["uniprotID_B"]))
    )


def test_split_proteins_inductively_is_disjoint_and_deterministic() -> None:
    positives = _pair_frame(
        [
            ("A", "B", 1),
            ("B", "C", 1),
            ("D", "E", 1),
            ("F", "G", 1),
        ]
    )

    split_a = split_proteins_inductively(positives, train_ratio=0.6, seed=17)
    split_b = split_proteins_inductively(positives, train_ratio=0.6, seed=17)

    assert split_a == split_b
    assert split_a.train_proteins.isdisjoint(split_a.valid_proteins)
    assert split_a.train_proteins | split_a.valid_proteins == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
    }


def test_induce_pairs_for_protein_split_keeps_only_pairs_whose_endpoints_stay_in_split() -> (
    None
):
    frame = pd.DataFrame(
        [
            {"uniprotID_A": "A", "uniprotID_B": "B", "isInteraction": 1},
            {"uniprotID_A": "A", "uniprotID_B": "C", "isInteraction": 1},
            {"uniprotID_A": "C", "uniprotID_B": "D", "isInteraction": 1},
        ]
    )

    induced = induce_pairs_for_protein_split(frame, {"A", "B", "D"})

    assert induced.to_dict(orient="records") == [
        {"uniprotID_A": "A", "uniprotID_B": "B", "isInteraction": 1}
    ]


def test_build_stage_datasets_from_pools_balances_pretrain_to_one_to_one() -> None:
    positives = _pair_frame(
        [
            ("A", "B", 1),
            ("C", "D", 1),
            ("E", "F", 1),
            ("G", "H", 1),
        ]
    )
    negative_pool = pd.DataFrame(
        [
            ("A", "C", 0, 0.01),
            ("A", "D", 0, 0.02),
            ("B", "C", 0, 0.03),
            ("B", "D", 0, 0.04),
            ("E", "G", 0, 0.05),
            ("E", "H", 0, 0.06),
            ("F", "G", 0, 0.07),
            ("F", "H", 0, 0.08),
        ],
        columns=["uniprotID_A", "uniprotID_B", "isInteraction", "score"],
    )
    split = ProteinSplit(
        train_proteins=frozenset({"A", "B", "C", "D"}),
        valid_proteins=frozenset({"E", "F", "G", "H"}),
    )

    result = build_stage_datasets_from_pools(
        stage_name="pretrain",
        positive_pairs=positives,
        negative_pool=negative_pool,
        protein_split=split,
        seed=5,
    )

    assert result.train_dataset["isInteraction"].value_counts().to_dict() == {1: 2, 0: 2}
    assert result.valid_dataset["isInteraction"].value_counts().to_dict() == {1: 2, 0: 2}
    train_negatives = result.train_dataset.loc[
        result.train_dataset["isInteraction"] == 0, ["uniprotID_A", "uniprotID_B"]
    ]
    assert set(map(tuple, train_negatives.to_numpy())) == {("A", "C"), ("A", "D")}


def test_build_stage_datasets_from_pools_normalizes_finetune_to_smaller_ratio() -> None:
    positives = _pair_frame(
        [
            ("A", "B", 1),
            ("C", "D", 1),
            ("E", "F", 1),
            ("G", "H", 1),
            ("I", "J", 1),
            ("K", "L", 1),
        ]
    )
    negative_rows: list[tuple[str, str, int, float]] = []
    score = 0.01
    for left in [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]:
        for right in [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]:
            if left >= right:
                continue
            for protein_a in left:
                for protein_b in right:
                    negative_rows.append((protein_a, protein_b, 0, score))
                    score += 0.01
    for left in [("I", "J")]:
        for right in [("K", "L")]:
            for protein_a in left:
                for protein_b in right:
                    negative_rows.append((protein_a, protein_b, 0, score))
                    score += 0.01
    negative_pool = pd.DataFrame(
        negative_rows,
        columns=["uniprotID_A", "uniprotID_B", "isInteraction", "score"],
    )
    split = ProteinSplit(
        train_proteins=frozenset({"A", "B", "C", "D", "E", "F", "G", "H"}),
        valid_proteins=frozenset({"I", "J", "K", "L"}),
    )

    result = build_stage_datasets_from_pools(
        stage_name="finetune",
        positive_pairs=positives,
        negative_pool=negative_pool,
        protein_split=split,
        seed=9,
    )

    train_counts = result.train_dataset["isInteraction"].value_counts().to_dict()
    valid_counts = result.valid_dataset["isInteraction"].value_counts().to_dict()
    assert train_counts == {0: 8, 1: 4}
    assert valid_counts == {0: 4, 1: 2}
    assert result.summary["target_neg_pos_ratio"] == pytest.approx(2.0)
    assert result.summary["train_ratio_before_normalization"] == pytest.approx(6.0)
    assert result.summary["valid_ratio_before_normalization"] == pytest.approx(2.0)
