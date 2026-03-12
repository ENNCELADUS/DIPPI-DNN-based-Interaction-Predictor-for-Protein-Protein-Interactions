"""Topology-driven preprocessing utilities for PPI datasets."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

PAIR_COLUMNS = ["uniprotID_A", "uniprotID_B", "isInteraction"]
PAIR_ID_COLUMNS = ["uniprotID_A", "uniprotID_B"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    """Cleaning controls for raw pair datasets."""

    drop_missing: bool = True
    drop_self_loops: bool = True
    canonicalize_undirected_pairs: bool = True
    deduplicate_within_label: bool = True
    conflict_policy: str = "error"


@dataclass(frozen=True)
class PairCleaningStats:
    """Summary statistics for one cleaning pass."""

    input_rows: int
    output_rows: int
    dropped_missing: int = 0
    dropped_self_loops: int = 0
    deduplicated_within_label: int = 0
    conflict_rows: int = 0


@dataclass
class PositiveGraph:
    """Positive-only undirected graph used by TPPNI generation."""

    proteins: tuple[str, ...]
    index_by_protein: dict[str, int]
    adjacency_matrix: np.ndarray
    degrees: np.ndarray
    edge_count: int


@dataclass
class CandidatePairs:
    """Compact candidate pair container."""

    source_index: np.ndarray
    target_index: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.source_index.size)

    def to_frame(self, graph: PositiveGraph) -> pd.DataFrame:
        """Materialize candidate indices as a pair dataframe."""
        records = {
            "uniprotID_A": [graph.proteins[index] for index in self.source_index],
            "uniprotID_B": [graph.proteins[index] for index in self.target_index],
        }
        if self.scores.size:
            records["score"] = self.scores.tolist()
        return pd.DataFrame(records)

    @classmethod
    def from_frame(
        cls,
        graph: PositiveGraph,
        frame: pd.DataFrame,
        scores: np.ndarray | None = None,
    ) -> CandidatePairs:
        """Create a candidate set from protein IDs."""
        required = set(PAIR_ID_COLUMNS)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Candidate frame is missing required columns: {missing}")
        source_index = frame["uniprotID_A"].map(graph.index_by_protein).to_numpy()
        target_index = frame["uniprotID_B"].map(graph.index_by_protein).to_numpy()
        if np.isnan(source_index).any() or np.isnan(target_index).any():
            raise ValueError("Candidate frame contains proteins missing from the graph")
        return cls(
            source_index=source_index.astype(np.int32, copy=False),
            target_index=target_index.astype(np.int32, copy=False),
            scores=(
                scores.astype(np.float64, copy=False)
                if scores is not None
                else np.zeros(len(frame), dtype=np.float64)
            ),
        )

    def subset(self, indices: np.ndarray) -> CandidatePairs:
        """Return a subset by integer indices."""
        return CandidatePairs(
            source_index=self.source_index[indices],
            target_index=self.target_index[indices],
            scores=self.scores[indices],
        )


def _configuration_model_probabilities(
    row_weights: np.ndarray,
    all_weights: np.ndarray,
) -> np.ndarray:
    """Return SCM link probabilities for a row block against all nodes."""
    product = np.multiply.outer(row_weights, all_weights)
    return product / (1.0 + product)


def fit_configuration_model_weights(
    graph: PositiveGraph,
    block_size: int = 256,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Fit the simple configuration model weights for one positive graph.

    The paper defines the simple unipartite configuration model by
    ``p_ij = 1 / (exp(lambda_i + lambda_j) + 1)`` with the expected degree
    constraints ``sum_j p_ij = k_i``. By setting ``x_i = exp(-lambda_i)``,
    the probability becomes ``p_ij = x_i x_j / (1 + x_i x_j)``. This routine
    fits ``x`` with multiplicative scaling until the expected degrees match the
    observed degrees to the requested tolerance.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if graph.edge_count <= 0:
        raise ValueError("Positive graph must contain at least one edge")

    target_degrees = graph.degrees.astype(np.float64, copy=False)
    weights = target_degrees / max(np.sqrt(float(graph.edge_count * 2)), 1.0)
    weights = np.clip(weights, 1e-12, 1e12)

    node_count = len(graph.proteins)
    for iteration in range(max_iterations):
        expected_degrees = np.zeros(node_count, dtype=np.float64)
        for row_start in range(0, node_count, block_size):
            row_end = min(node_count, row_start + block_size)
            row_weights = weights[row_start:row_end]
            probability_block = _configuration_model_probabilities(row_weights, weights)
            row_indices = np.arange(row_start, row_end, dtype=np.int32)
            probability_block[np.arange(row_end - row_start), row_indices] = 0.0
            expected_degrees[row_start:row_end] = probability_block.sum(axis=1)

        relative_error = np.max(
            np.abs(expected_degrees - target_degrees) / np.maximum(target_degrees, 1.0),
            initial=0.0,
        )
        if relative_error <= tolerance:
            return weights

        weights *= target_degrees / np.clip(expected_degrees, 1e-12, None)
        weights = np.clip(weights, 1e-12, 1e12)

        LOGGER.debug(
            "SCM fit iteration=%d max_relative_degree_error=%.6f",
            iteration + 1,
            relative_error,
        )

    LOGGER.warning(
        "SCM fit stopped after %d iterations with max_relative_degree_error=%.6f",
        max_iterations,
        relative_error,
    )
    return weights


def clean_pairs(
    frame: pd.DataFrame,
    cleaning_config: CleaningConfig,
) -> tuple[pd.DataFrame, PairCleaningStats]:
    """Clean raw pair rows and enforce the project pair contract."""
    missing_columns = [column for column in PAIR_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Input dataframe is missing required columns: {missing_columns}"
        )

    cleaned = frame.loc[:, PAIR_COLUMNS].copy()
    cleaned["uniprotID_A"] = cleaned["uniprotID_A"].astype("string").str.strip()
    cleaned["uniprotID_B"] = cleaned["uniprotID_B"].astype("string").str.strip()
    labels = pd.to_numeric(cleaned["isInteraction"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("isInteraction must contain only binary labels")
    cleaned["isInteraction"] = labels.astype(np.int8)

    dropped_missing = 0
    missing_mask = (
        cleaned["uniprotID_A"].isna()
        | cleaned["uniprotID_B"].isna()
        | (cleaned["uniprotID_A"] == "")
        | (cleaned["uniprotID_B"] == "")
    )
    if bool(missing_mask.any()):
        dropped_missing = int(missing_mask.sum())
        if cleaning_config.drop_missing:
            cleaned = cleaned.loc[~missing_mask].copy()
        else:
            raise ValueError("Input dataframe contains missing or blank protein IDs")

    dropped_self_loops = 0
    self_loop_mask = cleaned["uniprotID_A"] == cleaned["uniprotID_B"]
    if bool(self_loop_mask.any()):
        dropped_self_loops = int(self_loop_mask.sum())
        if cleaning_config.drop_self_loops:
            cleaned = cleaned.loc[~self_loop_mask].copy()
        else:
            raise ValueError("Input dataframe contains self-loop pairs")

    if cleaning_config.canonicalize_undirected_pairs:
        pair_a = cleaned["uniprotID_A"].to_numpy(dtype=str, copy=False)
        pair_b = cleaned["uniprotID_B"].to_numpy(dtype=str, copy=False)
        ordered_a = np.where(pair_a <= pair_b, pair_a, pair_b)
        ordered_b = np.where(pair_a <= pair_b, pair_b, pair_a)
        cleaned["uniprotID_A"] = ordered_a
        cleaned["uniprotID_B"] = ordered_b

    deduplicated_within_label = 0
    if cleaning_config.deduplicate_within_label:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=PAIR_COLUMNS, keep="first"
        ).reset_index(drop=True)
        deduplicated_within_label = before - len(cleaned)

    conflict_mask = cleaned.duplicated(subset=PAIR_ID_COLUMNS, keep=False)
    conflict_rows = int(conflict_mask.sum())
    if conflict_rows:
        conflict_policy = cleaning_config.conflict_policy.lower()
        if conflict_policy == "error":
            conflict_pairs = (
                cleaned.loc[conflict_mask, PAIR_ID_COLUMNS]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Found canonical protein pairs with conflicting labels: "
                f"{conflict_pairs}"
            )
        if conflict_policy == "positive_wins":
            cleaned = (
                cleaned.sort_values("isInteraction", ascending=False)
                .drop_duplicates(subset=PAIR_ID_COLUMNS, keep="first")
                .reset_index(drop=True)
            )
        elif conflict_policy == "drop":
            cleaned = cleaned.loc[~conflict_mask].reset_index(drop=True)
        else:
            raise ValueError(
                "cleaning.conflict_policy must be one of: error, positive_wins, drop"
            )

    cleaned["isInteraction"] = cleaned["isInteraction"].astype(np.int8)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned, PairCleaningStats(
        input_rows=len(frame),
        output_rows=len(cleaned),
        dropped_missing=dropped_missing,
        dropped_self_loops=dropped_self_loops,
        deduplicated_within_label=deduplicated_within_label,
        conflict_rows=conflict_rows,
    )


def split_positive_pairs(
    positive_pairs: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positive-only pairs into train and validation partitions."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if positive_pairs.empty:
        raise ValueError("positive_pairs must be non-empty")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(positive_pairs))
    train_cut = int(len(positive_pairs) * train_ratio)
    if train_cut <= 0 or train_cut >= len(positive_pairs):
        raise ValueError("train_ratio produced an empty train or validation split")

    shuffled = positive_pairs.iloc[permutation].reset_index(drop=True)
    train_frame = shuffled.iloc[:train_cut].reset_index(drop=True)
    valid_frame = shuffled.iloc[train_cut:].reset_index(drop=True)
    return train_frame, valid_frame


def build_positive_graph(positive_pairs: pd.DataFrame) -> PositiveGraph:
    """Build an undirected graph from positive-only pairs."""
    if positive_pairs.empty:
        raise ValueError("positive_pairs must be non-empty")

    unique_pairs = (
        positive_pairs.loc[:, PAIR_ID_COLUMNS].drop_duplicates().reset_index(drop=True)
    )
    proteins = tuple(
        sorted(
            set(unique_pairs["uniprotID_A"].tolist())
            | set(unique_pairs["uniprotID_B"].tolist())
        )
    )
    index_by_protein = {protein: index for index, protein in enumerate(proteins)}
    node_count = len(proteins)
    adjacency_matrix = np.zeros((node_count, node_count), dtype=bool)

    for protein_a, protein_b in unique_pairs.itertuples(index=False):
        source_index = index_by_protein[str(protein_a)]
        target_index = index_by_protein[str(protein_b)]
        adjacency_matrix[source_index, target_index] = True
        adjacency_matrix[target_index, source_index] = True

    np.fill_diagonal(adjacency_matrix, False)
    degrees = adjacency_matrix.sum(axis=1).astype(np.int32)
    edge_count = int(adjacency_matrix.sum() // 2)
    return PositiveGraph(
        proteins=proteins,
        index_by_protein=index_by_protein,
        adjacency_matrix=adjacency_matrix,
        degrees=degrees,
        edge_count=edge_count,
    )


def select_bottom_configuration_model_nonedges(
    graph: PositiveGraph,
    candidate_limit: int,
    block_size: int = 256,
) -> CandidatePairs:
    """Select bottom-N least probable non-edges via the fitted SCM."""
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    node_count = len(graph.proteins)
    max_nonedges = (node_count * (node_count - 1)) // 2 - graph.edge_count
    if max_nonedges <= 0:
        raise ValueError("Positive graph has no candidate non-edges")
    limit = min(candidate_limit, max_nonedges)

    weights = fit_configuration_model_weights(graph, block_size=block_size)
    column_indices = np.arange(node_count, dtype=np.int32)

    kept_sources = np.empty(0, dtype=np.int32)
    kept_targets = np.empty(0, dtype=np.int32)
    kept_scores = np.empty(0, dtype=np.float64)

    for row_start in range(0, node_count, block_size):
        row_end = min(node_count, row_start + block_size)
        row_indices = np.arange(row_start, row_end, dtype=np.int32)
        score_block = _configuration_model_probabilities(
            weights[row_start:row_end],
            weights,
        )
        candidate_mask = ~graph.adjacency_matrix[row_start:row_end].copy()
        candidate_mask &= column_indices[None, :] > row_indices[:, None]

        if kept_scores.size >= limit:
            candidate_mask &= score_block <= float(kept_scores.max())
        if not bool(candidate_mask.any()):
            continue

        block_row_offsets, block_targets = np.nonzero(candidate_mask)
        block_sources = row_indices[block_row_offsets]
        block_scores = score_block[block_row_offsets, block_targets]

        kept_sources = np.concatenate([kept_sources, block_sources])
        kept_targets = np.concatenate([kept_targets, block_targets])
        kept_scores = np.concatenate([kept_scores, block_scores])

        if kept_scores.size > limit:
            keep_indices = np.argpartition(kept_scores, limit - 1)[:limit]
            kept_sources = kept_sources[keep_indices]
            kept_targets = kept_targets[keep_indices]
            kept_scores = kept_scores[keep_indices]

    if kept_scores.size == 0:
        raise ValueError("Configuration-model screening produced no candidate pairs")

    order = np.lexsort((kept_targets, kept_sources, kept_scores))
    return CandidatePairs(
        source_index=kept_sources[order],
        target_index=kept_targets[order],
        scores=kept_scores[order],
    )


def filter_zero_l3_candidates(
    graph: PositiveGraph,
    candidates: CandidatePairs,
    target_block_size: int = 2048,
) -> CandidatePairs:
    """Keep only candidate pairs with zero induced L3 paths."""
    if target_block_size <= 0:
        raise ValueError("target_block_size must be positive")
    if len(candidates) == 0:
        return candidates

    source_order = np.lexsort((candidates.target_index, candidates.source_index))
    ordered_sources = candidates.source_index[source_order]
    ordered_targets = candidates.target_index[source_order]
    keep_mask = np.zeros(len(candidates), dtype=bool)

    unique_sources, source_starts = np.unique(ordered_sources, return_index=True)
    source_stops = np.append(source_starts[1:], len(ordered_sources))

    for source_index, start, stop in zip(
        unique_sources,
        source_starts,
        source_stops,
        strict=False,
    ):
        neighbor_indices = np.flatnonzero(graph.adjacency_matrix[int(source_index)])
        if neighbor_indices.size == 0:
            keep_mask[start:stop] = True
            continue

        second_hop_mask = graph.adjacency_matrix[neighbor_indices].any(axis=0)
        second_hop_mask[int(source_index)] = False

        for block_start in range(start, stop, target_block_size):
            block_stop = min(stop, block_start + target_block_size)
            target_indices = ordered_targets[block_start:block_stop]
            l3_exists = np.any(
                graph.adjacency_matrix[target_indices] & second_hop_mask[None, :],
                axis=1,
            )
            keep_mask[block_start:block_stop] = ~l3_exists

    kept_order = np.sort(source_order[keep_mask])
    return candidates.subset(kept_order.astype(np.int32, copy=False))


def sample_candidate_pairs(
    candidates: CandidatePairs,
    target_count: int,
    seed: int,
) -> CandidatePairs:
    """Sample a deterministic subset of candidate pairs without replacement."""
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if len(candidates) < target_count:
        raise ValueError(
            "Not enough TPPNI candidates after CL3 filtering to satisfy target_count"
        )
    if target_count == 0:
        return CandidatePairs(
            source_index=np.empty(0, dtype=np.int32),
            target_index=np.empty(0, dtype=np.int32),
            scores=np.empty(0, dtype=np.float64),
        )
    if len(candidates) == target_count:
        return candidates

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(candidates), size=target_count, replace=False))
    return candidates.subset(chosen.astype(np.int32, copy=False))


def build_split_dataset(
    positive_pairs: pd.DataFrame,
    candidate_limit: int,
    seed: int,
    block_size: int = 256,
) -> pd.DataFrame:
    """Build one train or validation dataset using TPPNI negatives."""
    if "isInteraction" not in positive_pairs.columns:
        raise ValueError("positive_pairs must include isInteraction")

    graph = build_positive_graph(positive_pairs)
    screened = select_bottom_configuration_model_nonedges(
        graph,
        candidate_limit=candidate_limit,
        block_size=block_size,
    )
    sampled_negatives = filter_zero_l3_candidates(graph, screened)
    if len(sampled_negatives) == 0:
        raise ValueError("TPPNI generation produced no CL3 negatives for this split")

    negative_frame = sampled_negatives.to_frame(graph).loc[:, PAIR_ID_COLUMNS]
    negative_frame["isInteraction"] = np.int8(0)
    positive_frame = positive_pairs.loc[:, PAIR_COLUMNS].copy()
    positive_frame["isInteraction"] = np.int8(1)

    output = pd.concat([positive_frame, negative_frame], ignore_index=True)
    output = output.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output["isInteraction"] = output["isInteraction"].astype(np.int8)
    return output
