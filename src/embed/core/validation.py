"""Lightweight validation helpers."""

from __future__ import annotations

from typing import Dict, Iterable


# Restrict to ASCII since upstream loaders expect canonical one-letter codes.
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYUBZXJ*-.")


def validate_protein_sequence(
    sequence: str,
    *,
    allow_extended: bool = True,
    min_length: int = 1,
    max_length: int | None = 4096,
) -> bool:
    """Return ``True`` when ``sequence`` passes structural validation."""

    if not isinstance(sequence, str) or not sequence:
        return False

    length = len(sequence)
    if length < min_length:
        return False
    if max_length is not None and length > max_length:
        return False

    allowed = EXTENDED_AMINO_ACIDS if allow_extended else STANDARD_AMINO_ACIDS
    return all(char in allowed for char in sequence.upper())


def ensure_sequence(
    sequence: str,
    *,
    allow_extended: bool = True,
    min_length: int = 1,
    max_length: int = 4096,
) -> None:
    """Raise ``ValueError`` when a provided sequence fails validation."""

    if not validate_protein_sequence(
        sequence,
        allow_extended=allow_extended,
        min_length=min_length,
        max_length=max_length,
    ):
        raise ValueError("invalid protein sequence")


def clean_protein_sequence(
    sequence: str,
    *,
    remove_gaps: bool = True,
    remove_stop: bool = True,
    to_upper: bool = True,
) -> str:
    """Return a canonicalised sequence suitable for downstream validation."""

    if not sequence:
        return ""

    cleaned = sequence.upper() if to_upper else sequence
    if remove_gaps:
        cleaned = cleaned.replace("-", "").replace(".", "")
    if remove_stop:
        cleaned = cleaned.replace("*", "")
    return cleaned


def calculate_sequence_stats(sequence: str) -> Dict[str, object]:
    """Compute lightweight statistics for ``sequence``.

    Returns a dictionary containing ``length``, ``composition`` (per amino acid
    frequency), and a coarse molecular weight estimate based on the standard
    amino acids. Unknown characters are ignored in both composition and weight.
    """

    if not sequence:
        return {"length": 0, "composition": {}, "molecular_weight": 0.0}

    aa_weights = {
        "A": 89.09,
        "C": 121.15,
        "D": 133.10,
        "E": 147.13,
        "F": 165.19,
        "G": 75.07,
        "H": 155.16,
        "I": 131.17,
        "K": 146.19,
        "L": 131.17,
        "M": 149.21,
        "N": 132.12,
        "P": 115.13,
        "Q": 146.15,
        "R": 174.20,
        "S": 105.09,
        "T": 119.12,
        "V": 117.15,
        "W": 204.23,
        "Y": 181.19,
    }

    normalized = sequence.upper()
    length = len(normalized)

    counts: Dict[str, Dict[str, float]] = {}
    for aa in STANDARD_AMINO_ACIDS:
        count = normalized.count(aa)
        if count:
            counts[aa] = {"count": count, "frequency": count / length}

    weight = sum(aa_weights.get(aa, 0.0) for aa in normalized)

    return {
        "length": length,
        "composition": counts,
        "molecular_weight": weight,
    }


def count_invalid_characters(
    sequence: str, allowed: Iterable[str] | None = None
) -> int:
    """Return the number of characters not present in ``allowed``.`"""

    if not sequence:
        return 0

    allowed_set = set(allowed) if allowed is not None else EXTENDED_AMINO_ACIDS
    return sum(1 for char in sequence.upper() if char not in allowed_set)
