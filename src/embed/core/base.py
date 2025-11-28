"""Base classes for embed pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from .types import EmbeddingConfig, EmbeddingResult


class BaseEmbedder(ABC):
    """Common glue shared by embed pipelines."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

    @abstractmethod
    def embed_one(self, uniprot_id: str, sequence: str) -> EmbeddingResult:
        """Produce an embedding for a single UniProt identifier."""

    def embed_many(self, items: Iterable[Tuple[str, str]]) -> list[EmbeddingResult]:
        """Process ``items`` by delegating to :meth:`embed_one`."""

        return [self.embed_one(uniprot_id, sequence) for uniprot_id, sequence in items]
