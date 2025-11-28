"""Embed pipeline entrypoints."""

from .multimodal import MultimodalEmbedder
from .sequence import SequenceEmbedder

__all__ = ["SequenceEmbedder", "MultimodalEmbedder"]
