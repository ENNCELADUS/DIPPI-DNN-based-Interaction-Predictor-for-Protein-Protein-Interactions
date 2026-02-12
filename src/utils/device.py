"""Device management helpers."""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve requested device with safe fallback.

    Args:
        requested_device: Device string from config.

    Returns:
        Available torch device.
    """
    normalized = requested_device.lower()
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized == "cuda":
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
    return torch.device("cpu")
