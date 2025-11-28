"""
Checkpoint utilities (MVP, PyTorch-first).

Role boundary:
- Stateless helpers to save/load weights via PyTorch APIs.
- DDP-safe (only rank 0 writes).
- Save weights-only (state_dict) + optional optimizer state.
- No CSV reads, no logging, no "best so far" tracking files.

Orchestrator (run.py) owns:
- Monitor metric selection and comparison.
- Path construction and logging.
- Reading training_step.csv to get current/best metric values.

References:
- DDP rank check: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- State dict pattern: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Atomic writes: https://github.com/untitaker/python-atomicwrites
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (private)
# ─────────────────────────────────────────────────────────────────────────────


def _is_rank_zero() -> bool:
    """
    Check if current process is rank 0 (or non-distributed).
    Only rank 0 writes checkpoints in DDP training.
    """
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _get_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Extract state_dict from model, handling DDP wrapping.
    If model is wrapped in DDP, unwrap via model.module.
    """
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _atomic_write(payload: Dict[str, Any], path: str) -> None:
    """
    Write checkpoint atomically to avoid corruption on crash.
    Write to temp file in same dir, then os.replace for atomic overwrite.
    """
    path_obj = Path(path)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

    # Write to temp file
    torch.save(payload, tmp_path)

    # Atomic replace (cross-platform safe)
    os.replace(tmp_path, path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    path: str,
    include_optim: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Weights-only checkpoint writer (DDP-safe: save on rank 0).
    Creates parent dir, writes atomically to {path}/epoch_{epoch:04d}.pth.

    Args:
        model: PyTorch model to save.
        epoch: Current epoch number.
        path: Directory path where checkpoint will be saved.
        include_optim: Whether to include optimizer state.
        optimizer: Optimizer instance (required if include_optim=True).
        extra: Optional extra metadata dict to include in checkpoint.

    Returns:
        Saved file path on rank 0, else None.
    """
    if not _is_rank_zero():
        return None

    # Create parent directory
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Build filename
    filename = f"epoch_{epoch:04d}.pth"
    ckpt_path = path_obj / filename

    # Construct payload
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "state_dict": _get_state_dict(model),
    }

    if include_optim and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = dict(extra)

    # Atomic write
    _atomic_write(payload, str(ckpt_path))

    return str(ckpt_path)


def maybe_save_best(
    model: nn.Module,
    epoch: int,
    current_metric: float,
    best_so_far: float,
    mode: str,
    best_path: str,
    include_optim: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float]:
    """
    Compare current_metric vs best_so_far and, if improved, atomically
    overwrite best_path with the current model checkpoint.

    Args:
        model: PyTorch model to save.
        epoch: Current epoch number.
        current_metric: Current metric value to compare.
        best_so_far: Best metric value seen so far.
        mode: "max" (higher is better) or "min" (lower is better).
        best_path: Full path where best checkpoint will be saved.
        include_optim: Whether to include optimizer state.
        optimizer: Optimizer instance (required if include_optim=True).
        extra: Optional extra metadata dict to include in checkpoint.

    Returns:
        (improved, new_best_value) tuple:
        - improved: True if current metric improved over best_so_far.
        - new_best_value: Updated best value (current_metric if improved, else best_so_far).
    """
    if not _is_rank_zero():
        return False, best_so_far

    # Validate mode
    if mode not in {"max", "min"}:
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    # Compute improvement
    if mode == "max":
        improved = current_metric > best_so_far
    else:  # mode == "min"
        improved = current_metric < best_so_far

    if not improved:
        return False, best_so_far

    # Create parent directory
    Path(best_path).parent.mkdir(parents=True, exist_ok=True)

    # Construct payload
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "state_dict": _get_state_dict(model),
    }

    if include_optim and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = dict(extra)

    # Atomic write
    _atomic_write(payload, best_path)

    return True, current_metric


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str | Any = "cpu",
    strict: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    load_optim: bool = False,
    weights_only: bool = False,
) -> Dict[str, Any]:
    """
    Load weights (and optionally optimizer state) from ckpt_path.
    Uses torch.load(map_location=...) then model.load_state_dict(...).

    Args:
        model: PyTorch model to load weights into.
        ckpt_path: Path to checkpoint file.
        map_location: Device to map tensors to (e.g., "cpu", "cuda:0").
        strict: Whether to strictly enforce state_dict keys match.
        optimizer: Optimizer instance to load state into (if load_optim=True).
        load_optim: Whether to load optimizer state.
        weights_only: If True, only load pickle data that is directly tensors
                      (safer, but incompatible with optimizer state).

    Returns:
        Payload dict containing at least {'epoch': int}. Full blob returned
        (minus state_dicts which are already loaded into model/optimizer).

    Raises:
        FileNotFoundError: If ckpt_path does not exist.
        RuntimeError: If state_dict keys mismatch (when strict=True).
    """
    # Load checkpoint blob
    blob = torch.load(ckpt_path, map_location=map_location, weights_only=weights_only)

    # Load model state
    state_dict = blob["state_dict"]
    model_is_ddp = hasattr(model, "module")
    ckpt_has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if model_is_ddp and not ckpt_has_module_prefix:
        # Checkpoint from non-DDP; load into the wrapped module
        model.module.load_state_dict(state_dict, strict=strict)
    elif (not model_is_ddp) and ckpt_has_module_prefix:
        # Checkpoint saved from DDP; strip 'module.' prefix
        stripped = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(stripped, strict=strict)
    else:
        # Matching shapes (both DDP-aware or both plain)
        model.load_state_dict(state_dict, strict=strict)

    # Load optimizer state if requested
    if load_optim and optimizer is not None and "optimizer_state_dict" in blob:
        optimizer.load_state_dict(blob["optimizer_state_dict"])

    # Return metadata (epoch + extra)
    result = {
        "epoch": blob.get("epoch", 0),
    }
    if "extra" in blob:
        result["extra"] = blob["extra"]

    return result
