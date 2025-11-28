"""
Pipeline stage runners for DIPPI.

This module exports the three main stage execution functions:
- run_pretrain: Pretrain stage orchestration
- run_finetune: Finetune stage orchestration
- run_evaluation: Evaluation stage orchestration

These are called by the main orchestrator (run.py).
"""

from .pretrain import run_pretrain
from .finetune import run_finetune
from .evaluate import run_evaluation

__all__ = ["run_pretrain", "run_finetune", "run_evaluation"]
