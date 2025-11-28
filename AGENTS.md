# AGENTS.md — DIPPI Project Guide

> Instructions for contributors (humans and agents) working in this repo: how the system is structured, how to code, test, and open PRs safely.

---

## Overview

- **Purpose**: DIPPI predicts protein–protein interactions using ESM-3 embeddings plus deep neural architectures and classical ML baselines.
- **Core pipeline**: config-driven pretraining → finetuning → evaluation, orchestrated by a central `run.py`-style entrypoint.
- **Key roles**:
  - **Orchestrator**: controls end-to-end runs, config parsing, logging, and checkpoints.
  - **Trainer**: runs a single training epoch, manages optimizers/schedulers.
  - **Evaluator**: runs validation/evaluation passes and computes metrics.
  - **Model**: architecture definition (`nn.Module` subclasses).

---

## Architecture

- **Repository layout (high level)**:
  - `src/embed/` – ESM embedding pipelines and utilities.
  - `src/model/` – model architectures (e.g., V3, TUnA) as `nn.Module` classes.
  - `src/train/` – trainer and training strategies (one-epoch training, optimizer/scheduler rebuild).
  - `src/finetune/` – finetuning routines and CLI entrypoints.
  - `src/evaluate/` – evaluation entrypoints and evaluator implementation.
  - `src/utils/` – shared utilities (config, logging, paths, seeds, devices, early stopping).
  - `configs/` – YAML config files for experiments (pretrain/finetune/eval).
  - `tests/` – unit/integration tests mirroring `src/`.
  - `logs/` – log files and CSV metrics (git-ignored).
  - `models/` – model checkpoints (git-ignored).

- **Central orchestration (`run.py`-style entrypoint)**:
  - Parses the full YAML config (run, data, model, train, finetune, evaluate).
  - Sets seeds, device(s), and distributed settings.
  - Builds model, trainer, evaluator, and data loaders.
  - Owns training/validation loops, logging, checkpointing, and early stopping.
  - Drives different run modes (`pretrain_only`, `finetune_from_pretrain`, `full_pipeline`, `eval_only`).

- **Artifacts and logging** (see `docs/logging_overview.md`):
  - `logs/{model}/pretrain/<run_id>/` and `logs/{model}/finetune/<run_id>/` – `log.log` + `training_step.csv`.
  - `logs/{model}/evaluate/<run_id>/` – evaluation CSV with metrics from `evaluate.metrics`.
  - Checkpoints under `models/{model}/{stage}/{run_id}/best_model.pth` (and optional per-epoch snapshots).

---

## Conventions

- **Language & environment**
  - Python ≥ 3.10, conda recommended.
  - Run commands from the project root with the active environment.
  - Activate the shared tooling env via `conda activate esm` before running repo commands.

- **Code style**
  - Follow PEP 8; use type hints on public APIs where practical.
  - Imports ordered: stdlib → third-party → local.
  - Use `src.utils.logging` (or project logger) instead of `print` in library code.
  - One architecture per file in `src/model/`, each as a single `nn.Module` with `__init__` and `forward`.

- **Naming**
  - Modules/packages: `snake_case` (`data_io.py`, `metrics.py`).
  - Classes: `PascalCase` (`V3`, `TUnA`, `Trainer`).
  - Functions/variables: `snake_case` (`train_one_epoch`, `batch_size`).
  - Constants: `UPPER_SNAKE_CASE`.
  - Tests: `test_*.py` files, `test_*` functions.

- **Config conventions**
  - Keep keys `lower_snake_case`.
  - Do not silently drop unknown keys—use strict config enforcement at the orchestrator level.
  - Prefer additive changes; avoid renaming/removing keys without compatibility notes.

- **Safety and data guards**
  - Validate paths, tensor shapes/dtypes, and NaNs at pipeline boundaries.
  - Never hard-code GPU device indices; use standard device helpers (`cuda`/`cpu` selection).
  - Do not commit datasets, embeddings, or checkpoints.

---

## Testing & CI

- **Local testing commands** (from repo root):
  - Lint/format: `ruff check .`
  - Tests: `python -m pytest`
  - (If configured) type-check: `mypy .`

- **Test layout**
  - Mirror `src/` in `tests/`:
    - `src/utils/config.py` → `tests/unit/utils/test_config.py`
    - `src/model/v3.py` → `tests/unit/model/test_v3.py`
    - Training/eval wiring → `tests/integration/train/`, `tests/integration/evaluate/`.
  - Use unit tests for small utilities and pure functions.
  - Use integration tests for orchestrator flows, CLIs, and config wiring.

- **Quality expectations**
  - New or changed behavior should have test coverage.
  - Keep tests hermetic (no network, no large artifacts).
  - Prefer small, fast smoke tests for long-running ML loops.

- **CI (expected pipeline)**
  - Run lint → type-check (if enabled) → `python -m pytest`.
  - CI must run from a clean checkout using documented commands only.

---

## Workflow & Approvals

- **Contributor role**
  - Treat yourself as a careful junior engineer with strong tooling.
  - Default to minimal, reversible changes; avoid repo-wide refactors unless requested.

- **Standard workflow**
  1. **Plan**: Describe your approach, list files to touch, and outline tests.
  2. **Confirm**: For risky, architectural, or behavior-changing work, get approval from a maintainer before coding.
  3. **Code**: Implement changes within the agreed scope; reuse existing patterns.
  4. **Test**: Run `ruff check .` and `python -m pytest` (or the narrowest relevant subset) and capture logs.
  5. **PR**: Open a focused PR with clear description and evidence of tests.

- **Approval policy**
  - **Design changes** (architecture, run modes, module responsibilities): require explicit maintainer review before merging.
  - **New dependencies**: must be justified in the PR description; avoid unless clearly necessary.
  - **Config/behavior changes**: document impact and any migration notes in the PR.
  - **Destructive actions** (removing logs, guards, or tests): only with stronger replacements and clear justification.

- **Security and secrets**
  - Do not log, commit, or paste secrets/PII.
  - Load credentials via environment variables or secret managers; document required keys in `README.md` or dedicated docs.

---

## Module responsibilities

- **Orchestrator (`run.py`-style entrypoints in `scripts/` or `src/train/`/`src/finetune/`)**
  - Parse full config: run, device, data, model, train, finetune, evaluate.
  - Set seeds, device, and distributed context.
  - Instantiate model, Trainer, Evaluator, data loaders, logger, and checkpoint helpers.
  - Control all loops:
    - Pretrain: call `trainer.train_one_epoch(...)`, then evaluator, logging, checkpointing, early stop.
    - Finetune: same pattern, with strategy-driven freeze/unfreeze and optimizer/scheduler rebuilds.
    - Eval-only: load checkpoint, run evaluator once, write evaluation CSV.
  - Own run modes (`pretrain_only`, `finetune_from_pretrain`, `full_pipeline`, `eval_only`) and checkpoint path selection.

- **Trainer (`src/train/base.py`, `src/train/strategies.py`)**
  - **Does**:
    - Implement `train_one_epoch(...)`: `model.train()`, forward, loss, backward, optimizer step, scheduler step, optional AMP and grad clipping.
    - Build/rebuild optimizer and scheduler on demand (e.g., when a strategy requests).
    - Expose hooks/properties for strategies (e.g., `named_parameters`, `rebuild_optimizer_and_scheduler`).
    - Return simple stats dicts (e.g., `{"loss": ..., "lr": ...}`) for logging.
  - **Does not**:
    - Parse global config, decide run mode, or manage seeds/devices.
    - Run validation, logging, checkpointing, or early stopping.

- **Evaluator (`src/evaluate/base.py` and friends)**
  - **Does**:
    - Consume a metrics config (e.g., `evaluate.metrics` or logging metrics for validation).
    - Run a single pass over a dataloader to compute loss and metrics (e.g., AUROC, AUPRC, accuracy, recall, F1, MCC).
    - Return a plain `dict[str, float|int]` (e.g., `{"val_loss": ..., "val_auroc": ...}`).
  - **Does not**:
    - Change model mode or grad context; orchestrator must call `model.eval()` and `torch.no_grad()`.
    - Write logs or choose checkpoints (orchestrator owns logging and checkpointing).

- **Model (`src/model/*.py`)**
  - One file per architecture (e.g., `v3.py`, `tuna.py`) defining a single `nn.Module` class.
  - Accepts only the subset of config relevant to that architecture (filtered by config utilities).
  - Exposes `forward` that returns loss and/or logits as needed by Trainer and Evaluator.
  - Contains any architecture-specific blocks or heads required for the task.

- **Utilities (`src/utils/`)**
  - `config.py`: parse YAML, extract model/trainer/evaluator-specific configs, enforce no unused keys.
  - `logging.py`: structured logging (console + CSV append helpers).
  - `data_io.py`: dataset loading and preprocessing.
  - `device.py` / `distributed.py`: device selection and distributed setup.
  - `early_stop.py`: early stopping logic for monitored metrics.

When adding or modifying functionality, align with these responsibilities and keep logic in the appropriate module (orchestrator vs trainer vs evaluator vs model).
