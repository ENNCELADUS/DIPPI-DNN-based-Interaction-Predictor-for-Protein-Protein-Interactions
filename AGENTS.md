# AGENTS.md — DIPPI Project Guide

> Instructions for contributors (humans and agents) working in this repo: how the system is structured, how to code, test, and open PRs safely.

---

## 1. Quick Context

- **What**: DIPPI predicts protein–protein interactions using ESM-3 embeddings + deep neural architectures (V3, TUnA).
- **Goal**: Reproducible, config-driven pipelines for Pretraining, Finetuning, and Evaluation.
- **Role**: Act as a careful junior engineer. Follow **Plan → Confirm → Code**.

---

## 2. Environment & Tools

- **Environment**: Conda is required.
  ```bash
  conda activate esm
  ```
- **Language**: Python 3.10+
- **Core Stack**: PyTorch, Pandas, NumPy, Ruff (lint/format), Pytest.

---

## 3. Repository Structure

| Path | Purpose |
| :--- | :--- |
| `src/model/` | Architectures as `nn.Module` (e.g., `v3.py`, `tuna.py`). |
| `src/train/` | Generic `Trainer` (`base.py`) and strategies (`strategies.py`). |
| `src/evaluate/` | Evaluator logic (`base.py`). |
| `src/utils/` | Shared helpers: `config`, `logging`, `device`, `early_stop`. |
| `src/run.py` | Central orchestrator for all stages. |
| `configs/` | YAML experiment configurations. |
| `logs/` | Training logs and CSV metrics (git-ignored). |
| `models/` | Model checkpoints (git-ignored). |

---

## 4. Development Workflow

1.  **Plan**: Describe your approach. List files to touch.
2.  **Code**:
    *   **Style**: PEP 8 via Ruff. Strict type hints for public APIs.
    *   **Docs**: Google-style docstrings.
    *   **Imports**: Absolute imports only (e.g., `from src.utils import config`).
3.  **Test**:
    *   `ruff check .` (Lint)
    *   `python -m pytest` (Test)
4.  **Commit**: Concise messages. Link issues.

---

## 5. Design Patterns

Refer to `docs/design_patterns/` for detailed architecture specs:

*   [**Pipeline**](docs/design_patterns/pipeline.md): Pretrain → Finetune → Evaluate orchestration.
*   [**Trainer**](docs/design_patterns/trainer.md): Generic training loop + Strategy pattern.
*   [**Model**](docs/design_patterns/model.md): `nn.Module` standards and config injection.
*   [**Evaluator**](docs/design_patterns/evaluator.md): Stateless metric computation.

---

## 6. Rules & Guardrails

*   **Secrets**: Never commit credentials. Use env vars.
*   **Data**: Do not commit large datasets or checkpoints.
*   **Config**: Do not hardcode hyperparameters. Use YAML.
*   **Safety**: Validate paths and tensor shapes at boundaries.
