# DIPPI — Deep Neural Network-based Interaction Predictor for Protein–Protein Interactions

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#requirements--dependencies)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A1E2B?style=flat-square)](#testing)

DIPPI predicts protein–protein interactions (PPIs) by combining ESM-3 embeddings with purpose-built neural architectures and rigorous evaluation pipelines.

## Quick Install

Recommended to use **Conda** for environment management:

```bash
# 1. Clone the repository
git clone https://github.com/ENNCELADUS/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions.git
cd DIPPI

# 2. Create environment
conda create -n dippi python=3.10
conda activate dippi

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: This project requires ESM-3 embeddings in `data/embedding/`. See [docs/design_patterns/pipeline.md](docs/design_patterns/pipeline.md) for data layout.

## Quick Usage

DIPPI uses a config-driven pipeline. Ensure your environment is active:

```bash
conda activate dippi

# Pretrain a V3 model
python -m src.train.train_v3 --config configs/v3.yaml

# Evaluate a trained model
python -m src.evaluate.evaluate_v3_model --config configs/v3.yaml --checkpoint models/v3/pretrain/<RUN_ID>/best_model.pth
```

## Documentation

- **Contributor Guide**: [`AGENTS.md`](AGENTS.md) - How to code, test, and contribute.
- **Roadmap**: [`docs/roadmap/`](docs/roadmap/) - Development milestones.

### Design Patterns
- [Pipeline Architecture](docs/design_patterns/pipeline.md) - Orchestration of Pretrain/Finetune/Evaluate.
- [Trainer Design](docs/design_patterns/trainer.md) - Training loop and strategy pattern.
- [Model Design](docs/design_patterns/model.md) - Architecture standards.
- [Evaluator & Metrics](docs/design_patterns/evaluator.md) - Validation and evaluation logic.
- [Logging & Artifacts](docs/design_patterns/logging_overview.md) - Log files and CSV schemas.

## Testing

```bash
ruff check .
python -m pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
