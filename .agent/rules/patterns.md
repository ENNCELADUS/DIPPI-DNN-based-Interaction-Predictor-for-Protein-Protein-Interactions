---
trigger: glob
globs: "src/**/*.py", "configs/**/*.yaml", "scripts/**/*.sh"
---

# Deep Learning Patterns

## Structure & Naming
**Layout**:
- `src/model/`: Architectures (`v3.py`, `tuna.py`) as `nn.Module` subclasses.
- `src/train/`: Generic `Trainer` (`base.py`) and strategies (`strategies.py`).
- `src/evaluate/`: Evaluator logic (`base.py`).
- `src/utils/`: Shared helpers (config, logging, device, early_stop).
- `scripts/` & `src/`: Orchestrators (`run.py`).
- `configs/`: YAML experiment configs.

## Orchestration Pattern

**Rule**: Always create shell scripts to run the pipeline. Do not run `python src/run.py` directly in documentation or repeatable workflows.

```bash
#!/bin/bash
# scripts/run_pipeline.sh

# 1. Environment Setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm

# 2. Configuration
CONFIG_PATH="configs/experiment_v1.yaml"
OUTPUT_DIR="logs/run_$(date +%Y%m%d_%H%M%S)"

# 3. Execution (run.py + config)
echo "Starting pipeline with config: $CONFIG_PATH"
python src/run.py \
    --config $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --mode train
```

## Skeleton Projects

When starting new DL components:
1. **Search**: Look for architectural precedents (e.g., HuggingFace transformers, timm).
2. **Evaluate**: Use agents to assess:
   - Dependency freshness
   - Performance benchmarks
   - Modularity/Ease of modification
3. **Adapt**: Clone references but strictly adhere to project folder structure (`src/model`, `src/train`).