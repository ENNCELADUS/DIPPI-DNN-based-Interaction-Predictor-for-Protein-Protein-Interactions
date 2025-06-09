# DIPPI: DNN-based Interaction Predictor for Protein-Protein Interactions

A deep learning framework for predicting protein-protein interactions using various machine learning approaches including XGBoost, SVM, Logistic Regression, and neural networks with protein embeddings.

## Project Structure

This project follows modern ML best practices for organization and reproducibility:

```
DIPPI/
├── data/                    # Data storage with clear separation
│   ├── raw/                # Original, immutable datasets
│   ├── splits/             # Train/validation/test data splits  
│   └── features/           # Processed features ready for modeling
│
├── src/                    # Source code organized by functionality
│   ├── model/              # Neural network architectures and model definitions
│   ├── training/           # Training scripts and procedures
│   ├── evaluation/         # Metrics calculation and model evaluation
│   ├── results/            # Result processing and analysis
│   └── utils.py            # Shared utility functions
│
├── notebooks/              # Jupyter notebooks for exploration (numbered progression)
│
├── experiments/            # Experiment tracking and model comparisons
│
├── models/                 # Trained and serialized models, checkpoints
|
├── scripts/                # Scripts for running on clusters
├── configs/                # Configuration files (YAML, JSON) for experiments
├── tests/                  # Unit tests for code validation
├── docs/                   # Project documentation
├── logs/                   # Training logs and experiment outputs
├── requirements.txt        # Python dependencies
├── Makefile               # Common commands for training, testing, data processing
└── organization.md        # Folder organization best practices reference
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for protein embedding generation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions.git
   cd DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python scripts/train_model.py --help
   python scripts/evaluate_model.py --help
   ```


## Usage

### Data Pipeline

1. **Data Analysis**: Start with `notebooks/01_full_dataset_analyze.ipynb`
2. **Data Preprocessing**: Run `notebooks/02_seperate_medium_dataset.ipynb`
3. **Embedding Generation**: Execute `notebooks/03_embed_ESMC.ipynb`
4. **Feature Standardization**: Use `notebooks/04_embeddings_standardize.ipynb`

### Model Training & Experiments

#### 🎯 Training with the Unified System

The new unified training system supports multiple model architectures with a single, flexible interface:

##### **Option 1: Command Line Training**
```bash
# Basic training with simplified model
python scripts/train_model.py --model simplified --epochs 20 --batch_size 32

# Advanced training with custom parameters
python scripts/train_model.py \
    --model attention \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.003 \
    --hidden_dim 512 \
    --experiment_name "attention_large"

# Quick training for testing
python scripts/train_model.py --model simplified --epochs 5 --batch_size 64 --experiment_name "quick_test"
```

##### **Option 2: Configuration File Training (Recommended)**
```bash
# Train with pre-configured settings
python scripts/train_model.py --config configs/training_config_simplified.yaml
python scripts/train_model.py --config configs/training_config_attention.yaml

# Override specific parameters
python scripts/train_model.py --config configs/training_config_simplified.yaml --epochs 50 --experiment_name "extended_training"
```

##### **Available Model Types:**
- **`simplified`**: Fast, lightweight model for quick experiments
- **`attention`**: More sophisticated model with self-attention mechanisms

##### **Training Configuration Options:**

| Parameter | Description | Default | Examples |
|-----------|-------------|---------|----------|
| `--model` | Model architecture | `simplified` | `simplified`, `attention` |
| `--epochs` | Training epochs | `20` | `10`, `50`, `100` |
| `--batch_size` | Batch size | `32` | `16`, `64`, `128` |
| `--learning_rate` | Learning rate | `0.005` | `0.001`, `0.01` |
| `--hidden_dim` | Hidden layer size | `256` | `128`, `512`, `1024` |
| `--dropout` | Dropout rate | `0.3` | `0.1`, `0.5` |
| `--scheduler_type` | LR scheduler | `onecycle` | `cosine`, `plateau` |
| `--experiment_name` | Experiment ID | `None` | `"baseline_v1"` |

##### **Training Outputs:**
After training, you'll find:
- **Checkpoints**: `models/checkpoints/{experiment_name}/best_checkpoint.pth`
- **Training logs**: `logs/{experiment_name}/training_history_*.json`
- **Configuration**: `logs/{experiment_name}/training_config.yaml`

##### **Resume Training:**
```bash
# Resume from checkpoint
python scripts/train_model.py --resume models/checkpoints/latest_checkpoint.pth
```

#### 📊 Legacy Experiments (Notebooks)

For research and experimentation, you can still use the notebooks in `experiments/`:
- **Logistic Regression**: `logistic_regression_medium.ipynb`
- **SVM**: `SVM.ipynb`
- **XGBoost Variants**: `Xgboost_*.ipynb`
- **Advanced Tuning**: `Xgboost_MAE_advanced_tuning.ipynb`

### Model Evaluation

#### 📊 Comprehensive Model Assessment

The unified evaluation system provides detailed analysis with professional visualizations and metrics:

##### **Option 1: Single Model Evaluation**

```bash
# Quick evaluation on all datasets
python scripts/evaluate_model.py --checkpoint models/checkpoints/best_checkpoint.pth --dataset all

# Evaluate on specific test dataset
python scripts/evaluate_model.py --checkpoint models/checkpoints/best_checkpoint.pth --dataset test1

# Evaluate with custom output directory
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best_checkpoint.pth \
    --dataset test2 \
    --save_dir results/my_evaluation \
    --experiment_name "final_test"
```

##### **Option 2: Model Comparison**

```bash
# Compare two models
python scripts/evaluate_model.py \
    --compare_checkpoints models/simplified_model.pth models/attention_model.pth \
    --model_names "Simplified" "Attention" \
    --dataset test1

# Compare multiple models with custom names
python scripts/evaluate_model.py \
    --compare_checkpoints model1.pth model2.pth model3.pth \
    --model_names "Baseline" "Improved" "Final" \
    --dataset test2
```

##### **Available Datasets:**

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `test1` | Balanced test set | Standard evaluation |
| `test2` | Imbalanced test set | Real-world scenario |
| `val` | Validation set | Development testing |
| `all` | All datasets | Comprehensive analysis |

##### **Evaluation Outputs:**

The evaluation system automatically generates:

1. **📈 Comprehensive Metrics:**
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC, PR AUC
   - Specificity, Sensitivity
   - Confusion Matrix

2. **📊 Professional Visualizations:**
   - ROC Curves with AUC scores
   - Precision-Recall Curves
   - Confusion Matrix Heatmaps
   - Performance Comparison Charts

3. **📁 Organized Results:**
   ```
   results/evaluation/
   ├── test1_evaluation_20240101_120000.json     # Detailed metrics
   ├── test1_roc_curve_20240101_120000.png       # ROC curve plot
   ├── test1_pr_curve_20240101_120000.png        # PR curve plot
   ├── test1_confusion_matrix_20240101_120000.png # Confusion matrix
   └── test1_metrics_summary_20240101_120000.png  # Summary bar chart
   ```

##### **Example Evaluation Workflow:**

```bash
# 1. Train a model
python scripts/train_model.py --config configs/training_config_simplified.yaml --experiment_name "baseline"

# 2. Evaluate on validation set during development
python scripts/evaluate_model.py --checkpoint models/checkpoints/baseline/best_checkpoint.pth --dataset val

# 3. Final evaluation on test sets
python scripts/evaluate_model.py --checkpoint models/checkpoints/baseline/best_checkpoint.pth --dataset all

# 4. Compare with other models
python scripts/evaluate_model.py \
    --compare_checkpoints models/checkpoints/baseline/best_checkpoint.pth models/checkpoints/improved/best_checkpoint.pth \
    --model_names "Baseline" "Improved" \
    --dataset test1
```

##### **Programmatic Evaluation:**

You can also use the evaluation system in your code:

```python
from src.model import create_model
from src.evaluation import ModelEvaluator
import torch

# Load model
model = create_model('simplified')
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
evaluator = ModelEvaluator(model, device='cuda')

# Evaluate on dataset
results = evaluator.evaluate_dataset(test_loader, 'test1')

# Create visualizations
figures = evaluator.create_visualizations(results)

# Save results
evaluator.save_results(results)
```

### 🔄 Migration from Legacy Code

If you have existing model checkpoints from the old system, use the migration script:

```bash
# Basic migration
python scripts/migrate_legacy_models.py --input models/old_model.pth --output models/migrated_model.pth

# Migrate multiple checkpoints
python scripts/migrate_legacy_models.py --input models/DNN_v4.pth --output models/unified_model.pth
python scripts/migrate_legacy_models.py --input models/fixed_model_best.pth --output models/simplified_baseline.pth
```

After migration, you can use the migrated checkpoints with the new evaluation system:

```bash
# Evaluate migrated model
python scripts/evaluate_model.py --checkpoint models/unified_model.pth --dataset all
```

## 🎓 Complete Tutorial: From Training to Evaluation

Here's a step-by-step tutorial showing how to use the unified templates for a complete ML workflow:

### Step 1: Train Your First Model

```bash
# Start with a quick test run
python scripts/train_model.py \
    --model simplified \
    --epochs 5 \
    --batch_size 32 \
    --experiment_name "quick_test"

# Check training outputs
ls models/checkpoints/quick_test/  # Should contain best_checkpoint.pth
ls logs/quick_test/                # Should contain training logs
```

### Step 2: Train a Production Model

```bash
# Train with optimized configuration
python scripts/train_model.py --config configs/training_config_simplified.yaml --experiment_name "production_v1"

# Monitor training (in another terminal)
tail -f logs/production_v1/training_*.json
```

### Step 3: Evaluate Your Model

```bash
# Quick validation check
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/production_v1/best_checkpoint.pth \
    --dataset val \
    --experiment_name "production_v1_validation"

# Full test evaluation
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/production_v1/best_checkpoint.pth \
    --dataset all \
    --experiment_name "production_v1_final"
```

### Step 4: Compare Models

```bash
# Train an attention model for comparison
python scripts/train_model.py --config configs/training_config_attention.yaml --experiment_name "attention_v1"

# Compare both models
python scripts/evaluate_model.py \
    --compare_checkpoints \
        models/checkpoints/production_v1/best_checkpoint.pth \
        models/checkpoints/attention_v1/best_checkpoint.pth \
    --model_names "Simplified" "Attention" \
    --dataset test1 \
    --experiment_name "model_comparison"
```

### Step 5: Review Results

```bash
# Check evaluation outputs
ls results/evaluation/model_comparison/
# You'll find:
# - JSON files with detailed metrics
# - PNG files with visualizations
# - Comparison plots
```

### Expected Performance Metrics

| Model Type | Training Time | Test1 AUC | Test2 AUC | Parameters |
|------------|---------------|-----------|-----------|------------|
| Simplified | ~10 minutes   | 0.85-0.90 | 0.75-0.85 | ~500K     |
| Attention  | ~20 minutes   | 0.87-0.92 | 0.78-0.88 | ~800K     |

## Experiment Tracking

All experiments are systematically organized in the `experiments/` folder:

- Each experiment includes comprehensive analysis and results
- Hyperparameter tuning results are saved for reproducibility
- Model comparison metrics are tracked across all approaches

## Results

Results and model outputs are stored in:
- `src/results/` - Processed analysis results
- `models/` - Trained model checkpoints
- `logs/` - Training logs and metrics

## Development

### Adding New Experiments

1. Create a new notebook in `experiments/` with descriptive naming
2. Follow the established pattern of data loading, preprocessing, training, and evaluation
3. Save results and model checkpoints appropriately

### Code Organization

The codebase follows a clean, unified architecture:

- **Model Architectures** (`src/model/architectures.py`): Define new models by inheriting from `BaseProteinModel`
- **Training System** (`src/training/trainer.py`): Unified `ProteinTrainer` works with all model types
- **Evaluation System** (`src/evaluation/evaluator.py`): Comprehensive `ModelEvaluator` for all models
- **CLI Scripts** (`scripts/`): Easy-to-use command-line interfaces for training and evaluation
- **Configuration** (`configs/`): YAML-based configuration files for reproducible experiments

#### Adding New Models

1. Create a new model class in `src/model/architectures.py` that inherits from `BaseProteinModel`
2. Add the model to the `MODEL_REGISTRY` dictionary
3. The model automatically works with the unified training and evaluation system

## Contributing

1. Follow the established folder structure
2. Use numbered notebooks for logical progression
3. Document all experiments thoroughly
4. Add unit tests for new functionality in `tests/`

## 🔧 Troubleshooting

### Common Issues and Solutions

#### **Training Issues:**

```bash
# Memory error during training
python scripts/train_model.py --model simplified --batch_size 16  # Reduce batch size

# CUDA out of memory
python scripts/train_model.py --model simplified --device cpu      # Use CPU

# Training too slow
python scripts/train_model.py --model simplified --epochs 10 --num_workers 4  # Reduce epochs, increase workers
```

#### **Evaluation Issues:**

```bash
# Checkpoint not found
python scripts/evaluate_model.py --checkpoint models/checkpoints/*/best_checkpoint.pth --dataset test1

# Model architecture mismatch (after code changes)
python scripts/migrate_legacy_models.py --input old_checkpoint.pth --output new_checkpoint.pth
```

#### **GPU/Device Issues:**

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU usage
python scripts/train_model.py --model simplified --device cpu

# Specify GPU device
python scripts/train_model.py --model simplified --device cuda:0
```

#### **Data Loading Issues:**

```bash
# Reduce number of workers if you see multiprocessing errors
python scripts/train_model.py --model simplified --num_workers 0

# Check data paths
python -c "from src.utils import load_data; train, val, test1, test2, emb = load_data(); print('Data loaded successfully')"
```

### Performance Tips

1. **For faster training**: Use `--model simplified --batch_size 64`
2. **For better accuracy**: Use `--model attention --epochs 30`
3. **For memory efficiency**: Use `--batch_size 16 --num_workers 2`
4. **For debugging**: Use `--epochs 2 --experiment_name debug`

### Getting Help

If you encounter issues:
1. Check the logs in `logs/{experiment_name}/`
2. Verify your configuration with `--help` flags
3. Try the tutorial examples first
4. Use smaller datasets/epochs for debugging

## References

This project structure follows modern ML best practices inspired by:
- Cookiecutter Data Science standard
- Andrew Ng's systematic approach
- Fast.ai's practical methodology

## License

[Add your license information here]

## Acknowledgments

[Add acknowledgments for datasets, tools, or collaborators]