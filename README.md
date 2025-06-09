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


## Usage

### Data Pipeline

1. **Data Analysis**: Start with `notebooks/01_full_dataset_analyze.ipynb`
2. **Data Preprocessing**: Run `notebooks/02_seperate_medium_dataset.ipynb`
3. **Embedding Generation**: Execute `notebooks/03_embed_ESMC.ipynb`
4. **Feature Standardization**: Use `notebooks/04_embeddings_standardize.ipynb`

### Model Training & Experiments


### Model Evaluation

Use scripts in `src/evaluation/` for comprehensive model assessment:


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

- **Data Processing**: Add new data utilities to `src/data/`
- **Model Architectures**: Define new models in `src/model/`
- **Training Logic**: Implement training procedures in `src/training/`
- **Evaluation Metrics**: Add evaluation functions to `src/evaluation/`

## Contributing

1. Follow the established folder structure
2. Use numbered notebooks for logical progression
3. Document all experiments thoroughly
4. Add unit tests for new functionality in `tests/`

## References

This project structure follows modern ML best practices inspired by:
- Cookiecutter Data Science standard
- Andrew Ng's systematic approach
- Fast.ai's practical methodology

## License

[Add your license information here]

## Acknowledgments

[Add acknowledgments for datasets, tools, or collaborators]