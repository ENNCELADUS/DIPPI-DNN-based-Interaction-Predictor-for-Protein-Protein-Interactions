#!/usr/bin/env python3
"""
Evaluation script for DIPPI protein-protein interaction models.

This script provides a command-line interface for evaluating trained models
on test datasets with comprehensive metrics and visualizations.

Usage:
    python scripts/evaluate_model.py --checkpoint models/checkpoints/best_checkpoint.pth --dataset test1
    python scripts/evaluate_model.py --checkpoint models/checkpoints/best_checkpoint.pth --dataset all
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architectures import create_model
from evaluation.evaluator import ModelEvaluator
from utils import load_data, ProteinPairDataset, collate_fn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DIPPI protein interaction models')
    
    # Model and checkpoint configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint to evaluate')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model architecture type (auto-detected from checkpoint if None)')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['test1', 'test2', 'val', 'all'],
                       help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Output configuration
    parser.add_argument('--save_dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save evaluation plots')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save evaluation results to JSON')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for organizing outputs')
    
    # Device configuration
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if None)')
    
    # Comparison mode
    parser.add_argument('--compare_checkpoints', nargs='+', default=None,
                       help='Multiple checkpoints to compare')
    parser.add_argument('--model_names', nargs='+', default=None,
                       help='Names for models in comparison (optional)')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint with automatic architecture detection."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model info
    model_info = checkpoint.get('model_info', {})
    model_name = model_info.get('name', 'SimplifiedProteinClassifier')
    
    # Determine model type and parameters
    if 'SimplifiedProteinClassifier' in model_name:
        model_type = 'simplified'
    elif 'AttentionProteinClassifier' in model_name:
        model_type = 'attention'
    else:
        print(f"Warning: Unknown model type {model_name}, defaulting to 'simplified'")
        model_type = 'simplified'
    
    # Extract model configuration from checkpoint or use defaults
    config = checkpoint.get('config', {})
    model_kwargs = {
        'input_dim': 960,  # Default for protein embeddings
        'hidden_dim': 256,  # Default
        'dropout': 0.3      # Default
    }
    
    # Create model
    model = create_model(model_type, **model_kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Model type: {model_type}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'epoch' in checkpoint:
        print(f"   - Training epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        val_auc = checkpoint['metrics'].get('roc_auc', 'unknown')
        print(f"   - Validation AUC: {val_auc}")
    
    return model, model_type


def create_test_data_loaders(dataset_names, batch_size, num_workers):
    """Create data loaders for specified test datasets."""
    print("Loading test data...")
    
    # Load all data
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
    
    data_loaders = {}
    
    # Create requested data loaders
    if 'val' in dataset_names or 'all' in dataset_names:
        val_dataset = ProteinPairDataset(val_data, protein_embeddings)
        data_loaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        print(f"Validation samples: {len(val_dataset)}")
    
    if 'test1' in dataset_names or 'all' in dataset_names:
        test1_dataset = ProteinPairDataset(test1_data, protein_embeddings)
        data_loaders['test1'] = DataLoader(
            test1_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        print(f"Test1 samples: {len(test1_dataset)}")
    
    if 'test2' in dataset_names or 'all' in dataset_names:
        test2_dataset = ProteinPairDataset(test2_data, protein_embeddings)
        data_loaders['test2'] = DataLoader(
            test2_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        print(f"Test2 samples: {len(test2_dataset)}")
    
    return data_loaders


def setup_evaluation_directories(args):
    """Setup evaluation output directories."""
    if args.experiment_name:
        save_dir = os.path.join(args.save_dir, args.experiment_name)
    else:
        save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def evaluate_single_model(args):
    """Evaluate a single model on specified datasets."""
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup directories
    save_dir = setup_evaluation_directories(args)
    
    # Load model
    model, model_type = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, save_dir)
    
    # Determine datasets to evaluate
    if args.dataset == 'all':
        dataset_names = ['val', 'test1', 'test2']
    else:
        dataset_names = [args.dataset]
    
    # Create data loaders
    data_loaders = create_test_data_loaders(dataset_names, args.batch_size, args.num_workers)
    
    # Evaluate on each dataset
    all_results = {}
    
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    
    for dataset_name, data_loader in data_loaders.items():
        print(f"\n📊 Evaluating on {dataset_name.upper()} dataset...")
        
        # Evaluate
        results = evaluator.evaluate_dataset(data_loader, dataset_name)
        all_results[dataset_name] = results
        
        # Create visualizations
        if args.save_plots:
            figures = evaluator.create_visualizations(results, save_plots=True)
            print(f"Visualizations saved for {dataset_name}")
        
        # Save results
        if args.save_results:
            evaluator.save_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Print summary table
    print(f"{'Dataset':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
    print("-" * 70)
    
    for dataset_name, results in all_results.items():
        metrics = results['metrics']
        print(f"{dataset_name:<10} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f}")
    
    print(f"\nResults saved in: {save_dir}")
    
    return all_results


def compare_multiple_models(args):
    """Compare multiple models on the same dataset."""
    if not args.compare_checkpoints:
        raise ValueError("No checkpoints provided for comparison")
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Comparing {len(args.compare_checkpoints)} models...")
    
    # Setup directories
    save_dir = setup_evaluation_directories(args)
    
    # Determine dataset
    if args.dataset == 'all':
        dataset_name = 'test1'  # Use test1 as default for comparison
        print("Warning: Using test1 dataset for comparison (specify specific dataset for clarity)")
    else:
        dataset_name = args.dataset
    
    # Create data loader
    data_loaders = create_test_data_loaders([dataset_name], args.batch_size, args.num_workers)
    data_loader = data_loaders[dataset_name]
    
    # Evaluate each model
    all_evaluation_results = []
    model_names = args.model_names or [f"Model_{i+1}" for i in range(len(args.compare_checkpoints))]
    
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    for i, checkpoint_path in enumerate(args.compare_checkpoints):
        model_name = model_names[i] if i < len(model_names) else f"Model_{i+1}"
        print(f"\n📊 Evaluating {model_name}...")
        
        # Load model
        model, model_type = load_model_from_checkpoint(checkpoint_path, device)
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device, save_dir)
        
        # Evaluate
        results = evaluator.evaluate_dataset(data_loader, f"{dataset_name}_{model_name}")
        all_evaluation_results.append(results)
    
    # Create comparison
    dummy_evaluator = ModelEvaluator(model, device, save_dir)  # Use last model for comparison utility
    comparison_results = dummy_evaluator.compare_models(all_evaluation_results, model_names)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Print comparison table
    metrics = comparison_results['comparison_metrics']
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
    print("-" * 80)
    
    for i, model_name in enumerate(metrics['model_names']):
        print(f"{model_name:<15} "
              f"{metrics['accuracy'][i]:<10.4f} "
              f"{metrics['precision'][i]:<10.4f} "
              f"{metrics['recall'][i]:<10.4f} "
              f"{metrics['f1_score'][i]:<10.4f} "
              f"{metrics['roc_auc'][i]:<10.4f}")
    
    # Print best models
    print("\n🏆 Best performing models:")
    best_indices = comparison_results['best_model_indices']
    for metric, best_idx in best_indices.items():
        best_model = model_names[best_idx]
        best_score = metrics[metric][best_idx]
        print(f"   {metric}: {best_model} ({best_score:.4f})")
    
    print(f"\nComparison results saved in: {save_dir}")
    
    return comparison_results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        if args.compare_checkpoints:
            # Comparison mode
            results = compare_multiple_models(args)
        else:
            # Single model evaluation
            results = evaluate_single_model(args)
        
        print("\n✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 