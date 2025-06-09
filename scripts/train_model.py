#!/usr/bin/env python3
"""
Training script for DIPPI protein-protein interaction models.

This script provides a command-line interface for training any model
with flexible configuration options.

Usage:
    python scripts/train_model.py --model simplified --epochs 20 --batch_size 32
    python scripts/train_model.py --config configs/training_config.yaml
"""

import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architectures import create_model, get_available_models
from training.trainer import ProteinTrainer, TrainingConfig
from utils import load_data, ProteinPairDataset, collate_fn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DIPPI protein interaction models')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='simplified',
                       choices=get_available_models(),
                       help='Model architecture to train')
    parser.add_argument('--input_dim', type=int, default=960,
                       help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler_type', type=str, default='onecycle',
                       choices=['onecycle', 'cosine', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Gradient clipping norm (None to disable)')
    
    # Data configuration
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Output configuration
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save training logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for organizing outputs')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='YAML configuration file (overrides command line args)')
    
    # Device configuration
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if None)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def load_config_from_file(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_directories(args):
    """Setup experiment directories with optional experiment name."""
    if args.experiment_name:
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        log_dir = os.path.join(args.log_dir, args.experiment_name)
    else:
        checkpoint_dir = args.checkpoint_dir
        log_dir = args.log_dir
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return checkpoint_dir, log_dir


def create_data_loaders(args):
    """Create training and validation data loaders."""
    print("Loading data...")
    train_data, val_data, _, _, protein_embeddings = load_data()
    
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        config_dict = load_config_from_file(args.config)
        # Update args with config values
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training model: {args.model}")
    
    # Setup experiment directories
    checkpoint_dir, log_dir = setup_experiment_directories(args)
    
    # Create model
    model_kwargs = {
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout
    }
    
    # Add model-specific arguments
    if args.model == 'attention':
        model_kwargs['num_heads'] = getattr(args, 'num_heads', 8)
    
    model = create_model(args.model, **model_kwargs)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_norm=args.gradient_clip_norm
    )
    
    # Create trainer
    trainer = ProteinTrainer(
        model=model,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Save training configuration
    config_save_path = os.path.join(log_dir, 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Training configuration saved to: {config_save_path}")
    
    # Start training
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    try:
        results = trainer.train(train_loader, val_loader)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Best validation AUC: {results['best_val_auc']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Checkpoints saved in: {checkpoint_dir}")
        print(f"Logs saved in: {log_dir}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Latest checkpoint saved in: {checkpoint_dir}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 