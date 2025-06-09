#!/usr/bin/env python3
"""
Migration script for converting legacy DIPPI model checkpoints to the new unified format.

This script helps convert old DNN_v1 checkpoints to be compatible with the new
unified model architecture system.

Usage:
    python scripts/migrate_legacy_models.py --input models/old_checkpoint.pth --output models/migrated_checkpoint.pth
"""

import sys
import os
import argparse
import torch
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architectures import SimplifiedProteinClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Migrate legacy DIPPI model checkpoints')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to legacy checkpoint file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save migrated checkpoint')
    parser.add_argument('--model_type', type=str, default='simplified',
                       help='Target model type for migration')
    
    return parser.parse_args()


def migrate_checkpoint(input_path, output_path, model_type='simplified'):
    """
    Migrate a legacy checkpoint to the new unified format.
    
    Args:
        input_path: Path to legacy checkpoint
        output_path: Path to save migrated checkpoint
        model_type: Type of model architecture
    """
    print(f"Migrating checkpoint: {input_path}")
    
    # Load legacy checkpoint
    try:
        legacy_checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading legacy checkpoint: {e}")
        return False
    
    # Create new model to get the expected state dict structure
    if model_type == 'simplified':
        model = SimplifiedProteinClassifier()
    else:
        print(f"Unsupported model type: {model_type}")
        return False
    
    # Extract relevant information from legacy checkpoint
    if isinstance(legacy_checkpoint, dict):
        # If it's a dict, extract model state dict
        if 'model_state_dict' in legacy_checkpoint:
            model_state_dict = legacy_checkpoint['model_state_dict']
            epoch = legacy_checkpoint.get('epoch', 0)
            val_auc = legacy_checkpoint.get('val_auc', 0.0)
            metrics = {'roc_auc': val_auc}
        else:
            # Assume the entire dict is the state dict
            model_state_dict = legacy_checkpoint
            epoch = 0
            metrics = {}
    else:
        print("Unsupported checkpoint format")
        return False
    
    # Try to load the state dict to verify compatibility
    try:
        model.load_state_dict(model_state_dict, strict=False)
        print("✅ Model state dict is compatible")
    except Exception as e:
        print(f"❌ Model state dict compatibility issue: {e}")
        return False
    
    # Create new unified checkpoint format
    unified_checkpoint = {
        'epoch': epoch,
        'global_step': 0,  # Unknown for legacy checkpoints
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': None,  # Not available in legacy
        'config': {
            'model_type': model_type,
            'migration_timestamp': datetime.now().isoformat(),
            'original_checkpoint': input_path
        },
        'metrics': metrics,
        'model_info': model.get_model_info()
    }
    
    # Save migrated checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(unified_checkpoint, output_path)
    
    print(f"✅ Successfully migrated checkpoint to: {output_path}")
    print(f"   - Original epoch: {epoch}")
    print(f"   - Model parameters: {unified_checkpoint['model_info']['total_parameters']:,}")
    if 'roc_auc' in metrics:
        print(f"   - Validation AUC: {metrics['roc_auc']:.4f}")
    
    return True


def main():
    """Main migration function."""
    args = parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input checkpoint not found: {args.input}")
        return
    
    # Perform migration
    success = migrate_checkpoint(args.input, args.output, args.model_type)
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("\nYou can now use the migrated checkpoint with the new unified system:")
        print(f"  python scripts/evaluate_model.py --checkpoint {args.output} --dataset test1")
    else:
        print("\n❌ Migration failed!")


if __name__ == "__main__":
    main() 