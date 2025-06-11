#!/usr/bin/env python3
"""
Standalone Evaluation script for No-Adapter PPI Classifier
Evaluates a trained no-adapter checkpoint on test1 and test2 datasets
"""

# Add project root to Python path to fix imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score, 
                           roc_curve, precision_recall_curve)

# Import the standalone no-adapter model
from src.model.v2_MAE_based import PPIClassifier_NoAdapter, create_ppi_classifier_no_adapter, count_parameters

# Import utilities
from src.utils import load_data_v2, ProteinPairDatasetV2, collate_fn_v52

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_checkpoint(checkpoint_path, model):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('config', {}), checkpoint.get('val_metrics', {})

def evaluate_test_set(model, test_loader, device, test_name="Test"):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_probs = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {test_name}")
        for emb_a, emb_b, lengths_a, lengths_b, interactions in pbar:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass (v5.2 requires lengths)
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            # Track metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # Calculate AUC metrics
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
        fpr, tpr = [0, 1], [0, 1]
        precision_curve, recall_curve = [1, 0], [0, 1]
    
    results = {
        'test_name': test_name,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'auprc': auprc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'logits': all_logits,
        'labels': all_labels,
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'precision_curve': precision_curve.tolist() if hasattr(precision_curve, 'tolist') else precision_curve,
        'recall_curve': recall_curve.tolist() if hasattr(recall_curve, 'tolist') else recall_curve
    }
    
    print(f"{test_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUROC: {auc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    
    return results

def plot_evaluation_results(test_results, save_path):
    """Plot evaluation results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('V5.2 Model Evaluation Results', fontsize=16, fontweight='bold')
    
    test_names = list(test_results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # ROC curves
    for i, (test_name, results) in enumerate(test_results.items()):
        color = colors[i % len(colors)]
        axes[0].plot(results['fpr'], results['tpr'], 
                    label=f"{test_name} (AUC={results['auc']:.3f})", 
                    color=color, linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('ROC Curves')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall curves
    for i, (test_name, results) in enumerate(test_results.items()):
        color = colors[i % len(colors)]
        axes[1].plot(results['recall_curve'], results['precision_curve'], 
                    label=f"{test_name} (AUPRC={results['auprc']:.3f})", 
                    color=color, linewidth=2)
    
    axes[1].set_title('Precision-Recall Curves')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Metrics comparison
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUROC', 'AUPRC']
    x = np.arange(len(metrics))
    width = 0.8 / len(test_names)
    
    for i, (test_name, results) in enumerate(test_results.items()):
        values = [results['accuracy'], results['f1'], results['precision'], 
                 results['recall'], results['auc'], results['auprc']]
        color = colors[i % len(colors)]
        axes[2].bar(x + i * width, values, width, label=test_name, color=color, alpha=0.7)
    
    axes[2].set_title('Metrics Comparison')
    axes[2].set_ylabel('Score')
    axes[2].set_xticks(x + width * (len(test_names) - 1) / 2)
    axes[2].set_xticklabels(metrics, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved evaluation plots: {save_path}")

def evaluate_no_adapter_model(checkpoint_path, v2_mae_path, batch_size=8, log_dir=None):
    """
    Main evaluation function for no-adapter models
    
    Args:
        checkpoint_path: Path to the no-adapter checkpoint file to evaluate
        v2_mae_path: Path to the v2 MAE checkpoint (needed for model creation)
        batch_size: Batch size for evaluation
        log_dir: Directory to save evaluation results
    """
    print("🧬 NO-ADAPTER PPI CLASSIFIER EVALUATION")
    print("=" * 50)
    print(f"No-Adapter Checkpoint: {checkpoint_path}")
    print(f"V2 MAE Path: {v2_mae_path}")
    print(f"Batch size: {batch_size}")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No-Adapter checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(v2_mae_path):
        raise FileNotFoundError(f"V2 MAE checkpoint not found: {v2_mae_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    
    # Create log directory
    if log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"logs/v5_2_evaluation_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data_v2()
    
    # Create test datasets
    test1_dataset = ProteinPairDatasetV2(test1_data, protein_embeddings)
    test2_dataset = ProteinPairDatasetV2(test2_data, protein_embeddings)
    
    print(f"Test1 samples: {len(test1_dataset)}")
    print(f"Test2 samples: {len(test2_dataset)}")
    
    # Create data loaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn_v52,
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'shuffle': False
    }
    
    test1_loader = DataLoader(test1_dataset, **dataloader_kwargs)
    test2_loader = DataLoader(test2_dataset, **dataloader_kwargs)
    
    # Create model
    print(f"\n🔧 Creating no-adapter model...")
    model = create_ppi_classifier_no_adapter(v2_mae_path)
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Load checkpoint
    print(f"\n📥 Loading v5.2 checkpoint...")
    epoch, loaded_config, val_metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from epoch {epoch}")
    if val_metrics and 'auc' in val_metrics:
        print(f"Original validation AUC: {val_metrics['auc']:.4f}")
    
    # Evaluate on test sets
    print("\n🧪 Evaluating on test sets...")
    test_results = {}
    
    test_results['Test1'] = evaluate_test_set(model, test1_loader, device, "Test1")
    test_results['Test2'] = evaluate_test_set(model, test2_loader, device, "Test2")
    
    # Save results
    results_dict = {
        'checkpoint_path': checkpoint_path,
        'v2_mae_path': v2_mae_path,
        'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epoch': epoch,
        'original_config': loaded_config,
        'original_val_metrics': val_metrics,
        'test_results': test_results,
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    }
    
    # Save results JSON
    results_path = os.path.join(log_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"💾 Saved results: {results_path}")
    
    # Generate plots
    plot_path = os.path.join(log_dir, 'evaluation_plots.png')
    plot_evaluation_results(test_results, plot_path)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 V5.2 EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"Evaluated model from epoch {epoch}")
    if val_metrics and 'auc' in val_metrics:
        print(f"Original validation AUC: {val_metrics['auc']:.4f}")
    
    print("\n📊 DETAILED TEST RESULTS:")
    for test_name, test_result in test_results.items():
        print(f"\n{test_name} Dataset:")
        print(f"  Accuracy:  {test_result['accuracy']:.4f}")
        print(f"  F1 Score:  {test_result['f1']:.4f}")
        print(f"  Precision: {test_result['precision']:.4f}")
        print(f"  Recall:    {test_result['recall']:.4f}")
        print(f"  AUROC:     {test_result['auc']:.4f}")
        print(f"  AUPRC:     {test_result['auprc']:.4f}")
    
    print(f"\nResults saved to: {log_dir}")
    
    return results_dict

def main():
    """Main function"""
    # Configuration for your specific model
    checkpoint_path = "models/v5_2_training_20250610_020129/checkpoint_epoch_9_best.pth"
    v2_mae_path = "src/mask_autoencoder/model/mae_best_20250528-174157.pth"
    
    # Alternative: you can specify different checkpoint via command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"Using checkpoint from command line: {checkpoint_path}")
    
    if len(sys.argv) > 2:
        v2_mae_path = sys.argv[2]
        print(f"Using v2 MAE path from command line: {v2_mae_path}")
    
    # Run evaluation
    results = evaluate_no_adapter_model(
        checkpoint_path=checkpoint_path,
        v2_mae_path=v2_mae_path,
        batch_size=8
    )
    
    return results

if __name__ == "__main__":
    main() 