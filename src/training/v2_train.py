#!/usr/bin/env python3
"""
Standalone Training script for No-Adapter PPI Classifier
Uses pretrained v2 MAE directly without adapter layer
"""

import sys
import os
# Add project root to Python path to fix imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ⚡ PERFORMANCE OPTIMIZATIONS
print("🚀 Applying PyTorch Performance Optimizations...")

# Enable cuDNN auto-tuner for consistent input sizes
torch.backends.cudnn.benchmark = True

# Enable cuDNN deterministic mode (set to False for max speed)
torch.backends.cudnn.deterministic = False

# Enable mixed precision training support
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Memory optimization
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# JIT compilation for consistent operations
torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])

print("✅ Performance optimizations applied")

from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler  # ⚡ Mixed precision training
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score, 
                           roc_curve, precision_recall_curve)

# Import the standalone no-adapter model
from src.model.v2_MAE_based import PPIClassifier_NoAdapter, create_ppi_classifier_no_adapter, count_parameters

# Import utilities
from src.utils import load_data_v2, ProteinPairDatasetV2, collate_fn_v52

# ============================================================================
# TRAINING FUNCTIONS  
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None, scaler=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(pbar):
        try:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast('cuda'):
                    logits = model(emb_a, emb_b, lengths_a, lengths_b)
                    loss = criterion(logits, interactions)
            else:
                logits = model(emb_a, emb_b, lengths_a, lengths_b)
                loss = criterion(logits, interactions)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                continue
            
            # Mixed precision backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Unscale before gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(interactions.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear CUDA cache periodically
            if batch_idx % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error at batch {batch_idx}: {e}")
                print("Attempting to recover by clearing CUDA cache...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error at batch {batch_idx}: {e}")
            continue
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC if we have both classes
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for emb_a, emb_b, lengths_a, lengths_b, interactions in pbar:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            loss = criterion(logits, interactions)
            
            # Track metrics
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC if we have both classes
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, 
                   config, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

def train_model(config):
    """Main training function"""
    print("🧬 V5.2 PPI CLASSIFIER TRAINING")
    print("=" * 50)
    
    # Setup device and mixed precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_amp = config.get('use_mixed_precision', True) and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    
    if device.type == 'cuda':
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
        torch.cuda.empty_cache()
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"models/v5_2_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print(f"Save directory: {save_dir}")
    
    # Load data
    print("\n📊 Loading data...")
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data_v2()
    
    # Create datasets
    train_dataset = ProteinPairDatasetV2(train_data, protein_embeddings)
    val_dataset = ProteinPairDatasetV2(val_data, protein_embeddings)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_v52,
        num_workers=config.get('num_workers', 4),
        pin_memory=device.type == 'cuda',
        persistent_workers=config.get('num_workers', 4) > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_v52,
        num_workers=config.get('num_workers', 4),
        pin_memory=device.type == 'cuda',
        persistent_workers=config.get('num_workers', 4) > 0
    )
    
    # Create model
    print(f"\n🔧 Creating No-Adapter model...")
    model = create_ppi_classifier_no_adapter(config['v2_mae_path'])
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('scheduler_T0', 10),
        T_mult=config.get('scheduler_T_mult', 2),
        eta_min=config.get('scheduler_eta_min', 1e-6)
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_auc': [], 'train_auprc': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_auprc': [],
        'learning_rates': []
    }
    
    best_val_auc = 0.0
    patience_counter = 0
    
    print(f"\n🚀 Starting training for {config['num_epochs']} epochs...")
    print(f"Early stopping patience: {config['patience']} epochs")
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler, scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_auprc'].append(train_metrics['auprc'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_auprc'].append(val_metrics['auprc'])
        
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"\nTRAIN - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, AUPRC: {train_metrics['auprc']:.4f}")
        print(f"VAL   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        is_best = val_metrics['auc'] > best_val_auc
        
        save_checkpoint(
            model, optimizer, scheduler, epoch+1,
            train_metrics, val_metrics, config,
            checkpoint_path, is_best
        )
        
        if is_best:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            print(f"🎉 New best validation AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"⏰ Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Results saved to: {save_dir}")
    
    return history, save_dir

def main():
    """Main function"""
    # Configuration
    config = {
        # Model path
        'v2_mae_path': 'models/mae_best_20250528-174157.pth',  # TODO: ⚠️ UPDATE THIS PATH
        
        # Training hyperparameters
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'patience': 10,
        
        # Data loading
        'num_workers': 4,
        
        # Scheduler
        'scheduler_T0': 10,
        'scheduler_T_mult': 2,
        'scheduler_eta_min': 1e-6,
        
        # Mixed precision
        'use_mixed_precision': True,
    }
    
    # Verify v2 MAE path exists
    if not os.path.exists(config['v2_mae_path']):
        print(f"❌ Error: v2 MAE checkpoint not found: {config['v2_mae_path']}")
        print("Please update the 'v2_mae_path' in the config")
        return
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run training
    history, save_dir = train_model(config)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Results saved to: {save_dir}")
    
    return history, save_dir

if __name__ == "__main__":
    main() 