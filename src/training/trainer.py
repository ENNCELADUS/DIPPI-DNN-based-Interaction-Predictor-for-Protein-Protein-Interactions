"""
Unified training utilities for all DIPPI models.

This module provides a flexible trainer class that can work with any model
architecture and includes comprehensive training, validation, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List, Optional, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self, 
                 epochs: int = 20,
                 batch_size: int = 32,
                 learning_rate: float = 5e-3,
                 weight_decay: float = 0.01,
                 scheduler_type: str = 'onecycle',
                 early_stopping_patience: int = 5,
                 save_best_only: bool = True,
                 gradient_clip_norm: Optional[float] = 1.0,
                 warmup_steps: int = 0,
                 eval_every_n_steps: Optional[int] = None,
                 log_every_n_steps: int = 50):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.save_best_only = save_best_only
        self.gradient_clip_norm = gradient_clip_norm
        self.warmup_steps = warmup_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.log_every_n_steps = log_every_n_steps
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


class ProteinTrainer:
    """
    Unified trainer for protein-protein interaction models.
    
    Supports multiple model architectures, flexible training configurations,
    and comprehensive logging and checkpointing.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device = None,
                 checkpoint_dir: str = "models/checkpoints",
                 log_dir: str = "logs"):
        
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.early_stopping_counter = 0
        self.training_history = []
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler (will be set up when we know steps_per_epoch)
        self.scheduler = None
    
    def _setup_scheduler(self, steps_per_epoch: int):
        """Setup scheduler once we know the number of steps per epoch."""
        if self.config.scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1
            )
        elif self.config.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs * steps_per_epoch
            )
        elif self.config.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        # Convert predictions to probabilities and binary predictions
        probabilities = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        binary_preds = (probabilities > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(labels, binary_preds),
            'f1_score': f1_score(labels, binary_preds, zero_division=0),
            'roc_auc': roc_auc_score(labels, probabilities),
        }
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
            # Move to device
            emb_a = emb_a.to(self.device).float()
            emb_b = emb_b.to(self.device).float()
            lengths_a = lengths_a.to(self.device)
            lengths_b = lengths_b.to(self.device)
            interactions = interactions.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            # Compute loss
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, interactions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None and self.config.scheduler_type != 'plateau':
                self.scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            all_predictions.extend(logits.detach().cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
            
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                logger.info(f"Epoch {self.current_epoch}, Step {batch_idx}: Loss = {loss.item():.4f}")
        
        # Compute epoch metrics
        epoch_metrics = self.compute_metrics(np.array(all_predictions), np.array(all_labels))
        epoch_metrics['loss'] = np.mean(epoch_losses)
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, lengths_a, lengths_b, interactions in val_loader:
                # Move to device
                emb_a = emb_a.to(self.device).float()
                emb_b = emb_b.to(self.device).float()
                lengths_a = lengths_a.to(self.device)
                lengths_b = lengths_b.to(self.device)
                interactions = interactions.to(self.device).float()
                
                # Forward pass
                logits = self.model(emb_a, emb_b, lengths_a, lengths_b)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
                # Compute loss
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, interactions)
                
                # Track metrics
                val_losses.append(loss.item())
                all_predictions.extend(logits.cpu().numpy())
                all_labels.extend(interactions.cpu().numpy())
        
        # Compute validation metrics
        val_metrics = self.compute_metrics(np.array(all_predictions), np.array(all_labels))
        val_metrics['loss'] = np.mean(val_losses)
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with AUC: {metrics['roc_auc']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        logger.info(f"Config: {self.config.to_dict()}")
        
        # Setup scheduler now that we know steps per epoch
        if self.scheduler is None:
            self._setup_scheduler(len(train_loader))
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler for plateau scheduler
            if self.scheduler is not None and self.config.scheduler_type == 'plateau':
                self.scheduler.step(val_metrics['roc_auc'])
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
            
            # Save training history
            epoch_history = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # Check if this is the best model
            is_best = val_metrics['roc_auc'] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['roc_auc']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if not self.config.save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final training history
        history_path = os.path.join(self.log_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(history_path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'history': self.training_history,
                'best_val_auc': self.best_val_metric,
                'training_time': training_time
            }, f, indent=2)
        
        return {
            'history': self.training_history,
            'best_val_auc': self.best_val_metric,
            'training_time': training_time
        } 