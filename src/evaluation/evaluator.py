"""
Unified evaluation utilities for all DIPPI models.

This module provides comprehensive model evaluation including metrics computation,
visualization, and model comparison utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Unified evaluator for protein-protein interaction models.
    
    Provides comprehensive evaluation metrics, visualization, and comparison utilities.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device = None,
                 save_dir: str = "results/evaluation"):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Evaluator initialized with device: {self.device}")
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(data_loader):
                # Move to device
                emb_a = emb_a.to(self.device).float()
                emb_b = emb_b.to(self.device).float()
                lengths_a = lengths_a.to(self.device)
                lengths_b = lengths_b.to(self.device)
                
                # Forward pass
                logits = self.model(emb_a, emb_b, lengths_a, lengths_b)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
                # Store results
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(interactions.numpy())
                
                if batch_idx % 50 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(data_loader)}")
        
        # Convert to numpy arrays
        logits = np.array(all_logits)
        labels = np.array(all_labels)
        probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities, labels
    
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       probabilities: np.ndarray, 
                       labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            labels: True labels
            
        Returns:
            Dictionary of computed metrics
        """
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # ROC metrics
        try:
            roc_auc = roc_auc_score(labels, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
        except ValueError:
            roc_auc = 0.0
            fpr, tpr, roc_thresholds = [], [], []
        
        # Precision-Recall metrics
        try:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probabilities)
            pr_auc = np.trapz(precision_curve, recall_curve)
        except ValueError:
            precision_curve, recall_curve, pr_thresholds = [], [], []
            pr_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            specificity = sensitivity = 0
            tn = fp = fn = tp = 0
        
        # Class distribution
        class_counts = np.bincount(labels.astype(int))
        class_distribution = {
            'negative_samples': int(class_counts[0]) if len(class_counts) > 0 else 0,
            'positive_samples': int(class_counts[1]) if len(class_counts) > 1 else 0
        }
        
        # Detailed classification report
        class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_distribution,
            'classification_report': class_report,
            'roc_curve': {
                'fpr': fpr.tolist() if len(fpr) > 0 else [],
                'tpr': tpr.tolist() if len(tpr) > 0 else [],
                'thresholds': roc_thresholds.tolist() if len(roc_thresholds) > 0 else []
            },
            'pr_curve': {
                'precision': precision_curve.tolist() if len(precision_curve) > 0 else [],
                'recall': recall_curve.tolist() if len(recall_curve) > 0 else [],
                'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            }
        }
    
    def evaluate_dataset(self, 
                        data_loader: DataLoader, 
                        dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate model on a dataset with comprehensive metrics.
        
        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of the dataset for logging
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} dataset...")
        logger.info(f"Dataset size: {len(data_loader.dataset)} samples")
        
        # Generate predictions
        predictions, probabilities, labels = self.predict(data_loader)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, probabilities, labels)
        
        # Add dataset info
        evaluation_results = {
            'dataset_name': dataset_name,
            'dataset_size': len(labels),
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions_summary': {
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities)),
                'mean_probability': float(np.mean(probabilities)),
                'std_probability': float(np.std(probabilities))
            }
        }
        
        # Log key metrics
        logger.info(f"Results for {dataset_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
        
        return evaluation_results
    
    def create_visualizations(self, 
                            evaluation_results: Dict[str, Any],
                            save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations for evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_dataset
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        dataset_name = evaluation_results['dataset_name']
        metrics = evaluation_results['metrics']
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curve
        if len(metrics['roc_curve']['fpr']) > 0:
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'], 
                       linewidth=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve - {dataset_name}')
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            figures['roc_curve'] = fig_roc
        
        # 2. Precision-Recall Curve
        if len(metrics['pr_curve']['precision']) > 0:
            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
            ax_pr.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'],
                      linewidth=2, label=f'PR Curve (AUC = {metrics["pr_auc"]:.3f})')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'Precision-Recall Curve - {dataset_name}')
            ax_pr.legend()
            ax_pr.grid(True, alpha=0.3)
            figures['pr_curve'] = fig_pr
        
        # 3. Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax_cm.set_title(f'Confusion Matrix - {dataset_name}')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        figures['confusion_matrix'] = fig_cm
        
        # 4. Metrics Summary Bar Plot
        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                        metrics['f1_score'], metrics['roc_auc'], metrics['pr_auc']]
        
        bars = ax_metrics.bar(metric_names, metric_values, alpha=0.7)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title(f'Model Performance Metrics - {dataset_name}')
        ax_metrics.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures['metrics_summary'] = fig_metrics
        
        # Save plots if requested
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for plot_name, fig in figures.items():
                filename = f"{dataset_name}_{plot_name}_{timestamp}.png"
                filepath = os.path.join(self.save_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot: {filepath}")
        
        return figures
    
    def save_results(self, evaluation_results: Dict[str, Any], filename: str = None):
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluation_results: Results from evaluate_dataset
            filename: Optional filename, will be auto-generated if not provided
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = evaluation_results['dataset_name']
            filename = f"{dataset_name}_evaluation_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {filepath}")
    
    def compare_models(self, 
                      evaluation_results_list: List[Dict[str, Any]],
                      model_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            evaluation_results_list: List of evaluation results from different models
            model_names: Optional list of model names for labeling
            
        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(evaluation_results_list))]
        
        # Extract key metrics for comparison
        comparison_metrics = {
            'model_names': model_names,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
        for results in evaluation_results_list:
            metrics = results['metrics']
            comparison_metrics['accuracy'].append(metrics['accuracy'])
            comparison_metrics['precision'].append(metrics['precision'])
            comparison_metrics['recall'].append(metrics['recall'])
            comparison_metrics['f1_score'].append(metrics['f1_score'])
            comparison_metrics['roc_auc'].append(metrics['roc_auc'])
            comparison_metrics['pr_auc'].append(metrics['pr_auc'])
        
        # Create comparison visualization
        fig_comparison, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.12
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2) * width
            ax.bar(x + offset, comparison_metrics[metric], width, 
                  label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save comparison plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.save_dir, f"model_comparison_{timestamp}.png")
        fig_comparison.savefig(comparison_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to: {comparison_path}")
        
        return {
            'comparison_metrics': comparison_metrics,
            'comparison_plot': fig_comparison,
            'best_model_indices': {
                metric: int(np.argmax(values)) 
                for metric, values in comparison_metrics.items() 
                if metric != 'model_names'
            }
        } 