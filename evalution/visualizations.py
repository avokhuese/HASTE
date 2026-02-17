"""
Visualizations for fraud detection results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionVisualizer:
    """Visualization tools for fraud detection results"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C5B7B', '#3E885B']
        
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: Optional[str] = None):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(history.get('train_loss', []), label='Train Loss', linewidth=2)
        axes[0, 0].plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot F1 score
        if 'train_f1' in history and 'val_f1' in history:
            axes[0, 1].plot(history['train_f1'], label='Train F1', linewidth=2)
            axes[0, 1].plot(history['val_f1'], label='Val F1', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Training and Validation F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Plot gradient norms if available
        if 'grad_norm' in history:
            axes[1, 1].plot(history['grad_norm'], label='Gradient Norm', linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norm During Training')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                            metrics_to_plot: List[str] = ['f1', 'roc_auc', 'precision', 'recall'],
                            save_path: Optional[str] = None):
        """Plot comparison of different models"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            
            model_names = []
            metric_values = []
            
            for model_name, metrics in metrics_dict.items():
                if metric in metrics:
                    model_names.append(model_name)
                    metric_values.append(metrics[metric])
            
            if metric_values:
                bars = axes[idx].bar(model_names, metric_values, color=self.colors[:len(model_names)])
                axes[idx].set_xlabel('Model')
                axes[idx].set_ylabel(metric.upper())
                axes[idx].set_title(f'{metric.upper()} Comparison')
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray],
                              save_path: Optional[str] = None):
        """Plot confusion matrices for multiple models"""
        
        n_models = len(confusion_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = np.array([axes])
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            if idx >= len(axes):
                break
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xticklabels(['Non-Fraud', 'Fraud'])
            axes[idx].set_yticklabels(['Non-Fraud', 'Fraud'])
        
        # Hide empty subplots
        for idx in range(len(confusion_matrices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrices plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                       save_path: Optional[str] = None):
        """Plot ROC curves for multiple models"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, (fpr, tpr, auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                                    save_path: Optional[str] = None):
        """Plot Precision-Recall curves for multiple models"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, (precision, recall, auc) in pr_data.items():
            ax.plot(recall, precision, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        # Add baseline (fraud rate)
        if pr_data:
            # Get fraud rate from first model's data
            first_model_data = list(pr_data.values())[0]
            fraud_rate = first_model_data[0].mean()  # Average precision
            ax.axhline(y=fraud_rate, color='r', linestyle='--', alpha=0.5, label='Baseline')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Precision-Recall curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, np.ndarray],
                               feature_names: List[str],
                               top_n: int = 20,
                               save_path: Optional[str] = None):
        """Plot feature importance for multiple models"""
        
        n_models = len(feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance) in enumerate(feature_importance.items()):
            if idx >= len(axes):
                break
            
            # Get top N features
            top_indices = np.argsort(importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            axes[idx].barh(range(len(top_features)), top_importance, color=self.colors[idx])
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name} - Top {top_n} Features')
            axes[idx].invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_attention_weights(self, attention_weights: Dict[str, np.ndarray],
                              model_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """Plot attention weights from HASTE model"""
        
        if not attention_weights:
            print("No attention weights to plot")
            return
        
        n_plots = len(attention_weights)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for idx, (attn_name, attn_matrix) in enumerate(attention_weights.items()):
            if attn_matrix.ndim == 3:  # (batch_size, seq_len, n_heads)
                # Average over batch and heads
                avg_attention = attn_matrix.mean(axis=(0, 2))
                axes[idx].plot(avg_attention, marker='o', linewidth=2)
                axes[idx].set_xlabel('Position')
                axes[idx].set_ylabel('Average Attention')
                axes[idx].set_title(f'{attn_name} - Average Attention by Position')
                
            elif attn_matrix.ndim == 2:  # (batch_size, n_models)
                # Heatmap
                if model_names is None:
                    model_names = [f'Model {i}' for i in range(attn_matrix.shape[1])]
                
                # Show first 50 samples
                sample_attention = attn_matrix[:50, :]
                im = axes[idx].imshow(sample_attention.T, aspect='auto', cmap='viridis')
                axes[idx].set_xlabel('Sample Index')
                axes[idx].set_ylabel('Base Model')
                axes[idx].set_title(f'{attn_name} - Attention Weights')
                axes[idx].set_yticks(range(len(model_names)))
                axes[idx].set_yticklabels(model_names)
                plt.colorbar(im, ax=axes[idx])
            
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention weights plot saved to {save_path}")
        
        plt.show()
    
    def plot_drift_detection(self, drift_metrics: Dict[str, List[float]],
                            window_sizes: List[int],
                            save_path: Optional[str] = None):
        """Plot concept drift detection results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot performance over time
        if 'performance' in drift_metrics:
            axes[0, 0].plot(drift_metrics['performance'], marker='o', linewidth=2)
            axes[0, 0].axhline(y=np.mean(drift_metrics['performance']), 
                              color='r', linestyle='--', alpha=0.5, label='Mean')
            axes[0, 0].set_xlabel('Time Window')
            axes[0, 0].set_ylabel('Performance (F1)')
            axes[0, 0].set_title('Performance Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot drift scores
        if 'drift_scores' in drift_metrics:
            axes[0, 1].plot(drift_metrics['drift_scores'], marker='s', linewidth=2, color='orange')
            axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[0, 1].set_xlabel('Time Window')
            axes[0, 1].set_ylabel('Drift Score')
            axes[0, 1].set_title('Concept Drift Detection')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot feature distribution changes
        if 'feature_changes' in drift_metrics:
            feature_changes = drift_metrics['feature_changes']
            n_features = len(feature_changes)
            feature_names = [f'Feature {i}' for i in range(n_features)]
            
            sorted_indices = np.argsort(feature_changes)[::-1]
            top_features = [feature_names[i] for i in sorted_indices[:10]]
            top_changes = [feature_changes[i] for i in sorted_indices[:10]]
            
            axes[1, 0].barh(range(len(top_features)), top_changes, color='steelblue')
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features)
            axes[1, 0].set_xlabel('Distribution Change')
            axes[1, 0].set_title('Top 10 Features with Distribution Changes')
            axes[1, 0].invert_yaxis()
        
        # Plot adaptation weights
        if 'adaptation_weights' in drift_metrics:
            adaptation_weights = drift_metrics['adaptation_weights']
            if adaptation_weights.ndim == 2:  # (n_windows, n_models)
                for i in range(adaptation_weights.shape[1]):
                    axes[1, 1].plot(adaptation_weights[:, i], 
                                   label=f'Model {i}', linewidth=2)
                axes[1, 1].set_xlabel('Time Window')
                axes[1, 1].set_ylabel('Adaptation Weight')
                axes[1, 1].set_title('Model Adaptation Over Time')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Drift detection plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, metrics_dict: Dict[str, Dict[str, float]],
                                   roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                                   pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                                   save_path: Optional[str] = 'dashboard.html'):
        """Create interactive dashboard with Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 
                          'F1 Score Comparison', 'Precision Comparison',
                          'Recall Comparison', 'AUC Comparison'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. ROC Curves
        for model_name, (fpr, tpr, auc) in roc_data.items():
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={auc:.3f})'),
                row=1, col=1
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      name='Random', line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        
        # 2. Precision-Recall Curves
        for model_name, (precision, recall, auc) in pr_data.items():
            fig.add_trace(
                go.Scatter(x=recall, y=precision, mode='lines', 
                          name=f'{model_name} (AUC={auc:.3f})'),
                row=1, col=2
            )
        
        # 3-6. Bar charts for metrics comparison
        metrics_to_plot = ['f1', 'precision', 'recall', 'roc_auc']
        positions = [(1, 3), (2, 1), (2, 2), (2, 3)]
        
        for metric, (row, col) in zip(metrics_to_plot, positions):
            model_names = []
            metric_values = []
            
            for model_name, metrics in metrics_dict.items():
                if metric in metrics:
                    model_names.append(model_name)
                    metric_values.append(metrics[metric])
            
            fig.add_trace(
                go.Bar(x=model_names, y=metric_values, name=metric.upper(),
                      marker_color=self.colors[:len(model_names)]),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title_text='Fraud Detection Model Comparison Dashboard',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
        fig.update_yaxes(title_text='True Positive Rate', row=1, col=1)
        fig.update_xaxes(title_text='Recall', row=1, col=2)
        fig.update_yaxes(title_text='Precision', row=1, col=2)
        
        for metric, (row, col) in zip(metrics_to_plot, positions):
            fig.update_xaxes(title_text='Model', row=row, col=col)
            fig.update_yaxes(title_text=metric.upper(), row=row, col=col)
        
        # Save dashboard
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_calibration_curves(self, calibration_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               save_path: Optional[str] = None):
        """Plot calibration curves"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, (prob_true, prob_pred) in calibration_data.items():
            ax.plot(prob_pred, prob_true, 's-', label=model_name, linewidth=2)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Calibration curves plot saved to {save_path}")
        
        plt.show()