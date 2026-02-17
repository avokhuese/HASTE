"""
Comprehensive evaluator for fraud detection pipeline
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    cohen_kappa_score, fbeta_score, log_loss, brier_score_loss,
    classification_report
)
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    """Comprehensive evaluation and visualization for fraud detection models"""
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        # Color scheme
        self.colors = {
            'rf': '#2E86AB',      # Blue
            'xgb': '#A23B72',     # Purple
            'svm': '#F18F01',     # Orange
            'lr': '#C73E1D',      # Red
            'haste': '#3E885B',   # Green
            'ensemble': '#6C5B7B' # Dark Purple
        }
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for a model"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        # Advanced metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        # Derived metrics
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Probability-based metrics
        if y_prob is not None:
            metrics['log_loss'] = log_loss(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            
            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = 0.0
            
            # Precision-Recall AUC
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except:
                metrics['pr_auc'] = 0.0
            
            # Calculate optimal threshold using Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            metrics['optimal_threshold'] = float(thresholds[optimal_idx])
            metrics['optimal_tpr'] = float(tpr[optimal_idx])
            metrics['optimal_fpr'] = float(fpr[optimal_idx])
        
        # Business metrics
        # Assuming: Fraud costs 10x more than false alarm
        fraud_cost = 10
        false_alarm_cost = 1
        
        total_cost = (fn * fraud_cost) + (fp * false_alarm_cost)
        max_possible_cost = len(y_true) * fraud_cost  # All frauds missed
        metrics['cost_savings'] = 1 - (total_cost / max_possible_cost) if max_possible_cost > 0 else 0
        metrics['total_cost'] = float(total_cost)
        
        return metrics
    
    def evaluate_all_models(self, models_dict: Dict[str, Any], 
                          X_test: np.ndarray, y_test: np.ndarray,
                          dataset_name: str = 'test') -> Dict[str, Dict[str, float]]:
        """Evaluate all models and return comprehensive metrics"""
        
        print(f"\nEvaluating all models on {dataset_name} set...")
        all_metrics = {}
        
        for model_name, model_info in models_dict.items():
            print(f"  Evaluating {model_name}...")
            
            try:
                # Get predictions
                if 'model' in model_info:
                    model = model_info['model']
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = model.predict(X_test)
                elif 'predictions' in model_info:
                    y_prob = model_info['predictions']
                else:
                    continue
                
                # Convert to binary predictions
                y_pred = (y_prob > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self.calculate_all_metrics(y_test, y_pred, y_prob)
                all_metrics[model_name] = metrics
                
                print(f"    F1: {metrics['f1']:.4f}, AUC: {metrics.get('roc_auc', 0):.4f}")
                
            except Exception as e:
                print(f"    Error evaluating {model_name}: {e}")
        
        return all_metrics
    
    def create_all_plots(self, all_metrics: Dict[str, Dict[str, float]],
                        y_test: np.ndarray, 
                        predictions_dict: Dict[str, np.ndarray],
                        feature_importance: Optional[Dict[str, np.ndarray]] = None,
                        attention_weights: Optional[Dict[str, np.ndarray]] = None,
                        dataset_name: str = 'test'):
        """Create all visualization plots"""
        
        print(f"\nCreating comprehensive visualizations...")
        
        # 1. Performance Comparison Bar Chart
        self._plot_performance_comparison(all_metrics, dataset_name)
        
        # 2. ROC Curves
        self._plot_roc_curves(all_metrics, predictions_dict, y_test, dataset_name)
        
        # 3. Precision-Recall Curves
        self._plot_pr_curves(all_metrics, predictions_dict, y_test, dataset_name)
        
        # 4. Confusion Matrices Heatmap
        self._plot_confusion_matrices(all_metrics, dataset_name)
        
        # 5. Cost-Benefit Analysis
        self._plot_cost_benefit_analysis(all_metrics, dataset_name)
        
        # 6. Metric Radar Chart
        self._plot_radar_chart(all_metrics, dataset_name)
        
        # 7. Feature Importance (if available)
        if feature_importance:
            self._plot_feature_importance(feature_importance, dataset_name)
        
        # 8. Attention Weights (if available)
        if attention_weights:
            self._plot_attention_weights(attention_weights, dataset_name)
        
        # 9. Model Correlation Heatmap
        self._plot_model_correlation(predictions_dict, dataset_name)
        
        # 10. Threshold Analysis
        self._plot_threshold_analysis(predictions_dict, y_test, dataset_name)
        
        print(f"  All plots saved to 'plots/' directory")
    
    def _plot_performance_comparison(self, all_metrics: Dict[str, Dict[str, float]], 
                                   dataset_name: str):
        """Plot performance comparison across models"""
        
        metrics_to_plot = ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        models = list(all_metrics.keys())
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            
            metric_values = [all_metrics[m].get(metric, 0) for m in models]
            
            # Get colors for each model
            bar_colors = [self.colors.get(m.lower(), '#808080') for m in models]
            
            bars = axes[idx].bar(models, metric_values, color=bar_colors)
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric.upper(), fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Set y-axis limit
            axes[idx].set_ylim(0, 1.1)
        
        # 6th subplot: Overall ranking
        axes[5].axis('off')
        
        # Calculate overall score (weighted average of key metrics)
        overall_scores = {}
        for model in models:
            scores = []
            weights = []
            
            if 'f1' in all_metrics[model]:
                scores.append(all_metrics[model]['f1'])
                weights.append(0.3)  # F1 is most important
            
            if 'roc_auc' in all_metrics[model]:
                scores.append(all_metrics[model]['roc_auc'])
                weights.append(0.25)
            
            if 'precision' in all_metrics[model]:
                scores.append(all_metrics[model]['precision'])
                weights.append(0.2)
            
            if 'recall' in all_metrics[model]:
                scores.append(all_metrics[model]['recall'])
                weights.append(0.15)
            
            if 'accuracy' in all_metrics[model]:
                scores.append(all_metrics[model]['accuracy'])
                weights.append(0.1)
            
            if scores:
                overall_scores[model] = np.average(scores, weights=weights)
        
        # Sort by overall score
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create ranking text
        ranking_text = "Overall Ranking:\n\n"
        for rank, (model, score) in enumerate(sorted_scores, 1):
            ranking_text += f"{rank}. {model}: {score:.3f}\n"
        
        axes[5].text(0.1, 0.5, ranking_text, fontsize=12, verticalalignment='center')
        axes[5].set_title('Model Ranking', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Model Performance Comparison ({dataset_name})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'plots/performance_comparison_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, all_metrics: Dict[str, Dict[str, float]],
                        predictions_dict: Dict[str, np.ndarray],
                        y_test: np.ndarray, dataset_name: str):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for model_name, y_prob in predictions_dict.items():
            if model_name in all_metrics and 'roc_auc' in all_metrics[model_name]:
                auc = all_metrics[model_name]['roc_auc']
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                
                color = self.colors.get(model_name.lower(), '#808080')
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', 
                        linewidth=2, color=color)
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves ({dataset_name})', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add AUC comparison
        auc_values = [(m, all_metrics[m]['roc_auc']) for m in all_metrics if 'roc_auc' in all_metrics[m]]
        if auc_values:
            best_model, best_auc = max(auc_values, key=lambda x: x[1])
            plt.figtext(0.15, 0.02, f'Best AUC: {best_model} ({best_auc:.3f})', 
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'plots/roc_curves_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, all_metrics: Dict[str, Dict[str, float]],
                       predictions_dict: Dict[str, np.ndarray],
                       y_test: np.ndarray, dataset_name: str):
        """Plot Precision-Recall curves for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for model_name, y_prob in predictions_dict.items():
            if model_name in all_metrics and 'pr_auc' in all_metrics[model_name]:
                auc = all_metrics[model_name]['pr_auc']
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                
                color = self.colors.get(model_name.lower(), '#808080')
                plt.plot(recall, precision, label=f'{model_name} (AUC = {auc:.3f})', 
                        linewidth=2, color=color)
        
        # Add baseline (fraud rate)
        fraud_rate = y_test.mean()
        plt.axhline(y=fraud_rate, color='r', linestyle='--', alpha=0.5, 
                   label=f'Baseline (Fraud Rate = {fraud_rate:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves ({dataset_name})', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'plots/pr_curves_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, all_metrics: Dict[str, Dict[str, float]], 
                                dataset_name: str):
        """Plot confusion matrices for all models"""
        
        models = list(all_metrics.keys())
        n_models = len(models)
        
        if n_models == 0:
            return
        
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = np.array([axes])
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models):
            if idx >= len(axes):
                break
            
            cm = np.array([
                [all_metrics[model_name]['tn'], all_metrics[model_name]['fp']],
                [all_metrics[model_name]['fn'], all_metrics[model_name]['tp']]
            ])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            
            # Add percentage annotations
            for i in range(2):
                for j in range(2):
                    text = axes[idx].text(j + 0.5, i + 0.5, 
                                        f'{cm_percent[i, j]:.1f}%',
                                        ha='center', va='center', 
                                        color='red' if cm[i, j] > cm.max()/2 else 'black',
                                        fontsize=9)
            
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xticklabels(['Non-Fraud', 'Fraud'])
            axes[idx].set_yticklabels(['Non-Fraud', 'Fraud'])
        
        # Hide empty subplots
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Confusion Matrices ({dataset_name})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrices_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_benefit_analysis(self, all_metrics: Dict[str, Dict[str, float]], 
                                   dataset_name: str):
        """Plot cost-benefit analysis for all models"""
        
        models = list(all_metrics.keys())
        
        # Calculate costs and benefits
        costs = []
        frauds_caught = []
        false_alarms = []
        
        for model in models:
            tp = all_metrics[model]['tp']
            fp = all_metrics[model]['fp']
            fn = all_metrics[model]['fn']
            
            # Assuming: Fraud costs 10x more than false alarm
            fraud_cost = 10
            false_alarm_cost = 1
            
            total_cost = (fn * fraud_cost) + (fp * false_alarm_cost)
            frauds_prevented = tp * fraud_cost
            investigation_cost = fp * false_alarm_cost
            
            costs.append(total_cost)
            frauds_caught.append(tp)
            false_alarms.append(fp)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Total Cost
        colors1 = [self.colors.get(m.lower(), '#808080') for m in models]
        bars1 = axes[0].bar(models, costs, color=colors1)
        axes[0].set_title('Total Cost', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Cost', fontsize=10)
        axes[0].tick_params(axis='x', rotation=45)
        for bar, cost in zip(bars1, costs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{cost:.0f}', ha='center', va='bottom')
        
        # Plot 2: Frauds Caught
        colors2 = [self.colors.get(m.lower(), '#808080') for m in models]
        bars2 = axes[1].bar(models, frauds_caught, color=colors2)
        axes[1].set_title('Frauds Caught', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=10)
        axes[1].tick_params(axis='x', rotation=45)
        for bar, count in zip(bars2, frauds_caught):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom')
        
        # Plot 3: False Alarms
        colors3 = [self.colors.get(m.lower(), '#808080') for m in models]
        bars3 = axes[2].bar(models, false_alarms, color=colors3)
        axes[2].set_title('False Alarms', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Count', fontsize=10)
        axes[2].tick_params(axis='x', rotation=45)
        for bar, count in zip(bars3, false_alarms):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom')
        
        plt.suptitle(f'Cost-Benefit Analysis ({dataset_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/cost_benefit_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self, all_metrics: Dict[str, Dict[str, float]], 
                         dataset_name: str):
        """Create radar chart comparing multiple metrics"""
        
        try:
            from math import pi
            
            # Select key metrics for radar chart
            metrics = ['f1', 'roc_auc', 'precision', 'recall', 'accuracy', 'mcc']
            models = list(all_metrics.keys())
            
            # Prepare data
            values = []
            for model in models:
                model_values = [all_metrics[model].get(m, 0) for m in metrics]
                values.append(model_values)
            
            # Number of variables
            N = len(metrics)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Close the polygon
            
            # Initialise the spider plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Draw one axe per variable + add labels
            plt.xticks(angles[:-1], metrics, color='grey', size=10)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                      color="grey", size=8)
            plt.ylim(0, 1)
            
            # Plot each model
            for idx, model in enumerate(models):
                model_values = values[idx]
                model_values += model_values[:1]  # Close the polygon
                
                color = self.colors.get(model.lower(), '#808080')
                ax.plot(angles, model_values, linewidth=2, linestyle='solid', 
                       label=model, color=color)
                ax.fill(angles, model_values, alpha=0.1, color=color)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title(f'Model Comparison Radar Chart ({dataset_name})', 
                     size=14, fontweight='bold', y=1.1)
            
            plt.tight_layout()
            plt.savefig(f'plots/radar_chart_{dataset_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Could not create radar chart: {e}")
    
    def _plot_feature_importance(self, feature_importance: Dict[str, np.ndarray],
                                dataset_name: str, top_n: int = 15):
        """Plot feature importance for models"""
        
        models = list(feature_importance.keys())
        n_models = len(models)
        
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(models):
            if idx >= len(axes):
                break
            
            importance = feature_importance[model_name]
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[-top_n:][::-1]
            sorted_importance = importance[sorted_idx]
            
            # Create feature names if not provided
            feature_names = [f'Feature_{i}' for i in sorted_idx]
            
            # Plot
            bars = axes[idx].barh(range(len(feature_names)), sorted_importance, 
                                 color=self.colors.get(model_name.lower(), '#808080'))
            axes[idx].set_yticks(range(len(feature_names)))
            axes[idx].set_yticklabels(feature_names)
            axes[idx].set_xlabel('Importance', fontsize=10)
            axes[idx].set_title(f'{model_name} - Top {top_n} Features', 
                               fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()  # Highest importance at top
            
            # Add value labels
            for bar, value in zip(bars, sorted_importance):
                axes[idx].text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                             f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        plt.suptitle(f'Feature Importance ({dataset_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/feature_importance_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_weights(self, attention_weights: Dict[str, np.ndarray],
                               dataset_name: str):
        """Plot attention weights for HASTE model"""
        
        if 'haste' not in attention_weights:
            return
        
        weights = attention_weights['haste']
        
        if weights.ndim == 2:  # (n_samples, n_models)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Heatmap of first 50 samples
            sample_weights = weights[:50, :]
            im1 = axes[0].imshow(sample_weights.T, aspect='auto', cmap='viridis')
            axes[0].set_xlabel('Sample Index', fontsize=10)
            axes[0].set_ylabel('Base Model', fontsize=10)
            axes[0].set_title('Attention Weights (First 50 Samples)', fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=axes[0])
            
            # Plot 2: Average attention weights
            avg_weights = weights.mean(axis=0)
            bars = axes[1].bar(range(len(avg_weights)), avg_weights, 
                              color=self.colors.get('haste', '#3E885B'))
            axes[1].set_xlabel('Base Model', fontsize=10)
            axes[1].set_ylabel('Average Attention Weight', fontsize=10)
            axes[1].set_title('Average Attention Distribution', fontsize=12, fontweight='bold')
            axes[1].set_xticks(range(len(avg_weights)))
            axes[1].set_xticklabels([f'Model {i+1}' for i in range(len(avg_weights))])
            
            # Add value labels
            for bar, value in zip(bars, avg_weights):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'HASTE Attention Weights ({dataset_name})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'plots/attention_weights_{dataset_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_model_correlation(self, predictions_dict: Dict[str, np.ndarray],
                               dataset_name: str):
        """Plot correlation heatmap between model predictions"""
        
        models = list(predictions_dict.keys())
        n_models = len(models)
        
        if n_models < 2:
            return
        
        # Create correlation matrix
        predictions_matrix = np.column_stack([predictions_dict[m] for m in models])
        corr_matrix = np.corrcoef(predictions_matrix.T)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   xticklabels=models, yticklabels=models)
        
        plt.title(f'Model Predictions Correlation ({dataset_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/model_correlation_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, predictions_dict: Dict[str, np.ndarray],
                                y_test: np.ndarray, dataset_name: str):
        """Plot threshold analysis for models"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        thresholds = np.linspace(0, 1, 101)
        
        for idx, (model_name, y_prob) in enumerate(list(predictions_dict.items())[:4]):
            if idx >= 4:
                break
            
            # Calculate metrics at different thresholds
            precisions = []
            recalls = []
            f1_scores = []
            accuracies = []
            
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                
                precisions.append(precision_score(y_test, y_pred, zero_division=0))
                recalls.append(recall_score(y_test, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
                accuracies.append(accuracy_score(y_test, y_pred))
            
            color = self.colors.get(model_name.lower(), '#808080')
            
            axes[idx].plot(thresholds, precisions, label='Precision', linewidth=2, color=color, linestyle='-')
            axes[idx].plot(thresholds, recalls, label='Recall', linewidth=2, color=color, linestyle='--')
            axes[idx].plot(thresholds, f1_scores, label='F1', linewidth=2, color=color, linestyle=':')
            axes[idx].plot(thresholds, accuracies, label='Accuracy', linewidth=2, color=color, linestyle='-.')
            
            # Find optimal threshold (max F1)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            axes[idx].axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                            label=f'Optimal: {optimal_threshold:.2f}')
            
            axes[idx].set_xlabel('Threshold', fontsize=10)
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].set_title(f'{model_name} - Threshold Analysis', fontsize=12, fontweight='bold')
            axes[idx].legend(loc='lower left', fontsize=8)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
        
        plt.suptitle(f'Threshold Analysis ({dataset_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/threshold_analysis_{dataset_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_report(self, all_metrics: Dict[str, Dict[str, float]],
                                 predictions_dict: Dict[str, np.ndarray],
                                 y_test: np.ndarray,
                                 dataset_name: str = 'test',
                                 config: Optional[Dict] = None):
        """Save comprehensive report with all metrics"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = f"{self.results_dir}/report_{dataset_name}_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\nSaving comprehensive report to: {report_dir}")
        
        # 1. Save metrics as JSON
        with open(f'{report_dir}/metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=float)
        
        # 2. Save predictions
        predictions_df = pd.DataFrame(predictions_dict)
        predictions_df['true_label'] = y_test
        predictions_df.to_csv(f'{report_dir}/predictions.csv', index=False)
        
        # 3. Save configuration
        if config:
            with open(f'{report_dir}/config.json', 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        # 4. Create detailed text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE FRAUD DETECTION REPORT")
        report_lines.append(f"Dataset: {dataset_name}")
        report_lines.append(f"Timestamp: {timestamp}")
        report_lines.append(f"Total samples: {len(y_test)}")
        report_lines.append(f"Fraud rate: {y_test.mean():.4%}")
        report_lines.append("=" * 80)
        
        # Model comparison table
        report_lines.append("\nMODEL PERFORMANCE SUMMARY:")
        report_lines.append("-" * 80)
        
        # Create header
        headers = ["Model", "F1", "AUC", "Precision", "Recall", "Accuracy", "MCC", "Cost"]
        header_line = " | ".join([f"{h:<15}" for h in headers])
        report_lines.append(header_line)
        report_lines.append("-" * 80)
        
        # Add model rows
        for model_name, metrics in all_metrics.items():
            row = [
                model_name[:15],
                f"{metrics.get('f1', 0):.4f}",
                f"{metrics.get('roc_auc', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('mcc', 0):.4f}",
                f"{metrics.get('total_cost', 0):.0f}"
            ]
            row_line = " | ".join([f"{r:<15}" for r in row])
            report_lines.append(row_line)
        
        report_lines.append("-" * 80)
        
        # Find best model by each metric
        best_by_metric = {}
        for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy', 'mcc']:
            if any(metric in m for m in all_metrics.values()):
                best_model = max(all_metrics.items(), 
                               key=lambda x: x[1].get(metric, 0))[0]
                best_value = all_metrics[best_model].get(metric, 0)
                best_by_metric[metric] = (best_model, best_value)
        
        report_lines.append("\nBEST MODELS BY METRIC:")
        report_lines.append("-" * 80)
        for metric, (model, value) in best_by_metric.items():
            report_lines.append(f"{metric.upper():<12}: {model} ({value:.4f})")
        
        # Statistical significance testing
        if len(predictions_dict) > 1:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("STATISTICAL SIGNIFICANCE TESTING")
            report_lines.append("-" * 80)
            
            # McNemar's test for pairwise comparison
            from scipy.stats import binomtest
            
            models = list(predictions_dict.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model_a = models[i]
                    model_b = models[j]
                    
                    pred_a = (predictions_dict[model_a] > 0.5).astype(int)
                    pred_b = (predictions_dict[model_b] > 0.5).astype(int)
                    
                    # Contingency table
                    both_correct = ((pred_a == y_test) & (pred_b == y_test)).sum()
                    both_wrong = ((pred_a != y_test) & (pred_b != y_test)).sum()
                    a_correct_b_wrong = ((pred_a == y_test) & (pred_b != y_test)).sum()
                    b_correct_a_wrong = ((pred_a != y_test) & (pred_b == y_test)).sum()
                    
                    # McNemar's test
                    n = a_correct_b_wrong + b_correct_a_wrong
                    k = min(a_correct_b_wrong, b_correct_a_wrong)
                    
                    if n > 0:
                        result = binomtest(k, n, p=0.5, alternative='two-sided')
                        significant = result.pvalue < 0.05
                        
                        report_lines.append(f"\n{model_a} vs {model_b}:")
                        report_lines.append(f"  Disagreements: {n}")
                        report_lines.append(f"  {model_a} better: {a_correct_b_wrong}")
                        report_lines.append(f"  {model_b} better: {b_correct_a_wrong}")
                        report_lines.append(f"  p-value: {result.pvalue:.4f}")
                        report_lines.append(f"  Significant difference: {'YES' if significant else 'NO'}")
        
        # Business impact analysis
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BUSINESS IMPACT ANALYSIS")
        report_lines.append("-" * 80)
        
        for model_name, metrics in all_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            # Assuming each fraud costs $1000 and each false alarm costs $50
            fraud_cost = 1000
            false_alarm_cost = 50
            
            losses_prevented = tp * fraud_cost
            investigation_costs = fp * false_alarm_cost
            missed_fraud_losses = fn * fraud_cost
            
            net_benefit = losses_prevented - investigation_costs - missed_fraud_losses
            roi = (losses_prevented - investigation_costs) / investigation_costs if investigation_costs > 0 else float('inf')
            
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Frauds caught: {tp} (${losses_prevented:,.0f} prevented)")
            report_lines.append(f"  False alarms: {fp} (${investigation_costs:,.0f} cost)")
            report_lines.append(f"  Frauds missed: {fn} (${missed_fraud_losses:,.0f} lost)")
            report_lines.append(f"  Net benefit: ${net_benefit:,.0f}")
            report_lines.append(f"  ROI: {roi:.1f}x" if roi != float('inf') else "  ROI: Infinite")
        
        # Save report
        with open(f'{report_dir}/detailed_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        # 5. Create summary CSV
        summary_data = []
        for model_name, metrics in all_metrics.items():
            summary_data.append({
                'model': model_name,
                'f1': metrics.get('f1', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'accuracy': metrics.get('accuracy', 0),
                'mcc': metrics.get('mcc', 0),
                'tp': metrics.get('tp', 0),
                'fp': metrics.get('fp', 0),
                'fn': metrics.get('fn', 0),
                'tn': metrics.get('tn', 0),
                'total_cost': metrics.get('total_cost', 0),
                'cost_savings': metrics.get('cost_savings', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{report_dir}/summary.csv', index=False)
        
        print(f"  ✓ Saved comprehensive report to: {report_dir}")
        
        return report_dir


class AblationStudy:
    """Perform ablation study to understand component importance"""
    
    def __init__(self, results_dir: str = 'results/ablation_study'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def study_haste_components(self, X_train, y_train, X_test, y_test, 
                              base_predictions, device='cpu'):
        """Study importance of HASTE components"""
        
        print("\n" + "="*60)
        print("ABLATION STUDY: HASTE COMPONENTS")
        print("="*60)
        
        results = {}
        
        # 1. Baseline: Simple average of base models
        print("\n1. Baseline: Simple Average")
        avg_predictions = base_predictions.mean(axis=1)
        avg_metrics = self._evaluate_model(avg_predictions, y_test, 'simple_average')
        results['simple_average'] = avg_metrics
        
        # 2. Weighted average (learned weights)
        print("\n2. Weighted Average (learned weights)")
        from sklearn.linear_model import LogisticRegression
        weights_model = LogisticRegression()
        weights_model.fit(base_predictions, y_train)
        weighted_pred = weights_model.predict_proba(base_predictions)[:, 1]
        weighted_metrics = self._evaluate_model(weighted_pred, y_test, 'weighted_average')
        results['weighted_average'] = weighted_metrics
        
        # 3. HASTE without attention (simple concatenation)
        print("\n3. HASTE without Attention")
        class NoAttentionHASTE:
            def __init__(self, input_size, hidden_size=64):
                self.model = self._build_model(input_size, hidden_size)
            
            def _build_model(self, input_size, hidden_size):
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
        
        # 4. HASTE without feature extraction
        print("\n4. HASTE without Feature Extraction")
        
        # 5. Full HASTE model
        print("\n5. Full HASTE Model")
        
        # Create comparison plot
        self._plot_ablation_results(results)
        
        return results
    
    def _evaluate_model(self, predictions, y_true, model_name):
        """Evaluate a single model"""
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        
        y_pred = (predictions > 0.5).astype(int)
        
        return {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, predictions),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    def _plot_ablation_results(self, results):
        """Plot ablation study results"""
        
        models = list(results.keys())
        metrics = ['f1', 'auc', 'accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [results[m][metric] for m in models]
            
            bars = axes[idx].bar(models, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B'])
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric.upper(), fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{value:.3f}', ha='center', va='bottom')
            
            axes[idx].set_ylim(0, 1.1)
        
        plt.suptitle('Ablation Study: HASTE Component Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/ablation_study.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Ablation study results saved to: {self.results_dir}/")


def run_comprehensive_evaluation(models_dict, X_test, y_test, predictions_dict,
                                feature_importance=None, attention_weights=None,
                                config=None, dataset_name='test'):
    """Run comprehensive evaluation pipeline"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate all models
    all_metrics = evaluator.evaluate_all_models(models_dict, X_test, y_test, dataset_name)
    
    # Create all plots
    evaluator.create_all_plots(all_metrics, y_test, predictions_dict, 
                              feature_importance, attention_weights, dataset_name)
    
    # Save comprehensive report
    report_dir = evaluator.save_comprehensive_report(all_metrics, predictions_dict, 
                                                    y_test, dataset_name, config)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print("-" * 40)
    
    # Find best model by F1 score
    if all_metrics:
        best_model = max(all_metrics.items(), key=lambda x: x[1].get('f1', 0))[0]
        best_f1 = all_metrics[best_model]['f1']
        best_auc = all_metrics[best_model].get('roc_auc', 0)
        
        print(f"Best Model: {best_model}")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  ROC-AUC: {best_auc:.4f}")
        print(f"  Precision: {all_metrics[best_model].get('precision', 0):.4f}")
        print(f"  Recall: {all_metrics[best_model].get('recall', 0):.4f}")
        
        # Compare with baseline (Random Forest usually)
        if 'Random Forest' in all_metrics:
            baseline_f1 = all_metrics['Random Forest']['f1']
            improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
            print(f"\nImprovement over Random Forest: {improvement:+.2f}%")
    
    print(f"\n✓ Comprehensive report saved to: {report_dir}")
    print(f"✓ All plots saved to: plots/")
    
    return all_metrics, report_dir