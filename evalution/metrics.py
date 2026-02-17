"""
Evaluation metrics for fraud detection
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    cohen_kappa_score, fbeta_score, log_loss, brier_score_loss
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from typing import Tuple, Dict, Any, List, Optional

class FraudDetectionMetrics:
    """Comprehensive metrics for fraud detection evaluation"""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        
        metrics = {}
        
        # Binary classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        # Correlation metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
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
        
        return metrics
    
    @staticmethod
    def calculate_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced fraud-specific metrics"""
        
        metrics = {}
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Fraud-specific metrics
        metrics['fraud_capture_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['precision_fraud'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Cost-sensitive metrics (assuming fraud costs 10x more than false alarm)
        cost_fraud = 10  # Cost of missing a fraud
        cost_false_alarm = 1  # Cost of false alarm
        
        total_cost = (fn * cost_fraud) + (fp * cost_false_alarm)
        max_cost = len(y_true) * cost_fraud  # If all frauds missed
        metrics['cost_savings'] = 1 - (total_cost / max_cost) if max_cost > 0 else 0
        
        # Gini coefficient (2 * AUC - 1)
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
                metrics['gini'] = 2 * auc - 1
            except:
                metrics['gini'] = 0.0
        
        # Kolmogorov-Smirnov statistic
        if y_prob is not None:
            try:
                fraud_probs = y_prob[y_true == 1]
                non_fraud_probs = y_prob[y_true == 0]
                
                if len(fraud_probs) > 0 and len(non_fraud_probs) > 0:
                    ks_stat, _ = stats.ks_2samp(fraud_probs, non_fraud_probs)
                    metrics['ks_statistic'] = ks_stat
                else:
                    metrics['ks_statistic'] = 0.0
            except:
                metrics['ks_statistic'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  transaction_amounts: Optional[np.ndarray] = None,
                                  fraud_cost_multiplier: float = 10.0) -> Dict[str, float]:
        """Calculate business-oriented metrics"""
        
        metrics = {}
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Default metrics
        metrics['frauds_detected'] = tp
        metrics['frauds_missed'] = fn
        metrics['false_alarms'] = fp
        
        # Amount-based metrics if transaction amounts provided
        if transaction_amounts is not None and len(transaction_amounts) == len(y_true):
            fraud_amounts = transaction_amounts[y_true == 1]
            detected_fraud_amounts = transaction_amounts[(y_true == 1) & (y_pred == 1)]
            missed_fraud_amounts = transaction_amounts[(y_true == 1) & (y_pred == 0)]
            
            metrics['total_fraud_amount'] = fraud_amounts.sum()
            metrics['detected_fraud_amount'] = detected_fraud_amounts.sum()
            metrics['missed_fraud_amount'] = missed_fraud_amounts.sum()
            metrics['fraud_amount_recovery_rate'] = (
                detected_fraud_amounts.sum() / fraud_amounts.sum() 
                if fraud_amounts.sum() > 0 else 0
            )
        
        # Cost-benefit analysis
        avg_transaction_value = 100  # Default average transaction value
        
        # Benefits: Fraud losses prevented
        fraud_prevention_benefit = tp * avg_transaction_value * fraud_cost_multiplier
        
        # Costs: Investigation costs for false alarms
        investigation_cost_per_alarm = 50  # Cost to investigate a false alarm
        false_alarm_cost = fp * investigation_cost_per_alarm
        
        metrics['net_benefit'] = fraud_prevention_benefit - false_alarm_cost
        metrics['return_on_investment'] = (
            fraud_prevention_benefit / false_alarm_cost 
            if false_alarm_cost > 0 else float('inf')
        )
        
        return metrics
    
    @staticmethod
    def calculate_drift_metrics(y_true_old: np.ndarray, y_pred_old: np.ndarray,
                               y_true_new: np.ndarray, y_pred_new: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for concept drift detection"""
        
        metrics = {}
        
        # Performance degradation
        f1_old = f1_score(y_true_old, y_pred_old, zero_division=0)
        f1_new = f1_score(y_true_new, y_pred_new, zero_division=0)
        metrics['performance_degradation'] = f1_old - f1_new
        
        # Distribution shift (simplified)
        pred_ratio_old = y_pred_old.mean()
        pred_ratio_new = y_pred_new.mean()
        metrics['prediction_distribution_shift'] = abs(pred_ratio_old - pred_ratio_new)
        
        # Error rate change
        error_rate_old = 1 - accuracy_score(y_true_old, y_pred_old)
        error_rate_new = 1 - accuracy_score(y_true_new, y_pred_new)
        metrics['error_rate_change'] = error_rate_new - error_rate_old
        
        return metrics
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray] = None,
                             transaction_amounts: Optional[np.ndarray] = None,
                             y_true_old: Optional[np.ndarray] = None,
                             y_pred_old: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all metrics"""
        
        all_metrics = {}
        
        # Basic metrics
        basic_metrics = FraudDetectionMetrics.calculate_basic_metrics(y_true, y_pred, y_prob)
        all_metrics.update(basic_metrics)
        
        # Advanced metrics
        advanced_metrics = FraudDetectionMetrics.calculate_advanced_metrics(y_true, y_pred, y_prob)
        all_metrics.update(advanced_metrics)
        
        # Business metrics
        business_metrics = FraudDetectionMetrics.calculate_business_metrics(
            y_true, y_pred, transaction_amounts
        )
        all_metrics.update(business_metrics)
        
        # Drift metrics if old data provided
        if y_true_old is not None and y_pred_old is not None:
            drift_metrics = FraudDetectionMetrics.calculate_drift_metrics(
                y_true_old, y_pred_old, y_true, y_pred
            )
            all_metrics.update(drift_metrics)
        
        return all_metrics
    
    @staticmethod
    def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None,
                                      n_bootstraps: int = 1000,
                                      confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using bootstrapping"""
        
        n_samples = len(y_true)
        metrics_list = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices] if y_prob is not None else None
            
            # Calculate metrics
            metrics = FraudDetectionMetrics.calculate_basic_metrics(
                y_true_boot, y_pred_boot, y_prob_boot
            )
            metrics_list.append(metrics)
        
        # Calculate confidence intervals
        ci_dict = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            lower = np.percentile(values, (1 - confidence_level) / 2 * 100)
            upper = np.percentile(values, (1 + confidence_level) / 2 * 100)
            ci_dict[metric_name] = (float(lower), float(upper))
        
        return ci_dict
    
    @staticmethod
    def get_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal threshold for a given metric"""
        
        if metric == 'f1':
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = f1_scores[best_idx]
            
        elif metric == 'cost':
            # Find threshold that minimizes cost
            costs = []
            for threshold in np.linspace(0, 1, 101):
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                cost = fn * 10 + fp * 1  # Fraud costs 10x more
                costs.append(cost)
            
            best_idx = np.argmin(costs)
            best_threshold = np.linspace(0, 1, 101)[best_idx]
            best_score = costs[best_idx]
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_threshold, best_score
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None,
                                      model_name: str = "Model") -> str:
        """Generate comprehensive classification report"""
        
        from tabulate import tabulate
        
        # Calculate all metrics
        metrics = FraudDetectionMetrics.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Prepare table
        table_data = [
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["Precision", f"{metrics['precision']:.4f}"],
            ["Recall", f"{metrics['recall']:.4f}"],
            ["F1-Score", f"{metrics['f1']:.4f}"],
            ["F2-Score", f"{metrics['f2']:.4f}"],
            ["MCC", f"{metrics['mcc']:.4f}"],
            ["Cohen's Kappa", f"{metrics['kappa']:.4f}"]
        ]
        
        if y_prob is not None:
            table_data.extend([
                ["ROC-AUC", f"{metrics['roc_auc']:.4f}"],
                ["PR-AUC", f"{metrics['pr_auc']:.4f}"],
                ["Gini", f"{metrics['gini']:.4f}"],
                ["Log Loss", f"{metrics['log_loss']:.4f}"],
                ["Brier Score", f"{metrics['brier_score']:.4f}"]
            ])
        
        table_data.extend([
            ["Fraud Capture Rate", f"{metrics['fraud_capture_rate']:.4f}"],
            ["False Alarm Rate", f"{metrics['false_alarm_rate']:.4f}"],
            ["Cost Savings", f"{metrics['cost_savings']:.4f}"],
            ["KS Statistic", f"{metrics['ks_statistic']:.4f}"]
        ])
        
        # Generate report
        report = f"\n{'='*60}\n"
        report += f"CLASSIFICATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"
        report += tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")
        report += f"\n\nConfusion Matrix:\n"
        report += f"  True Negatives: {metrics['true_negative']}\n"
        report += f"  False Positives: {metrics['false_positive']}\n"
        report += f"  False Negatives: {metrics['false_negative']}\n"
        report += f"  True Positives: {metrics['true_positive']}\n"
        
        if 'net_benefit' in metrics:
            report += f"\nBusiness Metrics:\n"
            report += f"  Net Benefit: ${metrics['net_benefit']:.2f}\n"
            report += f"  ROI: {metrics['return_on_investment']:.2f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report