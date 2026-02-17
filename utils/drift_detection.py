"""
Simplified drift detection
"""
import numpy as np
from scipy import stats

class ConceptDriftDetector:
    """Simple concept drift detector"""
    
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
    
    def detect_multivariate_drift(self, old_data, new_data):
        """Detect drift using KS test"""
        
        if len(old_data.shape) == 1:
            old_data = old_data.reshape(-1, 1)
            new_data = new_data.reshape(-1, 1)
        
        n_features = old_data.shape[1]
        p_values = []
        
        for i in range(n_features):
            try:
                _, p_value = stats.ks_2samp(old_data[:, i], new_data[:, i])
                p_values.append(p_value)
            except:
                p_values.append(1.0)
        
        avg_p_value = np.mean(p_values)
        drift_detected = avg_p_value < self.threshold
        
        # Calculate covariance difference
        try:
            cov_old = np.cov(old_data.T)
            cov_new = np.cov(new_data.T)
            cov_diff = np.linalg.norm(cov_old - cov_new, 'fro')
        except:
            cov_diff = 0.0
        
        return {
            'ks_p_value': float(avg_p_value),
            'mmd_score': float(1 - avg_p_value),  # Simplified MMD
            'covariance_diff': float(cov_diff),
            'overall_drift': drift_detected,
            'overall_score': float(1 - avg_p_value)
        }