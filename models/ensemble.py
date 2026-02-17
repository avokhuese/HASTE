"""
Ensemble methods for fraud detection
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsemble:
    """Advanced ensemble methods for fraud detection"""
    
    def __init__(self, base_models: Dict[str, Any], 
                 ensemble_method: str = 'stacking', 
                 meta_model_type: str = 'logistic'):
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.meta_model_type = meta_model_type
        self.ensemble_model = None
        self.meta_model = None
    
    def create_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                                cv_folds: int = 5) -> Any:
        """Create stacking ensemble"""
        print("Creating stacking ensemble...")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Choose meta model
        if self.meta_model_type == 'logistic':
            meta_model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                solver='lbfgs'
            )
        elif self.meta_model_type == 'rf':
            meta_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced_subsample'
            )
        elif self.meta_model_type == 'xgboost':
            from xgboost import XGBClassifier
            meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown meta model type: {self.meta_model_type}")
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=meta_model,
            cv=cv_folds,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
        
        return stacking_model
    
    def create_voting_ensemble(self, voting_type: str = 'soft') -> Any:
        """Create voting ensemble"""
        print(f"Creating {voting_type} voting ensemble...")
        
        voting_model = VotingClassifier(
            estimators=list(self.base_models.items()),
            voting=voting_type,
            weights=None,  # Equal weights
            n_jobs=-1
        )
        
        return voting_model
    
    def create_dynamic_weighted_ensemble(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Create dynamic weighted ensemble based on validation performance"""
        print("Creating dynamic weighted ensemble...")
        
        weights = {}
        
        for name, model in self.base_models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = model.predict(X_val)
                
                # Calculate performance metric (F1 score)
                from sklearn.metrics import f1_score
                y_pred = (y_pred_proba > 0.5).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # Store weight (F1 score as weight)
                weights[name] = f1
                
                print(f"  {name}: F1 = {f1:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                weights[name] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def dynamic_weighted_predict(self, X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Make predictions using dynamic weighted ensemble"""
        predictions = []
        
        for name, model in self.base_models.items():
            if name in weights and weights[name] > 0:
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    
                    predictions.append(pred * weights[name])
                except:
                    continue
        
        if predictions:
            weighted_pred = np.sum(predictions, axis=0)
            return weighted_pred
        else:
            return np.zeros(len(X))
    
    def create_calibrated_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  method: str = 'isotonic', cv: int = 5) -> Dict[str, Any]:
        """Create calibrated ensemble"""
        print(f"Creating calibrated ensemble ({method})...")
        
        calibrated_models = {}
        
        for name, model in self.base_models.items():
            try:
                calibrated_model = CalibratedClassifierCV(
                    estimator=model,
                    method=method,
                    cv=cv,
                    n_jobs=-1
                )
                
                calibrated_model.fit(X_train, y_train)
                calibrated_models[name] = calibrated_model
                
                print(f"  Calibrated {name}")
                
            except Exception as e:
                print(f"  Error calibrating {name}: {e}")
                calibrated_models[name] = model
        
        return calibrated_models
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None) -> Any:
        """Train ensemble model"""
        
        if self.ensemble_method == 'stacking':
            self.ensemble_model = self.create_stacking_ensemble(X_train, y_train)
        
        elif self.ensemble_method == 'voting_soft':
            self.ensemble_model = self.create_voting_ensemble('soft')
        
        elif self.ensemble_method == 'voting_hard':
            self.ensemble_model = self.create_voting_ensemble('hard')
        
        elif self.ensemble_method == 'dynamic_weighted':
            if X_val is None or y_val is None:
                raise ValueError("Validation data required for dynamic weighted ensemble")
            
            self.weights = self.create_dynamic_weighted_ensemble(X_val, y_val)
            return self.weights
        
        elif self.ensemble_method == 'calibrated':
            self.calibrated_models = self.create_calibrated_ensemble(X_train, y_train)
            return self.calibrated_models
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Fit ensemble model
        if self.ensemble_method in ['stacking', 'voting_soft', 'voting_hard']:
            self.ensemble_model.fit(X_train, y_train)
            print(f"{self.ensemble_method.capitalize()} ensemble trained successfully")
        
        return self.ensemble_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with ensemble"""
        
        if self.ensemble_method == 'dynamic_weighted':
            return self.dynamic_weighted_predict(X, self.weights)
        
        elif self.ensemble_method == 'calibrated':
            predictions = []
            for name, model in self.calibrated_models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
        
        else:
            if hasattr(self.ensemble_model, 'predict_proba'):
                return self.ensemble_model.predict_proba(X)[:, 1]
            else:
                return self.ensemble_model.predict(X)


class NeuralEnsemble(nn.Module):
    """Neural network based ensemble"""
    
    def __init__(self, n_models: int, input_size: int, hidden_size: int = 128):
        super().__init__()
        
        # Attention-based weighting
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_models),
            nn.Softmax(dim=-1)
        )
        
        # Meta network
        self.meta_network = nn.Sequential(
            nn.Linear(n_models + input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, base_predictions: torch.Tensor, 
                input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        # Compute attention weights
        attention_weights = self.attention(input_features)
        
        # Apply attention to base predictions
        weighted_predictions = (attention_weights * base_predictions).sum(dim=1, keepdim=True)
        
        # Combine with input features
        combined = torch.cat([weighted_predictions, input_features], dim=1)
        
        # Final prediction
        output = self.meta_network(combined)
        
        return output.squeeze(), attention_weights


class BayesianEnsemble:
    """Bayesian ensemble with uncertainty estimation"""
    
    def __init__(self, base_models: Dict[str, Any], n_samples: int = 1000):
        self.base_models = base_models
        self.n_samples = n_samples
        self.model_weights = None
    
    def fit(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fit Bayesian ensemble using validation data"""
        from scipy import stats
        
        n_models = len(self.base_models)
        performances = np.zeros(n_models)
        
        # Evaluate each model
        for idx, (name, model) in enumerate(self.base_models.items()):
            try:
                y_pred = model.predict_proba(X_val)[:, 1]
                # Use AUC as performance metric
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, y_pred)
                performances[idx] = auc
            except:
                performances[idx] = 0.5  # Default performance
        
        # Bayesian weighting using Dirichlet distribution
        # Higher performance -> higher concentration parameter
        alpha = performances * 10 + 1  # Scale and add 1 for positivity
        
        # Sample from Dirichlet distribution
        dirichlet_samples = stats.dirichlet.rvs(alpha, size=self.n_samples)
        
        # Average weights
        self.model_weights = dirichlet_samples.mean(axis=0)
        
        print("Bayesian ensemble weights:")
        for (name, _), weight in zip(self.base_models.items(), self.model_weights):
            print(f"  {name}: {weight:.4f}")
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """Make predictions with uncertainty estimation"""
        from scipy import stats
        
        if self.model_weights is None:
            raise ValueError("Model must be fitted first")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.base_models.values():
            try:
                preds = model.predict_proba(X)[:, 1]
                all_predictions.append(preds)
            except:
                all_predictions.append(np.zeros(len(X)))
        
        all_predictions = np.array(all_predictions)  # (n_models, n_samples)
        
        # Weighted average prediction
        weighted_pred = np.average(all_predictions, axis=0, weights=self.model_weights)
        
        if return_uncertainty:
            # Calculate prediction uncertainty
            variance = np.average((all_predictions - weighted_pred) ** 2, 
                                 axis=0, weights=self.model_weights)
            uncertainty = np.sqrt(variance)
            
            return weighted_pred, uncertainty
        else:
            return weighted_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        preds = self.predict(X, return_uncertainty=False)
        # Convert to 2D probability array
        proba = np.column_stack([1 - preds, preds])
        return proba