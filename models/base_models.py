"""
Base models for fraud detection ensemble
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as LGBMClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any, Tuple, Optional
import joblib
from tqdm import tqdm

class BaseModelFactory:
    """Factory for creating base models"""
    
    def __init__(self, device: str = 'cpu', random_state: int = 42):
        self.device = device
        self.random_state = random_state
        self.models = {}
        self.models_config = {
            'rf': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced_subsample',
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'xgb': {
                'class': XGBClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 10,  # For class imbalance
                    'random_state': random_state,
                    'tree_method': 'hist' if device == 'cpu' else 'gpu_hist',
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            },
            'catboost': {
                'class': CatBoostClassifier,
                'params': {
                    'iterations': 500,
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 3,
                    'border_count': 128,
                    'random_seed': random_state,
                    'verbose': False,
                    'task_type': 'GPU' if 'cuda' in device else 'CPU',
                    'auto_class_weights': 'Balanced'
                }
            },
            'svm': {
                'class': SVC,
                'params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,
                    'class_weight': 'balanced',
                    'random_state': random_state
                }
            },
            'lightgbm': {
                'class': LGBMClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'num_leaves': 63,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'scale_pos_weight': 10,
                    'device': 'gpu' if 'cuda' in device else 'cpu'
                }
            },
            'mlp': {
                'class': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive',
                    'max_iter': 200,
                    'early_stopping': True,
                    'random_state': random_state
                }
            }
        }
    
    def create_model(self, model_name: str) -> Any:
        """Create a specific model"""
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.models_config[model_name]
        model_class = config['class']
        params = config['params'].copy()
        
        # Handle GPU-specific parameters
        if model_name == 'xgb' and 'cuda' in self.device:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
        
        return model_class(**params)
    
    def create_all_models(self, model_names: Optional[list] = None) -> Dict[str, Any]:
        """Create all specified models"""
        if model_names is None:
            model_names = list(self.models_config.keys())
        
        print(f"Creating {len(model_names)} base models...")
        
        for name in tqdm(model_names, desc="Initializing models"):
            self.models[name] = self.create_model(name)
        
        return self.models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: Optional[np.ndarray] = None, 
                    y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models"""
        print(f"\nTraining {len(self.models)} base models...")
        
        trained_models = {}
        predictions = {}
        
        for name, model in tqdm(self.models.items(), desc="Training models"):
            print(f"\nTraining {name}...")
            
            try:
                # Special handling for each model type
                if name == 'catboost':
                    # CatBoost needs eval set
                    if X_val is not None and y_val is not None:
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=50,
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train, verbose=False)
                
                elif name == 'xgb':
                    # XGBoost with early stopping
                    if X_val is not None and y_val is not None:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=50,
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train, verbose=False)
                
                elif name == 'lightgbm':
                    # LightGBM with early stopping
                    if X_val is not None and y_val is not None:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric='binary_logloss',
                            early_stopping_rounds=50,
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train, verbose=False)
                
                else:
                    # Standard sklearn models
                    model.fit(X_train, y_train)
                
                # Store trained model
                trained_models[name] = model
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    preds_train = model.predict_proba(X_train)[:, 1]
                    if X_val is not None:
                        preds_val = model.predict_proba(X_val)[:, 1]
                    else:
                        preds_val = None
                else:
                    preds_train = model.predict(X_train)
                    if X_val is not None:
                        preds_val = model.predict(X_val)
                    else:
                        preds_val = None
                
                predictions[name] = {
                    'train': preds_train,
                    'val': preds_val
                }
                
                print(f"  {name} trained successfully")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        return trained_models, predictions
    
    def save_models(self, path: str = 'models/base_models'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{path}/{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  Saved {name} to {model_path}")
    
    def load_models(self, path: str = 'models/base_models'):
        """Load trained models"""
        import glob
        
        model_files = glob.glob(f"{path}/*.joblib")
        
        for file_path in model_files:
            model_name = file_path.split('/')[-1].replace('.joblib', '')
            self.models[model_name] = joblib.load(file_path)
            print(f"  Loaded {model_name} from {file_path}")
        
        return self.models


class NeuralBaseModel(nn.Module):
    """Neural network base model for GPU training"""
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (128, 64, 32), 
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()


class LSTMModel(nn.Module):
    """LSTM model for sequential fraud detection"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3, 
                 bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        output = self.classifier(last_output)
        
        return output.squeeze()


class AttentionModel(nn.Module):
    """Attention-based model for fraud detection"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        attn_output, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        # Use mean pooling
        pooled = x.mean(dim=1)
        output = self.classifier(pooled)
        
        return output.squeeze(), attn_weights