"""
Training module for fraud detection models
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import joblib
import os

class FraudDataset(Dataset):
    """Custom dataset for fraud detection"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 base_predictions: Optional[np.ndarray] = None,
                 sequence_length: int = 10, 
                 create_sequences: bool = False):
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.base_predictions = base_predictions.astype(np.float32) if base_predictions is not None else None
        self.sequence_length = sequence_length
        self.create_sequences = create_sequences
        
        if create_sequences and len(X) > sequence_length:
            self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences for time series models"""
        sequences_X = []
        sequences_y = []
        sequences_pred = [] if self.base_predictions is not None else None
        
        for i in range(len(self.X) - self.sequence_length):
            sequences_X.append(self.X[i:i + self.sequence_length])
            sequences_y.append(self.y[i + self.sequence_length - 1])
            
            if sequences_pred is not None:
                sequences_pred.append(self.base_predictions[i:i + self.sequence_length])
        
        self.X = np.array(sequences_X)
        self.y = np.array(sequences_y)
        
        if sequences_pred is not None:
            self.base_predictions = np.array(sequences_pred)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.base_predictions is not None:
            if self.create_sequences:
                return (
                    self.base_predictions[idx],  # shape: (seq_len, n_models)
                    self.X[idx],  # shape: (seq_len, n_features)
                    self.y[idx],  # scalar
                    self.base_predictions[idx]  # temporal sequence
                )
            else:
                return (
                    self.base_predictions[idx],  # shape: (n_models,)
                    self.X[idx],  # shape: (n_features,)
                    self.y[idx],  # scalar
                    None  # No temporal sequence
                )
        else:
            if self.create_sequences:
                return (
                    self.X[idx],  # shape: (seq_len, n_features)
                    self.y[idx],  # scalar
                    self.X[idx]   # temporal sequence
                )
            else:
                return (
                    self.X[idx],  # shape: (n_features,)
                    self.y[idx],  # scalar
                    None  # No temporal sequence
                )


class ModelTrainer:
    """Main trainer for fraud detection models"""
    
    def __init__(self, config: Any, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.base_models = {}
        self.haste_model = None
        self.ensemble_model = None
        self.train_history = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    sequence_length: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders"""
        
        print("Preparing data loaders...")
        
        # Create datasets
        train_dataset = FraudDataset(X_train, y_train, sequence_length=sequence_length)
        val_dataset = FraudDataset(X_val, y_val, sequence_length=sequence_length)
        test_dataset = FraudDataset(X_test, y_test, sequence_length=sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Train base models for ensemble"""
        
        print("\n" + "="*50)
        print("TRAINING BASE MODELS")
        print("="*50)
        
        from models.base_models import BaseModelFactory
        
        # Create base models
        model_factory = BaseModelFactory(device=self.device, 
                                        random_state=self.config.RANDOM_STATE)
        
        # Create all models
        self.base_models = model_factory.create_all_models(
            self.config.BASE_MODELS
        )
        
        # Train models
        trained_models, predictions = model_factory.train_models(
            X_train, y_train, X_val, y_val
        )
        
        # Save models
        model_factory.save_models('models/base_models')
        
        # Prepare stacking features
        X_train_stack = self._prepare_stacking_features(predictions, 'train')
        X_val_stack = self._prepare_stacking_features(predictions, 'val')
        
        print(f"\nBase models trained:")
        print(f"  Number of models: {len(trained_models)}")
        print(f"  Stacking features shape: {X_train_stack.shape}")
        
        return trained_models, X_train_stack, X_val_stack
    
    def _prepare_stacking_features(self, predictions: Dict[str, Dict[str, np.ndarray]], 
                                  dataset_type: str = 'train') -> np.ndarray:
        """Prepare stacking features from base model predictions"""
        
        stacking_features = []
        
        for model_name, pred_dict in predictions.items():
            if dataset_type in pred_dict and pred_dict[dataset_type] is not None:
                stacking_features.append(pred_dict[dataset_type])
        
        if stacking_features:
            return np.column_stack(stacking_features)
        else:
            return np.zeros((len(predictions[list(predictions.keys())[0]][dataset_type]), 1))
    
    def train_haste(self, X_train_stack: np.ndarray, y_train: np.ndarray,
                   X_val_stack: np.ndarray, y_val: np.ndarray,
                   X_train_features: Optional[np.ndarray] = None,
                   X_val_features: Optional[np.ndarray] = None) -> Any:
        """Train HASTE model"""
        
        print("\n" + "="*50)
        print("TRAINING HASTE MODEL")
        print("="*50)
        
        from models.haste import HASTE, HASTETrainer
        
        # Prepare data for HASTE
        n_base_models = X_train_stack.shape[1]
        input_size = X_train_features.shape[1] if X_train_features is not None else n_base_models
        
        # Create HASTE model
        self.haste_model = HASTE(
            n_base_models=n_base_models,
            input_size=input_size,
            hidden_size=self.config.HASTE_HIDDEN_SIZE,
            n_heads=4,
            dropout=self.config.HASTE_DROPOUT,
            device=self.device
        )
        
        # Create datasets
        if X_train_features is not None:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_stack),
                torch.FloatTensor(X_train_features),
                torch.FloatTensor(y_train),
                torch.FloatTensor(X_train_stack).unsqueeze(1)  # Temporal sequence
            )
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_stack),
                torch.FloatTensor(X_train_stack),  # Use stacking features as input features
                torch.FloatTensor(y_train),
                torch.FloatTensor(X_train_stack).unsqueeze(1)  # Temporal sequence
            )
        
        if X_val_features is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_stack),
                torch.FloatTensor(X_val_features),
                torch.FloatTensor(y_val),
                torch.FloatTensor(X_val_stack).unsqueeze(1)  # Temporal sequence
            )
        else:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_stack),
                torch.FloatTensor(X_val_stack),  # Use stacking features as input features
                torch.FloatTensor(y_val),
                torch.FloatTensor(X_val_stack).unsqueeze(1)  # Temporal sequence
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create trainer
        haste_trainer = HASTETrainer(
            model=self.haste_model,
            device=self.device,
            learning_rate=self.config.LEARNING_RATE
        )
        
        # Train model
        history = haste_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.NUM_EPOCHS,
            patience=self.config.PATIENCE,
            save_path='models/haste_best.pt'
        )
        
        self.train_history['haste'] = history
        
        print("\nHASTE model training completed!")
        
        return self.haste_model
    
    def train_ensemble(self, X_train_stack: np.ndarray, y_train: np.ndarray,
                      X_val_stack: np.ndarray, y_val: np.ndarray,
                      ensemble_method: str = 'stacking') -> Any:
        """Train traditional ensemble"""
        
        print(f"\nTraining {ensemble_method} ensemble...")
        
        from models.ensemble import AdvancedEnsemble
        
        # Create ensemble
        ensemble = AdvancedEnsemble(
            base_models=self.base_models,
            ensemble_method=ensemble_method,
            meta_model_type='logistic'
        )
        
        # Train ensemble
        if ensemble_method == 'dynamic_weighted':
            weights = ensemble.train_ensemble(X_train_stack, y_train, X_val_stack, y_val)
            self.ensemble_weights = weights
        else:
            ensemble_model = ensemble.train_ensemble(X_train_stack, y_train, X_val_stack, y_val)
            self.ensemble_model = ensemble_model
        
        # Save ensemble
        if ensemble_method != 'dynamic_weighted':
            joblib.dump(ensemble_model, f'models/ensemble_{ensemble_method}.joblib')
        
        return ensemble
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      model_type: str = 'haste', n_folds: int = 5) -> Dict[str, np.ndarray]:
        """Perform cross-validation"""
        
        print(f"\nPerforming {n_folds}-fold cross-validation for {model_type}...")
        
        from sklearn.model_selection import StratifiedKFold
        from evaluation.metrics import calculate_all_metrics
        
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                               random_state=self.config.RANDOM_STATE)
        
        fold_metrics = []
        all_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\n  Fold {fold + 1}/{n_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            if model_type == 'haste':
                # Train HASTE on this fold
                haste_model = self.train_haste(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold
                )
                
                # Predict
                # Note: This is simplified - in practice need to handle the stacking features
                predictions = np.zeros(len(y_val_fold))  # Placeholder
                
            elif model_type in ['rf', 'xgb', 'catboost', 'svm']:
                # Train single model
                from models.base_models import BaseModelFactory
                factory = BaseModelFactory(device=self.device, 
                                         random_state=self.config.RANDOM_STATE)
                model = factory.create_model(model_type)
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(X_val_fold)[:, 1]
                else:
                    predictions = model.predict(X_val_fold)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Calculate metrics
            metrics = calculate_all_metrics(y_val_fold, predictions)
            fold_metrics.append(metrics)
            
            # Store predictions
            all_predictions[val_idx] = predictions
            
            print(f"    Fold {fold + 1} F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in fold_metrics[0].keys():
            metric_values = [fold[metric_name] for fold in fold_metrics]
            avg_metrics[f'mean_{metric_name}'] = np.mean(metric_values)
            avg_metrics[f'std_{metric_name}'] = np.std(metric_values)
        
        print(f"\nCross-validation results for {model_type}:")
        print(f"  Average F1: {avg_metrics['mean_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
        print(f"  Average AUC: {avg_metrics['mean_roc_auc']:.4f} ± {avg_metrics['std_roc_auc']:.4f}")
        
        return avg_metrics, all_predictions
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray, 
                            model_type: str = 'haste') -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        print(f"\nHyperparameter tuning for {model_type}...")
        
        if model_type == 'haste':
            return self._tune_haste_hyperparameters(X_train, y_train, X_val, y_val)
        elif model_type == 'xgb':
            return self._tune_xgb_hyperparameters(X_train, y_train, X_val, y_val)
        elif model_type == 'rf':
            return self._tune_rf_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            print(f"Hyperparameter tuning not implemented for {model_type}")
            return {}
    
    def _tune_haste_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune HASTE hyperparameters"""
        
        import optuna
        
        def objective(trial):
            # Define hyperparameters to tune
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
            n_heads = trial.suggest_int('n_heads', 2, 8)
            lambda_div = trial.suggest_uniform('lambda_div', 0.0, 0.2)
            lambda_temp = trial.suggest_uniform('lambda_temp', 0.0, 0.1)
            
            # Create HASTE model with trial hyperparameters
            n_base_models = X_train.shape[1]  # Assuming X_train is stacking features
            
            from models.haste import HASTE, HASTETrainer
            
            haste_model = HASTE(
                n_base_models=n_base_models,
                input_size=n_base_models,  # Using stacking features as input
                hidden_size=hidden_size,
                n_heads=n_heads,
                dropout=dropout,
                device=self.device
            )
            
            # Create trainer
            trainer = HASTETrainer(
                model=haste_model,
                device=self.device,
                learning_rate=learning_rate
            )
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(X_train),  # Using stacking features as input features
                torch.FloatTensor(y_train),
                torch.FloatTensor(X_train).unsqueeze(1)
            )
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
                torch.FloatTensor(X_val).unsqueeze(1)
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            # Train for a few epochs
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50,  # Reduced for tuning
                patience=10,
                save_path='models/haste_tuning.pt'
            )
            
            # Return validation loss
            return min(history['val_loss'])
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE)
        )
        
        # Optimize
        study.optimize(objective, n_trials=20, timeout=3600)  # 20 trials or 1 hour
        
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        
        return study.best_params
    
    def _tune_xgb_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters"""
        
        import optuna
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 20),
                'random_state': self.config.RANDOM_STATE,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'tree_method': 'hist' if self.device == 'cpu' else 'gpu_hist'
            }
            
            model = XGBClassifier(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            
            return 1 - auc  # Minimize 1 - AUC
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def _tune_rf_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters"""
        
        import optuna
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': self.config.RANDOM_STATE,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            
            return 1 - auc
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        
        return study.best_params
    
    def train_full_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           X_train_features: Optional[np.ndarray] = None,
                           X_val_features: Optional[np.ndarray] = None,
                           X_test_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train full pipeline including base models, HASTE, and ensembles"""
        
        print("\n" + "="*60)
        print("FULL TRAINING PIPELINE")
        print("="*60)
        
        results = {}
        
        # 1. Train base models
        base_models, X_train_stack, X_val_stack = self.train_base_models(
            X_train, y_train, X_val, y_val
        )
        results['base_models'] = base_models
        
        # 2. Train HASTE model
        haste_model = self.train_haste(
            X_train_stack, y_train,
            X_val_stack, y_val,
            X_train_features, X_val_features
        )
        results['haste_model'] = haste_model
        
        # 3. Train traditional ensembles
        ensemble_methods = ['stacking', 'dynamic_weighted']
        
        for method in ensemble_methods:
            print(f"\nTraining {method} ensemble...")
            ensemble = self.train_ensemble(
                X_train_stack, y_train,
                X_val_stack, y_val,
                ensemble_method=method
            )
            results[f'{method}_ensemble'] = ensemble
        
        # 4. Cross-validation for HASTE
        print("\nPerforming cross-validation for HASTE...")
        cv_metrics, cv_predictions = self.cross_validate(
            X_train_stack, y_train,
            model_type='haste',
            n_folds=self.config.CV_FOLDS
        )
        results['haste_cv_metrics'] = cv_metrics
        results['haste_cv_predictions'] = cv_predictions
        
        # 5. Hyperparameter tuning for HASTE
        print("\nPerforming hyperparameter tuning for HASTE...")
        best_params = self.hyperparameter_tuning(
            X_train_stack, y_train,
            X_val_stack, y_val,
            model_type='haste'
        )
        results['haste_best_params'] = best_params
        
        # 6. Retrain HASTE with best hyperparameters
        if best_params:
            print("\nRetraining HASTE with best hyperparameters...")
            # This would retrain with best params - implementation depends on your needs
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETE")
        print("="*60)
        
        return results