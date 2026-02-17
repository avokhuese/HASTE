"""
Data preprocessing for fraud detection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
from typing import Tuple, Optional, Dict, Any
import gc

class FraudDataPreprocessor:
    """Preprocess fraud detection datasets"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
    
    def preprocess_ieee_cis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess IEEE-CIS dataset"""
        print("Preprocessing IEEE-CIS dataset...")
        
        # Separate features and target
        if 'isFraud' in df.columns:
            y = df['isFraud'].copy()
            X = df.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'isFraud' not found")
        
        # Handle missing values
        print(f"  Missing values: {X.isna().sum().sum()}")
        
        # Drop columns with too many missing values
        missing_threshold = 0.8
        cols_to_drop = [col for col in X.columns 
                       if X[col].isna().mean() > missing_threshold]
        X = X.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} columns with >{missing_threshold*100:.0f}% missing values")
        
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        
        # Impute missing values
        if len(numeric_cols) > 0:
            # For numeric columns, use median imputation
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        if len(categorical_cols) > 0:
            # For categorical columns, use mode imputation
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            
            # Encode categorical variables
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale features
        self.scaler = RobustScaler()  # Robust to outliers
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Remove low variance features
        var_threshold = 0.01
        selector = VarianceThreshold(threshold=var_threshold)
        X_selected = selector.fit_transform(X)
        selected_cols = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_cols)
        
        print(f"  Selected {X.shape[1]} features after variance threshold")
        
        # Add transaction amount features
        if 'TransactionAmt' in df.columns:
            X['TransactionAmt'] = df['TransactionAmt'].values
            X['TransactionAmt_log'] = np.log1p(df['TransactionAmt'].values)
            X['TransactionAmt_scaled'] = df['TransactionAmt'] / df['TransactionAmt'].mean()
        
        return X, y
    
    def preprocess_paysim(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess PaySim dataset"""
        print("Preprocessing PaySim dataset...")
        
        # Separate features and target
        if 'isFraud' in df.columns:
            y = df['isFraud'].copy()
            X = df.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], 
                       axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'isFraud' not found")
        
        # Encode transaction type
        if 'type' in X.columns:
            type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 
                           'PAYMENT': 3, 'TRANSFER': 4}
            X['type'] = X['type'].map(type_mapping)
            X['type'] = X['type'].fillna(-1)
        
        # Create balance difference features
        if all(col in X.columns for col in ['oldbalanceOrg', 'newbalanceOrig']):
            X['balance_diff_orig'] = X['oldbalanceOrg'] - X['newbalanceOrig']
            X['balance_diff_ratio_orig'] = np.where(
                X['oldbalanceOrg'] > 0,
                (X['oldbalanceOrg'] - X['newbalanceOrig']) / X['oldbalanceOrg'],
                0
            )
        
        if all(col in X.columns for col in ['oldbalanceDest', 'newbalanceDest']):
            X['balance_diff_dest'] = X['newbalanceDest'] - X['oldbalanceDest']
            X['balance_diff_ratio_dest'] = np.where(
                X['oldbalanceDest'] > 0,
                (X['newbalanceDest'] - X['oldbalanceDest']) / X['oldbalanceDest'],
                0
            )
        
        # Create transaction features
        X['amount_log'] = np.log1p(X['amount'])
        X['amount_to_balance_ratio'] = X['amount'] / (X['oldbalanceOrg'] + 1)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    
    def preprocess_banksim(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess BankSim dataset"""
        print("Preprocessing BankSim dataset...")
        
        # Separate features and target
        if 'fraud' in df.columns:
            y = df['fraud'].copy()
            X = df.drop(['fraud', 'customer', 'merchant'], 
                       axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'fraud' not found")
        
        # Encode categorical variables
        categorical_cols = ['age', 'gender', 'category']
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Create time-based features
        if 'step' in X.columns:
            X['hour_of_day'] = X['step'] % 24
            X['day_of_week'] = (X['step'] // 24) % 7
            X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        
        # Create amount features
        if 'amount' in X.columns:
            X['amount_log'] = np.log1p(X['amount'])
            X['amount_category'] = pd.cut(X['amount'], 
                                         bins=[0, 10, 50, 100, 500, np.inf],
                                         labels=[0, 1, 2, 3, 4])
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    
    def preprocess_fraud_guard(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess FraudGuard dataset"""
        print("Preprocessing FraudGuard dataset...")
        
        # Separate features and target
        if 'is_fraud' in df.columns:
            y = df['is_fraud'].copy()
            X = df.drop(['is_fraud', 'timestamp', 'phase'], 
                       axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'is_fraud' not found")
        
        # Create time-based features from timestamp if available
        if 'timestamp' in df.columns:
            X['hour'] = df['timestamp'].dt.hour
            X['day_of_week'] = df['timestamp'].dt.dayofweek
            X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
            X['month'] = df['timestamp'].dt.month
        
        # Create amount features
        if 'amount' in X.columns:
            X['amount_log'] = np.log1p(X['amount'])
            X['amount_zscore'] = (X['amount'] - X['amount'].mean()) / X['amount'].std()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        self.scaler = RobustScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    
    def detect_drift_features(self, X: pd.DataFrame, window_size: int = 1000) -> Dict:
        """Detect concept drift in features"""
        drift_metrics = {}
        
        for col in X.columns[:10]:  # Check first 10 features
            values = X[col].values
            n_windows = len(values) // window_size
            
            means = []
            stds = []
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                window_data = values[start:end]
                means.append(np.mean(window_data))
                stds.append(np.std(window_data))
            
            # Calculate drift metric (coefficient of variation of means)
            if np.mean(means) != 0:
                drift_metric = np.std(means) / np.mean(means)
                drift_metrics[col] = drift_metric
        
        return drift_metrics
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i + sequence_length])
            targets.append(y[i + sequence_length - 1])
        
        return np.array(sequences), np.array(targets)
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, 
                        method: str = 'smote', ratio: float = 0.3):
        """Handle class imbalance"""
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek
        
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=ratio, random_state=self.random_state)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=ratio, random_state=self.random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(sampling_strategy=ratio, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"  Resampled: {len(y)} -> {len(y_resampled)} samples")
        print(f"  Class distribution: {np.bincount(y_resampled.astype(int))}")
        
        return X_resampled, y_resampled
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str, 
                          handle_imbalance: bool = True, 
                          imbalance_method: str = 'smote') -> Tuple[np.ndarray, np.ndarray, dict]:
        """Main preprocessing pipeline"""
        
        preprocess_methods = {
            'ieee_cis': self.preprocess_ieee_cis,
            'paysim': self.preprocess_paysim,
            'banksim': self.preprocess_banksim,
            'fraud_guard': self.preprocess_fraud_guard
        }
        
        if dataset_name not in preprocess_methods:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Preprocess dataset
        X, y = preprocess_methods[dataset_name](df)
        
        # Convert to numpy
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        
        # Handle imbalance if requested
        if handle_imbalance and y_np.mean() < 0.3:  # Only if imbalance is severe
            print(f"\nHandling class imbalance using {imbalance_method}...")
            X_np, y_np = self.handle_imbalance(X_np, y_np, imbalance_method)
        
        # Detect concept drift
        drift_metrics = self.detect_drift_features(X)
        
        metadata = {
            'n_samples': X_np.shape[0],
            'n_features': X_np.shape[1],
            'fraud_rate': y_np.mean(),
            'feature_names': X.columns.tolist(),
            'drift_metrics': drift_metrics,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        print(f"\nPreprocessing complete:")
        print(f"  Samples: {X_np.shape[0]}")
        print(f"  Features: {X_np.shape[1]}")
        print(f"  Fraud rate: {y_np.mean():.4%}")
        
        return X_np, y_np, metadata