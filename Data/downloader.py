"""
Data downloader for benchmark fraud datasets
"""
import pandas as pd
import numpy as np
import kagglehub
import zipfile
import os
from pathlib import Path
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FraudDataDownloader:
    """Download and prepare benchmark fraud datasets"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_ieee_cis(self):
        """Download IEEE-CIS Fraud Detection dataset"""
        print("Downloading IEEE-CIS dataset...")
        try:
            # Download from Kaggle
            path = kagglehub.dataset_download("c/ieee-fraud-detection")
            print(f"Dataset downloaded to: {path}")
            
            # Load data
            train_transaction = pd.read_csv(f"{path}/train_transaction.csv")
            train_identity = pd.read_csv(f"{path}/train_identity.csv")
            
            # Merge datasets
            df = pd.merge(train_transaction, train_identity, 
                         on='TransactionID', how='left')
            
            # Save processed data
            save_path = self.data_dir / 'ieee_cis' / 'processed'
            save_path.mkdir(exist_ok=True, parents=True)
            df.to_parquet(save_path / 'ieee_cis_processed.parquet', index=False)
            print(f"Saved processed data to {save_path}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading IEEE-CIS: {e}")
            print("Using synthetic data as fallback...")
            return self._create_synthetic_ieee()
    
    def download_paysim(self):
        """Download PaySim Mobile Money Simulator dataset"""
        print("Downloading PaySim dataset...")
        try:
            url = "https://www.kaggle.com/datasets/ealaxi/paysim1/download"
            # Alternative source
            url = "https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/PS_20174392719_1491204439457_log.csv"
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            save_path = self.data_dir / 'paysim' / 'raw'
            save_path.mkdir(exist_ok=True, parents=True)
            file_path = save_path / 'paysim.csv'
            
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            df = pd.read_csv(file_path)
            
            # Save processed
            processed_path = self.data_dir / 'paysim' / 'processed'
            processed_path.mkdir(exist_ok=True, parents=True)
            df.to_parquet(processed_path / 'paysim_processed.parquet', index=False)
            
            return df
            
        except Exception as e:
            print(f"Error downloading PaySim: {e}")
            return self._create_synthetic_paysim()
    
    def download_banksim(self):
        """Download BankSim dataset"""
        print("Downloading BankSim dataset...")
        try:
            url = "https://raw.githubusercontent.com/akmand/datasets/master/banksim.csv"
            df = pd.read_csv(url)
            
            save_path = self.data_dir / 'banksim' / 'processed'
            save_path.mkdir(exist_ok=True, parents=True)
            df.to_parquet(save_path / 'banksim_processed.parquet', index=False)
            
            return df
            
        except Exception as e:
            print(f"Error downloading BankSim: {e}")
            return self._create_synthetic_banksim()
    
    def download_fraud_guard(self):
        """Download FraudGuard 2025 synthetic dataset"""
        print("Downloading FraudGuard 2025 dataset...")
        try:
            # This is a hypothetical dataset - create synthetic version
            return self._create_fraud_guard_2025()
        except Exception as e:
            print(f"Error: {e}")
            return self._create_fraud_guard_2025()
    
    def _create_synthetic_ieee(self):
        """Create synthetic IEEE-CIS like data"""
        np.random.seed(42)
        n_samples = 100000
        n_features = 300
        
        print(f"Creating synthetic IEEE-CIS data with {n_samples} samples...")
        
        # Create feature names similar to IEEE-CIS
        features = []
        for i in range(1, 14):
            features.extend([f'V{i}_{j}' for j in range(1, 21)])
        features = features[:n_features]
        
        # Generate data with fraud patterns
        X = np.random.randn(n_samples, n_features)
        
        # Add fraud patterns (1.5% fraud rate)
        n_fraud = int(n_samples * 0.015)
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        
        # Fraud patterns in specific features
        fraud_features = [f'V{i}_1' for i in [1, 2, 3, 4, 7, 9, 10, 11, 12, 13]]
        fraud_feature_idx = [features.index(f) for f in fraud_features if f in features]
        
        for idx in fraud_indices:
            # Increase values in fraud features
            X[idx, fraud_feature_idx] += np.random.randn(len(fraud_feature_idx)) * 3
            # Add anomalies
            anomaly_features = np.random.choice(n_features, 5, replace=False)
            X[idx, anomaly_features] *= np.random.uniform(2, 5, 5)
        
        # Create target
        y = np.zeros(n_samples)
        y[fraud_indices] = 1
        
        # Create transaction amounts
        amounts = np.random.exponential(100, n_samples)
        amounts[fraud_indices] *= np.random.uniform(2, 10, n_fraud)
        
        # Create DataFrame
        data = {f: X[:, i] for i, f in enumerate(features)}
        data['TransactionAmt'] = amounts
        data['isFraud'] = y
        data['TransactionID'] = range(n_samples)
        
        df = pd.DataFrame(data)
        
        save_path = self.data_dir / 'ieee_cis' / 'processed'
        save_path.mkdir(exist_ok=True, parents=True)
        df.to_parquet(save_path / 'ieee_cis_synthetic.parquet', index=False)
        
        return df
    
    def _create_synthetic_paysim(self):
        """Create synthetic PaySim like data"""
        np.random.seed(42)
        n_samples = 100000
        
        print(f"Creating synthetic PaySim data with {n_samples} samples...")
        
        data = {
            'step': np.random.randint(1, 744, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], 
                                    n_samples, p=[0.2, 0.2, 0.1, 0.3, 0.2]),
            'amount': np.random.exponential(1000, n_samples),
            'nameOrig': ['C' + str(i).zfill(6) for i in range(100000, 100000 + n_samples)],
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.zeros(n_samples),
            'nameDest': ['M' + str(i).zfill(6) for i in range(200000, 200000 + n_samples)],
            'oldbalanceDest': np.random.exponential(3000, n_samples),
            'newbalanceDest': np.zeros(n_samples),
            'isFraud': np.zeros(n_samples),
            'isFlaggedFraud': np.zeros(n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add fraud patterns (0.1% fraud rate for PaySim)
        n_fraud = int(n_samples * 0.001)
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        
        df.loc[fraud_indices, 'isFraud'] = 1
        df.loc[fraud_indices, 'type'] = 'TRANSFER'
        df.loc[fraud_indices, 'amount'] *= np.random.uniform(10, 100, n_fraud)
        
        # Calculate balances
        df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
        
        save_path = self.data_dir / 'paysim' / 'processed'
        save_path.mkdir(exist_ok=True, parents=True)
        df.to_parquet(save_path / 'paysim_synthetic.parquet', index=False)
        
        return df
    
    def _create_synthetic_banksim(self):
        """Create synthetic BankSim data"""
        np.random.seed(42)
        n_samples = 80000
        
        print(f"Creating synthetic BankSim data with {n_samples} samples...")
        
        categories = ['es_transportation', 'es_health', 'es_leisure', 
                     'es_hotelservices', 'es_otherservices', 'es_food', 
                     'es_home', 'es_hyper', 'es_wellnessandbeauty', 
                     'es_tech', 'es_sportsandtoys', 'es_fashion', 
                     'es_travel', 'es_contents']
        
        data = {
            'step': np.random.randint(1, 180, n_samples),
            'customer': ['C' + str(i) for i in range(n_samples)],
            'age': np.random.choice(['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+'], 
                                   n_samples, p=[0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.1]),
            'gender': np.random.choice(['M', 'F', 'E', 'U'], n_samples, p=[0.45, 0.45, 0.05, 0.05]),
            'zipcodeOri': np.random.randint(1000, 10000, n_samples),
            'merchant': ['M' + str(i) for i in range(5000, 5000 + n_samples)],
            'zipMerchant': np.random.randint(1000, 10000, n_samples),
            'category': np.random.choice(categories, n_samples),
            'amount': np.random.exponential(50, n_samples),
            'fraud': np.zeros(n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add fraud (11% fraud rate for BankSim)
        n_fraud = int(n_samples * 0.11)
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        
        df.loc[fraud_indices, 'fraud'] = 1
        df.loc[fraud_indices, 'amount'] *= np.random.uniform(2, 20, n_fraud)
        
        # Fraud concentrated in specific categories
        fraud_categories = ['es_transportation', 'es_tech', 'es_travel']
        df.loc[fraud_indices, 'category'] = np.random.choice(fraud_categories, n_fraud)
        
        save_path = self.data_dir / 'banksim' / 'processed'
        save_path.mkdir(exist_ok=True, parents=True)
        df.to_parquet(save_path / 'banksim_synthetic.parquet', index=False)
        
        return df
    
    def _create_fraud_guard_2025(self):
        """Create FraudGuard 2025 synthetic dataset with concept drift"""
        np.random.seed(42)
        n_samples = 150000
        n_features = 50
        
        print(f"Creating FraudGuard 2025 data with {n_samples} samples...")
        
        # Generate base features
        X = np.random.randn(n_samples, n_features)
        
        # Create time-based concept drift
        timestamps = pd.date_range('2025-01-01', periods=n_samples, freq='T')
        
        # Phase 1: Normal behavior (first 30%)
        phase1_end = int(n_samples * 0.3)
        
        # Phase 2: New fraud pattern emerges (next 40%)
        phase2_start = phase1_end
        phase2_end = phase2_start + int(n_samples * 0.4)
        
        # Phase 3: Both patterns active (last 30%)
        phase3_start = phase2_end
        
        # Create fraud labels with concept drift
        y = np.zeros(n_samples)
        
        # Phase 1 fraud (1% fraud rate)
        n_fraud_phase1 = int(phase1_end * 0.01)
        fraud_idx_phase1 = np.random.choice(phase1_end, n_fraud_phase1, replace=False)
        y[fraud_idx_phase1] = 1
        
        # Fraud pattern 1: Features 0-9 are important
        for idx in fraud_idx_phase1:
            X[idx, :10] += np.random.randn(10) * 2
        
        # Phase 2 fraud (2% fraud rate, new pattern)
        n_fraud_phase2 = int((phase2_end - phase2_start) * 0.02)
        fraud_idx_phase2 = np.random.choice(
            range(phase2_start, phase2_end), n_fraud_phase2, replace=False
        )
        y[fraud_idx_phase2] = 1
        
        # Fraud pattern 2: Features 10-19 are important
        for idx in fraud_idx_phase2:
            X[idx, 10:20] += np.random.randn(10) * 2.5
        
        # Phase 3 fraud (3% fraud rate, mixed patterns)
        n_fraud_phase3 = int((n_samples - phase3_start) * 0.03)
        fraud_idx_phase3 = np.random.choice(
            range(phase3_start, n_samples), n_fraud_phase3, replace=False
        )
        y[fraud_idx_phase3] = 1
        
        # Mixed patterns in phase 3
        for idx in fraud_idx_phase3:
            if np.random.rand() > 0.5:
                # Pattern 1
                X[idx, :10] += np.random.randn(10) * 2
            else:
                # Pattern 2
                X[idx, 10:20] += np.random.randn(10) * 2.5
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = {f: X[:, i] for i, f in enumerate(feature_names)}
        data['timestamp'] = timestamps
        data['amount'] = np.random.exponential(100, n_samples) * (1 + y * np.random.uniform(2, 5, n_samples))
        data['is_fraud'] = y
        data['phase'] = np.zeros(n_samples)
        data.loc[:phase1_end, 'phase'] = 1
        data.loc[phase2_start:phase2_end, 'phase'] = 2
        data.loc[phase3_start:, 'phase'] = 3
        
        df = pd.DataFrame(data)
        
        save_path = self.data_dir / 'fraud_guard' / 'processed'
        save_path.mkdir(exist_ok=True, parents=True)
        df.to_parquet(save_path / 'fraud_guard_2025.parquet', index=False)
        
        print(f"Created FraudGuard 2025 with concept drift across {len(df)} samples")
        print(f"Overall fraud rate: {y.mean():.3%}")
        print(f"Fraud rates by phase: Phase 1: {y[:phase1_end].mean():.3%}, "
              f"Phase 2: {y[phase2_start:phase2_end].mean():.3%}, "
              f"Phase 3: {y[phase3_start:].mean():.3%}")
        
        return df
    
    def load_dataset(self, dataset_name):
        """Load specified dataset"""
        dataset_methods = {
            'ieee_cis': self.download_ieee_cis,
            'paysim': self.download_paysim,
            'banksim': self.download_banksim,
            'fraud_guard': self.download_fraud_guard
        }
        
        if dataset_name not in dataset_methods:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Choose from: {list(dataset_methods.keys())}")
        
        return dataset_methods[dataset_name]()
    
    def load_all_datasets(self):
        """Load all benchmark datasets"""
        datasets = {}
        for name in ['ieee_cis', 'paysim', 'banksim', 'fraud_guard']:
            print(f"\n{'='*50}")
            print(f"Loading {name} dataset...")
            datasets[name] = self.load_dataset(name)
        
        return datasets