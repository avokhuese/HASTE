"""
Configuration file for fraud detection system
"""
import torch
import numpy as np
from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Dataset paths
    DATASETS = {
        'ieee_cis': DATA_DIR / 'ieee_cis',
        'paysim': DATA_DIR / 'paysim',
        'banksim': DATA_DIR / 'banksim',
        'fraud_guard': DATA_DIR / 'fraud_guard'
    }
    
    # GPU settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    
    # Training parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Model parameters
    BASE_MODELS = ['rf', 'xgb', 'catboost', 'svm', 'lightgbm', 'mlp']
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    PATIENCE = 15
    LEARNING_RATE = 1e-3
    
    # HASTE parameters
    HASTE_HIDDEN_SIZE = 128
    HASTE_NUM_LAYERS = 2
    HASTE_DROPOUT = 0.3
    
    # Imbalance handling
    USE_SMOTE = True
    USE_ADASYN = False
    SMOTE_RATIO = 0.3
    
    # Concept drift
    DRIFT_DETECTION_WINDOW = 1000
    DRIFT_ADAPTATION_RATE = 0.1
    
    # Cross-validation
    CV_FOLDS = 5
    CV_STRATEGY = 'stratified'
    
    # Metrics
    EVALUATION_METRICS = [
        'accuracy', 'precision', 'recall', 'f1', 
        'roc_auc', 'pr_auc', 'average_precision',
        'f2', 'matthews_corrcoef', 'cohen_kappa'
    ]
    
    # Visualization
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C5B7B']
    
    @classmethod
    def setup_gpu(cls):
        """Setup GPU configuration"""
        if cls.DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Using CPU")
    
    @classmethod
    def set_random_seeds(cls):
        """Set random seeds for reproducibility"""
        np.random.seed(cls.RANDOM_STATE)
        torch.manual_seed(cls.RANDOM_STATE)
        if cls.DEVICE == "cuda":
            torch.cuda.manual_seed_all(cls.RANDOM_STATE)

config = Config()