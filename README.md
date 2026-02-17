# HASTE: Hierarchical Attention-based Stacked Ensemble for Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

HASTE (Hierarchical Attention-based Stacked Ensemble) is a novel deep learning architecture for fraud detection that combines multiple base models through a hierarchical attention mechanism with contextual and temporal awareness. The model dynamically learns to weight base model predictions based on input features, enabling it to prioritize high-confidence fraud predictions while adapting to concept drift through temporal attention.

This repository contains the complete implementation of HASTE along with comprehensive evaluation on four benchmark datasets:
- **BankSim**: POS transaction dataset with 11% fraud rate
- **FraudGuard 2025**: Synthetic dataset with concept drift across temporal phases
- **IEEE-CIS**: High-dimensional real-world fraud dataset with 1% fraud rate
- **PaySim**: Mobile money simulator with 1% fraud rate

## Key Features

- **Attention-based Ensemble**: Dynamically weights base model predictions based on input features
- **Temporal Attention**: Adapts to concept drift through temporal modeling
- **Hierarchical Architecture**: Multi-level attention for feature aggregation
- **Comprehensive Evaluation**: Statistical metrics, significance testing, and business impact analysis
- **GPU Support**: Optimized for CUDA-enabled training
- **Visualization Suite**: ROC curves, PR curves, confusion matrices, cost-benefit analysis, and attention visualization

## Requirements

Python 3.8+
PyTorch 2.0+
scikit-learn 1.0+
imbalanced-learn 0.9+
numpy 1.21+
pandas 1.3+
matplotlib 3.5+
seaborn 0.11+
scipy 1.7+

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Project Structure
fraud_detection_haste/
├── main.py                    # Main execution script
├── run_full_pipeline.py       # Complete pipeline runner
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
├── models/
│   └── haste.py                # HASTE model implementation
├── evaluation/
│   └── comprehensive_evaluator.py  # Evaluation and visualization
├── utils/
│   └── drift_detection.py      # Concept drift utilities
├── data/                        # Dataset storage
├── plots/                        # Generated visualizations
├── results/                      # Evaluation results
└── models/                        # Saved model checkpoints

# Run Complete Pipeline
python run_full_pipeline.py --dataset banksim --n_samples 10000

# Run with Custom Configuration
python main.py \
    --dataset fraud_guard \
    --n_samples 5000 \
    --train_base \
    --train_haste \
    --use_smote \
    --handle_drift \
    --create_plots \
    --save_results

# Available Datasets
1. banksim: BankSim POS transaction dataset

2. fraud_guard: Synthetic dataset with concept drift

3. ieee_cis: IEEE-CIS fraud detection dataset

4. paysim: PaySim mobile money simulator

# Key Arguments

Argument	Description	Default
--dataset	Dataset to use	fraud_guard
--n_samples	Number of samples	10000
--use_smote	Apply SMOTE oversampling	True
--train_base	Train base models	True
--train_haste	Train HASTE model	True
--handle_drift	Detect and handle concept drift	True
--create_plots	Generate visualizations	True
--save_results	Save results to disk	True
--gpu_id	GPU device ID	0


## Model Architecture
HASTE consists of four key components:

1. Hierarchical Attention: Multi-level attention for feature aggregation

2. Contextual Attention: Dynamic weighting of base model predictions based on input features

3. Temporal Attention: GRU-based attention for handling concept drift

4. Final Classifier: Combines attended predictions with extracted features

## Evaluation Metrics
The comprehensive evaluator calculates over 20 metrics:

1. Statistical: Accuracy, Precision, Recall, F1, F2, MCC, Cohen's Kappa

2. Probabilistic: ROC-AUC, PR-AUC, Log Loss, Brier Score

3. Confusion Matrix: TP, FP, FN, TN, TPR, TNR, FPR, FNR

4. Business: Net Benefit, ROI, Cost Savings (fraud cost = $1000, false alarm cost = $50)

5. Threshold Analysis: Optimal threshold via Youden's J statistic



## Citation
If you use HASTE in your research, please cite:

@article{haste2025,
  title={HASTE: Hierarchical Attention-based Stacked Ensemble for Fraud Detection},
  author={Alexander Okhuese Victor},
  journal={arXiv preprint},
  year={2025}
}

# License
This project is licensed under the MIT License 

# Acknowledgments
1. IEEE-CIS Fraud Detection dataset

2. PaySim mobile money simulator

3. BankSim POS transaction dataset
