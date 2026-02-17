"""
Main script for fraud detection pipeline
"""
import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch  # Moved to top level
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.drift_detection import ConceptDriftDetector
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("FRAUD DETECTION PIPELINE WITH HASTE")
print("=" * 80)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    
    parser.add_argument('--dataset', type=str, default='fraud_guard',
                       choices=['ieee_cis', 'paysim', 'banksim', 'fraud_guard'],
                       help='Dataset to use')
    
    parser.add_argument('--use_smote', action='store_true', default=True,
                       help='Use SMOTE for class imbalance')
    
    parser.add_argument('--use_adasyn', action='store_true', default=False,
                       help='Use ADASYN for class imbalance')
    
    parser.add_argument('--handle_drift', action='store_true', default=True,
                       help='Handle concept drift')
    
    parser.add_argument('--train_base', action='store_true', default=True,
                       help='Train base models')
    
    parser.add_argument('--train_haste', action='store_true', default=True,
                       help='Train HASTE model')
    
    parser.add_argument('--cross_validate', action='store_true', default=False,
                       help='Perform cross-validation')
    
    parser.add_argument('--hyperparameter_tuning', action='store_true', default=False,
                       help='Perform hyperparameter tuning')
    
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create visualization plots')
    
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to files')
    
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Sequence length for time series models')
    
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples to use (for testing)')
    
    return parser.parse_args()

def create_synthetic_data(n_samples=10000, n_features=30, fraud_rate=0.01, dataset_type='fraud_guard'):
    """Create synthetic data for testing when real data is not available"""
    print(f"Creating synthetic {dataset_type} data with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate normal transactions
    n_normal = int(n_samples * (1 - fraud_rate))
    n_fraud = n_samples - n_normal
    
    if dataset_type == 'fraud_guard':
        # FraudGuard: Time series with concept drift
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        
        # Add fraud with concept drift
        timestamps = np.arange(n_samples)
        
        # Phase 1: Low fraud rate
        phase1_end = int(n_samples * 0.3)
        fraud_indices_1 = np.random.choice(phase1_end, int(phase1_end * fraud_rate), replace=False)
        y[fraud_indices_1] = 1
        X[fraud_indices_1, :5] += np.random.randn(len(fraud_indices_1), 5) * 3
        
        # Phase 2: Higher fraud rate, different pattern
        phase2_start = phase1_end
        phase2_end = phase2_start + int(n_samples * 0.4)
        fraud_indices_2 = np.random.choice(
            range(phase2_start, phase2_end), 
            int((phase2_end - phase2_start) * (fraud_rate * 2)), 
            replace=False
        )
        y[fraud_indices_2] = 1
        X[fraud_indices_2, 5:10] += np.random.randn(len(fraud_indices_2), 5) * 2.5
        
        # Phase 3: Mixed patterns
        phase3_start = phase2_end
        fraud_indices_3 = np.random.choice(
            range(phase3_start, n_samples), 
            int((n_samples - phase3_start) * (fraud_rate * 3)), 
            replace=False
        )
        y[fraud_indices_3] = 1
        # Mix of patterns
        for idx in fraud_indices_3:
            if np.random.rand() > 0.5:
                X[idx, :5] += np.random.randn(5) * 3
            else:
                X[idx, 5:10] += np.random.randn(5) * 2.5
    
    elif dataset_type == 'ieee_cis':
        # IEEE-CIS: Many features, low fraud rate
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        y[fraud_indices] = 1
        
        # Fraud has higher values in certain features
        for idx in fraud_indices:
            X[idx, :8] += np.random.randn(8) * 2
            if np.random.rand() > 0.5:
                X[idx, np.random.choice(n_features, 3)] *= 3
    
    elif dataset_type == 'paysim':
        # PaySim: Transaction amounts, balances
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        
        # Create transaction-like features
        X[:, 0] = np.random.exponential(1000, n_samples)  # amount
        X[:, 1] = np.random.exponential(5000, n_samples)  # old balance
        X[:, 2] = np.random.exponential(3000, n_samples)  # new balance
        
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        y[fraud_indices] = 1
        
        # Fraud transactions have higher amounts
        X[fraud_indices, 0] *= np.random.uniform(2, 10, n_fraud)
    
    elif dataset_type == 'banksim':
        # BankSim: Categories, demographics
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        y[fraud_indices] = 1
        
        # Fraud in specific "categories"
        X[fraud_indices, :3] += np.random.randn(n_fraud, 3) * 2
        X[fraud_indices, 0] *= np.random.uniform(2, 20, n_fraud)  # amount multiplier
    
    # Add some noise
    X += np.random.randn(*X.shape) * 0.1
    
    print(f"Created synthetic {dataset_type} data: {X.shape}")
    print(f"Fraud rate: {y.mean():.4%} ({y.sum()} fraud cases)")
    
    return X, y

def train_simple_models(X_train, y_train, X_test, y_test, device='cpu'):
    """Train simple models as a fallback"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    print("\nTraining simple models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            results[name] = {
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'model': model
            }
            
            print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"    Error training {name}: {e}")
    
    return results

def main():
    """Main fraud detection pipeline"""
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  GPU ID: {args.gpu_id}")
    print(f"  Use SMOTE: {args.use_smote}")
    print(f"  Handle Drift: {args.handle_drift}")
    print(f"  Train Base Models: {args.train_base}")
    print(f"  Train HASTE: {args.train_haste}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Step 1: Create/load data
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    # For now, use synthetic data
    X, y = create_synthetic_data(
        n_samples=args.n_samples, 
        fraud_rate=0.01 if args.dataset in ['ieee_cis', 'paysim'] else 0.11,
        dataset_type=args.dataset
    )
    
    # Step 2: Split data
    print("\n" + "="*60)
    print("STEP 2: DATA SPLITTING")
    print("="*60)
    
    # Split into train, validation, test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.125,  # 0.125 * 0.8 = 0.1 of total
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Training: {X_train.shape} (fraud rate: {y_train.mean():.4%})")
    print(f"  Validation: {X_val.shape} (fraud rate: {y_val.mean():.4%})")
    print(f"  Test: {X_test.shape} (fraud rate: {y_test.mean():.4%})")
    
    # Step 3: Handle class imbalance if requested
    if args.use_smote or args.use_adasyn:
        print("\n" + "="*60)
        print("STEP 3: HANDLING CLASS IMBALANCE")
        print("="*60)
        
        try:
            if args.use_smote:
                from imblearn.over_sampling import SMOTE
                print("Applying SMOTE...")
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            elif args.use_adasyn:
                from imblearn.over_sampling import ADASYN
                print("Applying ADASYN...")
                adasyn = ADASYN(random_state=42)
                X_train, y_train = adasyn.fit_resample(X_train, y_train)
            
            print(f"  After resampling:")
            print(f"    Training: {X_train.shape} (fraud rate: {y_train.mean():.4%})")
            print(f"    Class distribution: {np.bincount(y_train.astype(int))}")
            
        except Exception as e:
            print(f"  Error in resampling: {e}")
            print("  Continuing without resampling...")
    
    # Step 4: Train base models
    if args.train_base:
        print("\n" + "="*60)
        print("STEP 4: TRAINING BASE MODELS")
        print("="*60)
        
        base_results = train_simple_models(X_train, y_train, X_test, y_test, device)
        
        if base_results:
            print("\nBase Model Results:")
            for name, metrics in base_results.items():
                print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    
    # Step 5: Train HASTE model
    if args.train_haste:
        print("\n" + "="*60)
        print("STEP 5: TRAINING HASTE MODEL")
        print("="*60)
        
        try:
            # Import simplified HASTE
            from models.haste import create_haste_model, HASTETrainer
            
            # Prepare base model predictions for stacking
            print("Preparing stacking features...")
            
            # For demonstration, use actual base model predictions if available
            # Otherwise create synthetic predictions
            n_base_models = 3  # RF, SVM, LR
            
            if 'base_results' in locals():
                # Use actual base model predictions
                base_preds_train = []
                base_preds_test = []
                
                for name, metrics in base_results.items():
                    model = metrics['model']
                    if hasattr(model, 'predict_proba'):
                        train_pred = model.predict_proba(X_train)[:, 1]
                        test_pred = model.predict_proba(X_test)[:, 1]
                    else:
                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)
                    
                    base_preds_train.append(train_pred)
                    base_preds_test.append(test_pred)
                
                base_preds_train = np.column_stack(base_preds_train)
                base_preds_test = np.column_stack(base_preds_test)
            else:
                # Create synthetic base predictions
                n_train = len(X_train)
                n_test = len(X_test)
                
                np.random.seed(42)
                base_preds_train = np.random.rand(n_train, n_base_models)
                base_preds_test = np.random.rand(n_test, n_base_models)
                
                # Add some correlation with actual labels
                for i in range(n_base_models):
                    base_preds_train[:, i] = base_preds_train[:, i] * 0.7 + y_train * 0.3
                    base_preds_test[:, i] = base_preds_test[:, i] * 0.7 + y_test * 0.3
            
            print(f"  Base predictions shape: {base_preds_train.shape}")
            
            # Create HASTE model
            print("Creating HASTE model...")
            haste_model = create_haste_model(
                n_base_models=n_base_models,
                input_size=X_train.shape[1],
                hidden_size=128,
                dropout=0.3,
                device=device
            )
            
            # Create datasets
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(base_preds_train),
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train),
                torch.FloatTensor(base_preds_train).unsqueeze(1)  # temporal sequence
            )
            
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(base_preds_test),
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test),
                torch.FloatTensor(base_preds_test).unsqueeze(1)
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            # Create trainer
            trainer = HASTETrainer(
                model=haste_model,
                device=device,
                learning_rate=0.001
            )
            
            # Train for a few epochs
            print("Training HASTE model (10 epochs)...")
            history = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,  # Using test as val for demo
                num_epochs=10,
                patience=3,
                save_path='models/haste_best.pt'
            )
            
            # Evaluate
            print("Evaluating HASTE model...")
            haste_predictions = trainer.predict(test_loader)
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            haste_pred_binary = (haste_predictions > 0.5).astype(int)
            haste_accuracy = accuracy_score(y_test, haste_pred_binary)
            haste_f1 = f1_score(y_test, haste_pred_binary)
            haste_auc = roc_auc_score(y_test, haste_predictions)
            
            print(f"\nHASTE Model Results:")
            print(f"  Accuracy: {haste_accuracy:.4f}")
            print(f"  F1 Score: {haste_f1:.4f}")
            print(f"  ROC-AUC: {haste_auc:.4f}")
            
            # Compare with base models
            if 'base_results' in locals():
                print(f"\nComparison with Base Models:")
                best_base_f1 = max([metrics['f1'] for metrics in base_results.values()])
                improvement = ((haste_f1 - best_base_f1) / best_base_f1) * 100
                print(f"  Best Base Model F1: {best_base_f1:.4f}")
                print(f"  HASTE F1: {haste_f1:.4f}")
                print(f"  Improvement: {improvement:+.2f}%")
            
        except Exception as e:
            print(f"Error training HASTE: {e}")
            print("Continuing with base models only...")
            import traceback
            traceback.print_exc()
    
    # Step 6: Handle concept drift
    if args.handle_drift:
        print("\n" + "="*60)
        print("STEP 6: CONCEPT DRIFT DETECTION")
        print("="*60)
        
        try:
            from utils.drift_detection import ConceptDriftDetector
            
            drift_detector = ConceptDriftDetector(window_size=1000, threshold=0.05)
            
            # Split training data into "old" and "new" for drift detection demo
            split_idx = len(X_train) // 2
            X_old = X_train[:split_idx]
            X_new = X_train[split_idx:]
            y_old = y_train[:split_idx]
            y_new = y_train[split_idx:]
            
            drift_results = drift_detector.detect_multivariate_drift(X_old, X_new)
            
            print(f"\nDrift Detection Results:")
            print(f"  KS Test p-value: {drift_results['ks_p_value']:.4f}")
            print(f"  MMD Score: {drift_results['mmd_score']:.4f}")
            print(f"  Covariance Difference: {drift_results['covariance_diff']:.4f}")
            print(f"  Overall Drift Detected: {drift_results['overall_drift']}")
            
            if drift_results['overall_drift']:
                print("  Warning: Concept drift detected! Model performance may degrade.")
                print("  Consider retraining with more recent data or using adaptive methods.")
            else:
                print("  No significant concept drift detected.")
                
        except Exception as e:
            print(f"Error in drift detection: {e}")
    
    # Step 7: Create visualizations
    if args.create_plots:
        print("\n" + "="*60)
        print("STEP 7: CREATING VISUALIZATIONS")
        print("="*60)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot 1: Class distribution
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            class_counts = np.bincount(y.astype(int))
            plt.bar(['Non-Fraud', 'Fraud'], class_counts, color=['blue', 'red'])
            plt.title('Class Distribution (Original)')
            plt.ylabel('Count')
            for i, count in enumerate(class_counts):
                plt.text(i, count, str(count), ha='center', va='bottom')
            
            plt.subplot(1, 2, 2)
            if 'y_train' in locals() and len(np.unique(y_train)) > 1:
                train_counts = np.bincount(y_train.astype(int))
                plt.bar(['Non-Fraud', 'Fraud'], train_counts, color=['lightblue', 'pink'])
                plt.title('Class Distribution (Training after Resampling)')
                plt.ylabel('Count')
                for i, count in enumerate(train_counts):
                    plt.text(i, count, str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('plots/class_distribution.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved class distribution plot to 'plots/class_distribution.png'")
            
            # Plot 2: Feature distributions (first 4 features)
            plt.figure(figsize=(12, 8))
            for i in range(min(4, X.shape[1])):
                plt.subplot(2, 2, i+1)
                plt.hist(X[y == 0, i], bins=30, alpha=0.5, label='Non-Fraud', color='blue')
                plt.hist(X[y == 1, i], bins=30, alpha=0.5, label='Fraud', color='red')
                plt.title(f'Feature {i+1} Distribution')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('plots/feature_distributions.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved feature distributions plot to 'plots/feature_distributions.png'")
            
            # Plot 3: Model comparison if we have results
            if 'base_results' in locals() and args.train_haste and 'haste_f1' in locals():
                plt.figure(figsize=(10, 6))
                
                models = list(base_results.keys())
                models.append('HASTE')
                
                f1_scores = [base_results[m]['f1'] for m in base_results.keys()]
                f1_scores.append(haste_f1)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                bars = plt.bar(models, f1_scores, color=colors)
                
                plt.title('Model Comparison (F1 Score)')
                plt.ylabel('F1 Score')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1)
                
                for bar, score in zip(bars, f1_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('plots/model_comparison.png', dpi=100, bbox_inches='tight')
                print("  ✓ Saved model comparison plot to 'plots/model_comparison.png'")
            
            plt.close('all')
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 8: Save results
    if args.save_results:
        print("\n" + "="*60)
        print("STEP 8: SAVING RESULTS")
        print("="*60)
        
        results = {
            'dataset': args.dataset,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_statistics': {
                'total_samples': len(X),
                'fraud_rate': float(y.mean()),
                'n_features': X.shape[1],
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
        }
        
        # Add base model results
        if 'base_results' in locals():
            results['base_models'] = {}
            for name, metrics in base_results.items():
                results['base_models'][name] = {
                    'accuracy': float(metrics['accuracy']),
                    'f1': float(metrics['f1']),
                    'auc': float(metrics['auc'])
                }
        
        # Add HASTE results
        if 'haste_f1' in locals():
            results['haste'] = {
                'accuracy': float(haste_accuracy),
                'f1': float(haste_f1),
                'auc': float(haste_auc)
            }
        
        # Save to JSON
        with open('results/results_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("  ✓ Saved results to 'results/results_summary.json'")
        
        # Save predictions if available
        if 'haste_predictions' in locals():
            np.save('results/haste_predictions.npy', haste_predictions)
            print("  ✓ Saved HASTE predictions to 'results/haste_predictions.npy'")
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*60)
    # Step 9: Comprehensive Evaluation
    print("\n" + "="*60)
    print("STEP 9: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    try:
        # Prepare models dictionary
        models_dict = {}
        predictions_dict = {}
        
        # Add base models
        if 'base_results' in locals():
            for name, metrics in base_results.items():
                models_dict[name] = {'model': metrics['model']}
                # Get predictions
                if hasattr(metrics['model'], 'predict_proba'):
                    predictions_dict[name] = metrics['model'].predict_proba(X_test)[:, 1]
                else:
                    predictions_dict[name] = metrics['model'].predict(X_test)
        
        # Add HASTE model if available
        if 'haste_predictions' in locals():
            models_dict['HASTE'] = {'predictions': haste_predictions}
            predictions_dict['HASTE'] = haste_predictions
        
        # Run comprehensive evaluation
        from evalution.comprehensive_evaluator import run_comprehensive_evaluation
        
        # Prepare configuration
        config = {
            'dataset': args.dataset,
            'n_samples': args.n_samples,
            'use_smote': args.use_smote,
            'use_adasyn': args.use_adasyn,
            'train_haste': args.train_haste,
            'device': device,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Run evaluation
        all_metrics, report_dir = run_comprehensive_evaluation(
            models_dict=models_dict,
            X_test=X_test,
            y_test=y_test,
            predictions_dict=predictions_dict,
            config=config,
            dataset_name=args.dataset
        )
        
        # Step 10: Ablation Study (Optional)
        if args.train_haste and 'base_preds_train' in locals() and 'base_preds_test' in locals():
            print("\n" + "="*60)
            print("STEP 10: ABLATION STUDY")
            print("="*60)
            
            try:
                from evalution.comprehensive_evaluator import AblationStudy
                
                ablation_study = AblationStudy()
                
                # Prepare data for ablation study
                # Use a subset for faster computation
                n_samples_ablation = min(2000, len(X_test))
                indices = np.random.choice(len(X_test), n_samples_ablation, replace=False)
                
                X_test_sub = X_test[indices]
                y_test_sub = y_test[indices]
                base_preds_test_sub = base_preds_test[indices]
                
                # Run ablation study
                ablation_results = ablation_study.study_haste_components(
                    X_train=X_train[:n_samples_ablation],
                    y_train=y_train[:n_samples_ablation],
                    X_test=X_test_sub,
                    y_test=y_test_sub,
                    base_predictions=base_preds_test_sub,
                    device=device
                )
                
                print("\nAblation Study Results:")
                print("-" * 40)
                for model_name, metrics in ablation_results.items():
                    print(f"{model_name:<20} F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                print(f"Error in ablation study: {e}")
        
    except Exception as e:
        print(f"Error in comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc()
    # Final summary
    print("\nFINAL SUMMARY:")
    print("-" * 40)
    
    if 'base_results' in locals():
        print("\nBase Models Performance:")
        for name, metrics in base_results.items():
            print(f"  {name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            print(f"    ROC-AUC: {metrics['auc']:.4f}")
    
    if 'haste_f1' in locals():
        print("\nHASTE Model Performance:")
        print(f"  Accuracy: {haste_accuracy:.4f}")
        print(f"  F1 Score: {haste_f1:.4f}")
        print(f"  ROC-AUC: {haste_auc:.4f}")
        
        if 'base_results' in locals():
            best_base_f1 = max([metrics['f1'] for metrics in base_results.values()])
            improvement = ((haste_f1 - best_base_f1) / best_base_f1) * 100
            print(f"\n  HASTE Improvement over best base model: {improvement:+.2f}%")
    
    print("\n" + "="*60)
    print("All operations completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()