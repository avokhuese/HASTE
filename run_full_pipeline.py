"""
Run full fraud detection pipeline with comprehensive evaluation
"""
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run full pipeline"""
    
    parser = argparse.ArgumentParser(description='Run Full Fraud Detection Pipeline')
    parser.add_argument('--dataset', type=str, default='fraud_guard',
                       choices=['ieee_cis', 'paysim', 'banksim', 'fraud_guard'],
                       help='Dataset to use')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples')
    parser.add_argument('--run_ablation', action='store_true', default=True,
                       help='Run ablation study')
    parser.add_argument('--create_report', action='store_true', default=True,
                       help='Create comprehensive report')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FULL FRAUD DETECTION PIPELINE")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Import and run main pipeline
    from main import main as run_pipeline
    
    # Set up arguments for main pipeline
    sys.argv = [
        'main.py',
        '--dataset', args.dataset,
        '--n_samples', str(args.n_samples),
        '--train_base',
        '--train_haste',
        '--use_smote',
        '--handle_drift',
        '--create_plots',
        '--save_results'
    ]
    
    # Run the pipeline
    run_pipeline()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print where to find results
    print("\nRESULTS LOCATION:")
    print("-" * 40)
    print("1. Plots:              plots/")
    print("2. Models:             models/")
    print("3. Results:            results/")
    print("4. Comprehensive Report: results/report_*/")
    print("5. Ablation Study:     results/ablation_study/")
    print("\nTo view the results:")
    print("  - Check 'plots/' directory for visualizations")
    print("  - Check 'results/report_*/' for detailed metrics")
    print("  - Check 'results/ablation_study/' for component analysis")

if __name__ == "__main__":
    main()