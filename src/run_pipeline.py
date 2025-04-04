#!/usr/bin/env python
"""
Sentiment Analysis Pipeline Runner

This script provides a command-line interface to run different components
of the sentiment analysis pipeline.
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

from scripts.compare_models import print_model_comparison, save_report_to_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def run_script(script_name, timeout=None):
    """Run a Python script and capture its output."""
    script_path = os.path.join(src_dir, 'scripts', script_name)
    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        return False
    
    try:
        # Run the script with Python executable from current environment
        cmd = [sys.executable, script_path]
        if timeout:
            cmd.extend(['--timeout', str(timeout)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log the output
        if result.stdout:
            logging.info(f"Output from {script_name}:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Errors from {script_name}:\n{result.stderr}")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e}")
        if e.stdout:
            logging.error(f"Output: {e.stdout}")
        if e.stderr:
            logging.error(f"Errors: {e.stderr}")
        return False

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Pipeline Runner"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train_mlp_basic", "train_mlp_enhanced", "train_roberta",
            "train_kernel", "train_pca", "train_lstm", "train_lstm_roberta",
            "train_all", "test", "compare"
        ],
        required=True,
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--test_model",
        type=str,
        choices=["mlp_basic", "mlp_enhanced", "roberta", "kernel", "pca", "lstm", "lstm_roberta"],
        help="Model to test (required for test mode)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20000,
        help="Sample size to use for training (default: 20000)"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level for model training (0=silent, 1=progress bar, 2=one line per epoch)"
    )
    
    parser.add_argument(
        "--include_roberta",
        action="store_true",
        help="Include RoBERTa model in the comparison (for 'compare' mode)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,  # 30 minutes default
        help="Timeout in seconds for each model script"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training and just generate comparison from existing results (for 'compare' mode)"
    )
    
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="Include all models in the comparison (for 'compare' mode)"
    )
    
    args = parser.parse_args()
    
    # Common args for all scripts - excluding timeout
    common_args = [
        "--sample_size", str(args.sample_size),
        "--verbose", str(args.verbose)
    ]
    
    # Map of model names to their script files
    script_mapping = {
        "mlp_basic": "mlp_basic_main.py",
        "mlp_enhanced": "mlp_enhanced_main.py",
        "roberta": "roberta_main.py",
        "kernel": "kernel_approximation_main.py",
        "pca": "randomized_pca_main.py",
        "lstm": "lstm_main.py",
        "lstm_roberta": "lstm_roberta_main.py"
    }
    
    # Run the appropriate pipeline component
    if args.mode == "train_mlp_basic":
        success = run_script(script_mapping["mlp_basic"], timeout=args.timeout)
        if success:
            logging.info("MLP Basic model training completed")
    
    elif args.mode == "train_mlp_enhanced":
        success = run_script(script_mapping["mlp_enhanced"], timeout=args.timeout)
        if success:
            logging.info("MLP Enhanced model training completed")
    
    elif args.mode == "train_roberta":
        success = run_script(script_mapping["roberta"], timeout=args.timeout)
        if success:
            logging.info("RoBERTa model training completed")
    
    elif args.mode == "train_kernel":
        success = run_script(script_mapping["kernel"], timeout=args.timeout)
        if success:
            logging.info("Kernel Approximation model training completed")
    
    elif args.mode == "train_pca":
        success = run_script(script_mapping["pca"], timeout=args.timeout)
        if success:
            logging.info("Randomized PCA model training completed")
    
    elif args.mode == "train_lstm":
        success = run_script(script_mapping["lstm"], timeout=args.timeout)
        if success:
            logging.info("LSTM model training completed")
    
    elif args.mode == "train_lstm_roberta":
        success = run_script(script_mapping["lstm_roberta"], timeout=args.timeout)
        if success:
            logging.info("LSTM-RoBERTa model training completed")
    
    elif args.mode == "train_all":
        # Train models
        successes = {}
        
        # MLP Basic model
        success1 = run_script(script_mapping["mlp_basic"], timeout=args.timeout)
        successes["mlp_basic"] = success1
        
        # MLP Enhanced model
        success2 = run_script(script_mapping["mlp_enhanced"], timeout=args.timeout)
        successes["mlp_enhanced"] = success2
        
        # Kernel Approximation model
        success3 = run_script(script_mapping["kernel"], timeout=args.timeout)
        successes["kernel"] = success3
        
        # Randomized PCA model
        success4 = run_script(script_mapping["pca"], timeout=args.timeout)
        successes["pca"] = success4
        
        # LSTM model (optional due to higher resource requirements)
        if args.include_roberta or args.include_all:
            # Reduce sample size for LSTM to avoid excessive training time
            lstm_args = [
                "--sample_size", str(min(args.sample_size, 10000)),  # Cap at 10000 for LSTM
                "--verbose", str(args.verbose)
            ]
            
            # Train LSTM model 
            success5 = run_script(script_mapping["lstm"], timeout=args.timeout)
            successes["lstm"] = success5
        
        # LSTM-RoBERTa model (optional due to higher resource requirements)
        if args.include_roberta or args.include_all:
            # Reduce sample size for LSTM-RoBERTa to avoid excessive training time
            lstm_roberta_args = [
                "--sample_size", str(min(args.sample_size, 10000)),  # Cap at 10000 for LSTM-RoBERTa
                "--verbose", str(args.verbose)
            ]
            
            # Train LSTM-RoBERTa model 
            success6 = run_script(script_mapping["lstm_roberta"], timeout=args.timeout)
            successes["lstm_roberta"] = success6
        
        # Count successful trainings
        successful_count = sum(1 for success in successes.values() if success)
        
        if successful_count > 0:
            logging.info(f"{successful_count} models trained successfully")
            
            # Run comparison
            compare_args = ["--timeout", str(args.timeout)]
            
            if args.include_all or args.include_roberta and successes.get("roberta", False):
                compare_args.append("--include_roberta")
                
            run_script("compare_models.py", timeout=args.timeout)
        else:
            logging.error("All model trainings failed")

    elif args.mode == "test":
        # Test the specified model
        if args.test_model in script_mapping:
            test_script = script_mapping[args.test_model]
            success = run_script(test_script, timeout=args.timeout)
            if success:
                logging.info(f"Testing {args.test_model} model completed")
        else:
            logging.error(f"Unknown model: {args.test_model}")
    
    elif args.mode == "compare":
        compare_args = []
        if args.include_roberta:
            compare_args.append("--include_roberta")
        
        if args.include_all:
            compare_args.append("--include_all")
        
        compare_args.extend(["--sample_size", str(args.sample_size), 
                             "--verbose", str(args.verbose)])
        
        if args.skip_training:
            compare_args.append("--skip_training")
            
        run_script("compare_models.py", timeout=args.timeout)
    
    logging.info("Pipeline execution completed")

if __name__ == "__main__":
    main() 