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

def run_script(script_name, args=None, timeout=None):
    """Run a Python script and capture its output."""
    script_path = os.path.join(src_dir, 'scripts', script_name)
    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        return False
    
    try:
        # Run the script with Python executable from current environment
        cmd = [sys.executable, script_path]
        
        # Add any additional arguments
        if args:
            cmd.extend(args)
        
        # Log the full command being run
        logging.info(f"Running command: {' '.join(cmd)}")
        
        # Use timeout in subprocess call, but don't add it as an argument to the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout  # Use timeout for subprocess management, not as a script arg
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
    except subprocess.TimeoutExpired as e:
        logging.error(f"Timeout expired running {script_name} after {timeout} seconds")
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
            "train_all", "test", "test_all", "compare"
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
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success = run_script(script_mapping["mlp_basic"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("MLP Basic model training completed")
    
    elif args.mode == "train_mlp_enhanced":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success = run_script(script_mapping["mlp_enhanced"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("MLP Enhanced model training completed")
    
    elif args.mode == "train_roberta":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success = run_script(script_mapping["roberta"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("RoBERTa model training completed")
    
    elif args.mode == "train_kernel":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success = run_script(script_mapping["kernel"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("Kernel Approximation model training completed")
    
    elif args.mode == "train_pca":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success = run_script(script_mapping["pca"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("Randomized PCA model training completed")
    
    elif args.mode == "train_lstm":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose),
            "--mode", "train"
        ]
        success = run_script(script_mapping["lstm"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("LSTM model training completed")
    
    elif args.mode == "train_lstm_roberta":
        # Common args for this mode
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose),
            "--mode", "train"
        ]
        success = run_script(script_mapping["lstm_roberta"], args=train_args, timeout=args.timeout)
        if success:
            logging.info("LSTM-RoBERTa model training completed")
    
    elif args.mode == "train_all":
        # Train models
        successes = {}
        
        # MLP Basic model
        train_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        success1 = run_script(script_mapping["mlp_basic"], args=train_args, timeout=args.timeout)
        successes["mlp_basic"] = success1
        
        # MLP Enhanced model
        success2 = run_script(script_mapping["mlp_enhanced"], args=train_args, timeout=args.timeout)
        successes["mlp_enhanced"] = success2
        
        # Kernel Approximation model
        success3 = run_script(script_mapping["kernel"], args=train_args, timeout=args.timeout)
        successes["kernel"] = success3
        
        # Randomized PCA model
        success4 = run_script(script_mapping["pca"], args=train_args, timeout=args.timeout)
        successes["pca"] = success4
        
        # LSTM and transformer models (optional due to higher resource requirements)
        if args.include_roberta or args.include_all:
            # Determine sample size for transformer models
            transformer_sample_size = min(args.sample_size, 20000) if args.sample_size > 20000 else args.sample_size
            transformer_args = [
                "--sample_size", str(transformer_sample_size),
                "--verbose", str(args.verbose)
            ]
            
            # LSTM models require mode parameter
            lstm_args = transformer_args + ["--mode", "train"]
            
            # Train additional models if requested
            if args.include_roberta or args.include_all:
                # Use longer timeout for transformer models
                transformer_timeout = max(args.timeout, 3600)  # At least 1 hour
                
                # RoBERTa model
                success5 = run_script(script_mapping["roberta"], args=transformer_args, timeout=transformer_timeout)
                successes["roberta"] = success5
                
                # LSTM model
                success6 = run_script(script_mapping["lstm"], args=lstm_args, timeout=transformer_timeout)
                successes["lstm"] = success6
                
                # LSTM-RoBERTa model
                success7 = run_script(script_mapping["lstm_roberta"], args=lstm_args, timeout=transformer_timeout)
                successes["lstm_roberta"] = success7
        
        # Count successful trainings
        successful_count = sum(1 for success in successes.values() if success)
        
        if successful_count > 0:
            logging.info(f"{successful_count} models trained successfully")
            
            # Run comparison
            compare_args = [
                "--sample_size", str(args.sample_size),
                "--timeout", str(args.timeout),
                "--verbose", str(args.verbose)
            ]
            
            if args.include_all:
                compare_args.append("--include_all")
            elif args.include_roberta:
                compare_args.append("--include_roberta")
                
            run_script("compare_models.py", args=compare_args)
        else:
            logging.error("All model trainings failed")

    elif args.mode == "test_all":
        # Test all existing models
        successes = {}
        
        # Common args for test mode
        test_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose)
        ]
        
        # Specific args for LSTM models (no verbose parameter)
        lstm_test_args = [
            "--sample_size", str(args.sample_size),
            "--mode", "test"
        ]
        
        # MLP Basic model
        success1 = run_script(script_mapping["mlp_basic"], args=test_args, timeout=args.timeout)
        successes["mlp_basic"] = success1
        
        # MLP Enhanced model
        success2 = run_script(script_mapping["mlp_enhanced"], args=test_args, timeout=args.timeout)
        successes["mlp_enhanced"] = success2
        
        # Kernel Approximation model
        success3 = run_script(script_mapping["kernel"], args=test_args, timeout=args.timeout)
        successes["kernel"] = success3
        
        # Randomized PCA model
        success4 = run_script(script_mapping["pca"], args=test_args, timeout=args.timeout)
        successes["pca"] = success4
        
        # LSTM and transformer models
        if args.include_roberta or args.include_all:
            # Determine timeout for transformer models
            transformer_timeout = max(args.timeout, 3600)  # At least 1 hour
            
            # RoBERTa model
            success5 = run_script(script_mapping["roberta"], args=test_args, timeout=transformer_timeout)
            successes["roberta"] = success5
            
            # LSTM model
            success6 = run_script(script_mapping["lstm"], args=lstm_test_args, timeout=transformer_timeout)
            successes["lstm"] = success6
            
            # LSTM-RoBERTa model
            success7 = run_script(script_mapping["lstm_roberta"], args=lstm_test_args, timeout=transformer_timeout)
            successes["lstm_roberta"] = success7
        
        # Count successful tests
        successful_count = sum(1 for success in successes.values() if success)
        
        if successful_count > 0:
            logging.info(f"{successful_count} models tested successfully")
            
            # Run comparison after testing
            compare_args = [
                "--sample_size", str(args.sample_size),
                "--timeout", str(args.timeout),
                "--verbose", str(args.verbose),
                "--skip_training"  # Skip training as we've just tested
            ]
            
            if args.include_all:
                compare_args.append("--include_all")
            elif args.include_roberta:
                compare_args.append("--include_roberta")
                
            run_script("compare_models.py", args=compare_args)
        else:
            logging.error("All model tests failed")

    elif args.mode == "test":
        # Test the specified model
        if not args.test_model:
            logging.error("No test model specified. Use --test_model to specify which model to test.")
            return
        
        # Common args for test mode
        test_args = [
            "--sample_size", str(args.sample_size)
        ]
        
        # Add verbose parameter for models that support it
        if args.test_model not in ["lstm", "lstm_roberta"]:
            test_args.extend(["--verbose", str(args.verbose)])
        
        # Add mode=test for LSTM models
        if args.test_model in ["lstm", "lstm_roberta"]:
            test_args.extend(["--mode", "test"])
        
        # Select script based on the model
        script = script_mapping.get(args.test_model)
        if not script:
            logging.error(f"Unknown model: {args.test_model}")
            return
        
        # Adjust timeout for transformer models
        model_timeout = args.timeout
        if args.test_model in ["roberta", "lstm_roberta"]:
            model_timeout = max(args.timeout, 3600)  # At least 1 hour for transformer models
        
        success = run_script(script, args=test_args, timeout=model_timeout)
        if success:
            logging.info(f"{args.test_model.capitalize()} model testing completed")
        else:
            logging.error(f"{args.test_model.capitalize()} model testing failed")

    elif args.mode == "compare":
        # Compare all models
        compare_args = [
            "--sample_size", str(args.sample_size),
            "--verbose", str(args.verbose),
            "--timeout", str(args.timeout)
        ]
        
        if args.include_all:
            compare_args.append("--include_all")
        elif args.include_roberta:
            compare_args.append("--include_roberta")
        
        if args.skip_training:
            compare_args.append("--skip_training")
        
        success = run_script("compare_models.py", args=compare_args)
        if success:
            logging.info("Model comparison completed")
        else:
            logging.error("Model comparison failed")
    
    else:
        logging.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main() 