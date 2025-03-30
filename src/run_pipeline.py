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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_name, script_args=None, timeout=1800):
    """Run a Python script with arguments."""
    if script_args is None:
        script_args = []
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'scripts', script_name)
    command = [sys.executable, script_path] + script_args
    
    logger.info(f"Running {script_name} with args: {' '.join(script_args)}")
    
    try:
        start_time = time.time()
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout  # Add timeout to prevent hanging
        )
        execution_time = time.time() - start_time
        logger.info(f"Script execution time: {execution_time:.2f} seconds")
        logger.info(f"{script_name} completed successfully")
        return True, process.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout after {timeout} seconds while running {script_name}")
        return False, f"Timeout after {timeout} seconds"
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False, e.stderr

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Pipeline Runner"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train_mlp_basic", 
            "train_mlp_enhanced", 
            "train_roberta", 
            "train_kernel", 
            "train_pca",
            "train_all", 
            "test", 
            "compare"
        ],
        default="compare",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--test_model",
        type=str,
        choices=["mlp_basic", "mlp_enhanced", "roberta", "kernel", "pca"],
        default="mlp_enhanced",
        help="Model to test (when mode is 'test')"
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
    
    # Script mapping
    script_mapping = {
        "mlp_basic": "mlp_basic_main.py",
        "mlp_enhanced": "mlp_enhanced_main.py",
        "roberta": "roberta_main.py",
        "kernel": "kernel_approximation_main.py",
        "pca": "randomized_pca_main.py"
    }
    
    # Run the appropriate pipeline component
    if args.mode == "train_mlp_basic":
        success, output = run_script(script_mapping["mlp_basic"], common_args, timeout=args.timeout)
        if success:
            logger.info("MLP Basic model training completed")
    
    elif args.mode == "train_mlp_enhanced":
        success, output = run_script(script_mapping["mlp_enhanced"], common_args, timeout=args.timeout)
        if success:
            logger.info("MLP Enhanced model training completed")
    
    elif args.mode == "train_roberta":
        success, output = run_script(script_mapping["roberta"], common_args, timeout=args.timeout)
        if success:
            logger.info("RoBERTa model training completed")
    
    elif args.mode == "train_kernel":
        success, output = run_script(script_mapping["kernel"], common_args, timeout=args.timeout)
        if success:
            logger.info("Kernel Approximation model training completed")
    
    elif args.mode == "train_pca":
        success, output = run_script(script_mapping["pca"], common_args, timeout=args.timeout)
        if success:
            logger.info("Randomized PCA model training completed")
    
    elif args.mode == "train_all":
        # Train models
        successes = {}
        
        # MLP Basic model
        success1, _ = run_script(script_mapping["mlp_basic"], common_args, timeout=args.timeout)
        successes["mlp_basic"] = success1
        
        # MLP Enhanced model
        success2, _ = run_script(script_mapping["mlp_enhanced"], common_args, timeout=args.timeout)
        successes["mlp_enhanced"] = success2
        
        # Kernel Approximation model
        success3, _ = run_script(script_mapping["kernel"], common_args, timeout=args.timeout)
        successes["kernel"] = success3
        
        # Randomized PCA model
        success4, _ = run_script(script_mapping["pca"], common_args, timeout=args.timeout)
        successes["pca"] = success4
        
        # RoBERTa model (optional due to higher resource requirements)
        if args.include_roberta or args.include_all:
            # Reduce sample size for RoBERTa to avoid excessive training time
            roberta_args = [
                "--sample_size", str(min(args.sample_size, 5000)),  # Cap at 5000 for RoBERTa
                "--verbose", str(args.verbose)
            ]
            
            # Train RoBERTa model 
            success5, _ = run_script(script_mapping["roberta"], roberta_args, timeout=args.timeout)
            successes["roberta"] = success5
        
        # Count successful trainings
        successful_count = sum(1 for success in successes.values() if success)
        
        if successful_count > 0:
            logger.info(f"{successful_count} models trained successfully")
            
            # Run comparison
            compare_args = ["--timeout", str(args.timeout)]
            
            if args.include_all or args.include_roberta and successes.get("roberta", False):
                compare_args.append("--include_roberta")
                
            run_script("compare_models.py", compare_args, timeout=args.timeout)
        else:
            logger.error("All model trainings failed")

    elif args.mode == "test":
        # Test the specified model
        if args.test_model in script_mapping:
            test_script = script_mapping[args.test_model]
            success, output = run_script(test_script, common_args, timeout=args.timeout)
            if success:
                logger.info(f"Testing {args.test_model} model completed")
        else:
            logger.error(f"Unknown model: {args.test_model}")
    
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
            
        success, output = run_script("compare_models.py", compare_args, timeout=args.timeout)
        if success:
            logger.info("Model comparison completed")
    
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main() 