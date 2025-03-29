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
        choices=["train_basic", "train_enhanced", "train_roberta", "train_all", "test", "compare"],
        default="compare",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--test_model",
        type=str,
        choices=["basic", "enhanced", "roberta"],
        default="enhanced",
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
    
    args = parser.parse_args()
    
    # Common args for all scripts
    common_args = [
        "--sample_size", str(args.sample_size),
        "--verbose", str(args.verbose)
    ]
    
    # Run the appropriate pipeline component
    if args.mode == "train_basic":
        success, output = run_script("main.py", common_args, timeout=args.timeout)
        if success:
            logger.info("Basic model training completed")
    
    elif args.mode == "train_enhanced":
        success, output = run_script("enhanced_main.py", common_args, timeout=args.timeout)
        if success:
            logger.info("Enhanced model training completed")
    
    elif args.mode == "train_roberta":
        # Add timeout to the roberta arguments
        roberta_args = common_args + ["--timeout", str(args.timeout)]
        success, output = run_script("roberta_main.py", roberta_args, timeout=args.timeout)
        if success:
            logger.info("RoBERTa model training completed")
    
    elif args.mode == "train_all":
        # Train basic model
        success1, _ = run_script("main.py", common_args, timeout=args.timeout)
        
        # Train enhanced model
        success2, _ = run_script("enhanced_main.py", common_args, timeout=args.timeout)
        
        # Reduce sample size for RoBERTa to avoid excessive training time
        # while still getting meaningful results
        roberta_args = [
            "--sample_size", str(min(args.sample_size, 5000)),  # Cap at 5000 for RoBERTa
            "--verbose", str(args.verbose),
            "--timeout", str(args.timeout)
        ]
        
        # Train RoBERTa model 
        success3, _ = run_script("roberta_main.py", roberta_args, timeout=args.timeout)
        
        if success1 and success2:
            logger.info("Basic and Enhanced models trained successfully")
            
            # Run comparison, including RoBERTa if it was successful
            if success3:
                logger.info("All models including RoBERTa trained successfully")
                compare_args = ["--include_roberta", "--timeout", str(args.timeout)]
                run_script("compare_models.py", compare_args, timeout=args.timeout)
            else:
                logger.info("RoBERTa model training failed, comparing only Basic and Enhanced models")
                run_script("compare_models.py", ["--timeout", str(args.timeout)], timeout=args.timeout)
        else:
            logger.error("One or more models failed to train")
    
    elif args.mode == "test":
        # Since we've moved test_model.py to misc, we'll now just run the appropriate model script
        if args.test_model == "basic":
            test_args = common_args
            success, output = run_script("main.py", test_args, timeout=args.timeout)
        elif args.test_model == "enhanced":
            test_args = common_args
            success, output = run_script("enhanced_main.py", test_args, timeout=args.timeout)
        else:  # roberta
            test_args = common_args
            success, output = run_script("roberta_main.py", test_args, timeout=args.timeout)
            
        if success:
            logger.info(f"Testing {args.test_model} model completed")
    
    elif args.mode == "compare":
        compare_args = []
        if args.include_roberta:
            compare_args.append("--include_roberta")
        
        compare_args.extend(["--timeout", str(args.timeout)])
        
        if args.skip_training:
            compare_args.append("--skip_training")
            
        success, output = run_script("compare_models.py", compare_args, timeout=args.timeout)
        if success:
            logger.info("Model comparison completed")
    
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main() 