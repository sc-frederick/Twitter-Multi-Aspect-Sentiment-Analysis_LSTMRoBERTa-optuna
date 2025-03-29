"""
Script for training a RoBERTa-based sentiment analysis model.
"""

import logging
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import time
import signal

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.data_processor import DataProcessor
from utils.roberta_classifier import RoBERTaSentimentClassifier, plot_training_history, plot_confusion_matrix
from utils.results_tracker import save_model_results, save_report_to_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")

def main():
    """Main function to train and evaluate the RoBERTa model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RoBERTa sentiment analysis model")
    
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=5000,  # Smaller default for RoBERTa as it's more resource-intensive
        help="Number of samples to use for training"
    )
    
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=600,  # 10 minutes default timeout
        help="Timeout in seconds for model operations"
    )
    
    args = parser.parse_args()
    
    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(src_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Load data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data(sample_size=args.sample_size)
        
        logger.info("Preparing data for training...")
        # For RoBERTa, we need the raw text, not the vectorized version
        X = df['text'].values  # Use raw text for RoBERTa
        y = (df['target'] == 4).astype(np.int32).values  # Convert 0/4 to 0/1
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Define model parameters
        model_params = {
            "model_type": "RoBERTa Transformer",
            "pretrained_model": "distilroberta-base",
            "max_length": 128,
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 2e-5,
            "sample_size": args.sample_size
        }
        
        # Create model
        logger.info("Creating RoBERTa model...")
        model = RoBERTaSentimentClassifier(
            pretrained_model=model_params["pretrained_model"],
            max_length=model_params["max_length"]
        )
        
        # Train model with timeout
        logger.info("Training RoBERTa model...")
        signal.alarm(args.timeout)
        try:
            history = model.train(
                X_train=X_train, 
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=model_params["epochs"],
                batch_size=model_params["batch_size"],
                verbose=args.verbose
            )
            signal.alarm(0)  # Cancel the timeout
        except TimeoutError:
            logger.warning("Model training timed out after %d seconds", args.timeout)
            logger.info("Continuing with evaluation of the partially trained model...")
        
        # Plot training history
        logger.info("Plotting training history...")
        if 'history' in locals():
            plot_training_history(history)
        
        # Evaluate the model with timeout
        logger.info("Evaluating RoBERTa model...")
        signal.alarm(args.timeout)
        try:
            metrics = model.evaluate(X_test, y_test)
            signal.alarm(0)  # Cancel the timeout
        except TimeoutError:
            logger.error("Model evaluation timed out after %d seconds", args.timeout)
            metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            
        logger.info("\nEvaluation Results:")
        logger.info(f"Model: RoBERTa Transformer")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Get confusion matrix with timeout
        signal.alarm(args.timeout)
        try:
            y_pred = model.predict_classes(X_test)
            signal.alarm(0)  # Cancel the timeout
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            logger.info("Plotting confusion matrix...")
            plot_confusion_matrix(cm)
        except TimeoutError:
            logger.warning("Getting predictions timed out after %d seconds", args.timeout)
            logger.info("Skipping confusion matrix plot")
        
        # Save model
        logger.info("Saving model...")
        model_path = os.path.join(models_dir, "roberta_sentiment_model")
        model.save(model_path)
        
        # Test model with example texts
        logger.info("\nTesting model with example texts...")
        examples = [
            "This product is amazing! I love it so much.",
            "Terrible experience, would not recommend.",
            "It's okay, not great but not bad either.",
            "The customer service was outstanding!",
            "Disappointed with the quality, not worth the price."
        ]
        
        example_predictions = {}
        
        # Process each example with timeout
        for text in examples:
            signal.alarm(int(args.timeout / 10))  # Shorter timeout for predictions
            try:
                # Predict
                prediction = model.predict(text)
                signal.alarm(0)  # Cancel the timeout
                
                sentiment = "Positive" if prediction['class'] == 1 else "Negative"
                confidence = prediction['confidence']
                
                example_predictions[text] = {
                    "label": sentiment,
                    "confidence": confidence
                }
                
                logger.info(f"\nText: '{text}'")
                logger.info(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
            except TimeoutError:
                logger.warning(f"Prediction for '{text}' timed out")
                example_predictions[text] = {
                    "label": "Unknown (timeout)",
                    "confidence": 0.0
                }
        
        # Save model results
        logger.info("Saving model results...")
        save_model_results(
            model_name="RoBERTaModel",
            metrics=metrics,
            parameters=model_params,
            example_predictions=example_predictions
        )
        
        # Generate and save a report
        save_report_to_file()
        
    except TimeoutError:
        logger.error(f"The operation timed out after {args.timeout} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Always ensure the alarm is canceled
        signal.alarm(0)

if __name__ == "__main__":
    main() 