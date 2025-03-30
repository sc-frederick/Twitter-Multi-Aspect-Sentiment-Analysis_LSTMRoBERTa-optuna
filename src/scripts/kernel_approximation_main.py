"""
Script for training a Kernel Approximation based sentiment analysis model.
"""

import logging
import os
import sys
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import signal

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.data_processor import DataProcessor
from utils.results_tracker import save_model_results, save_report_to_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs(os.path.join(src_dir, 'models'), exist_ok=True)

class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")

def plot_confusion_matrix(cm, classes=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names
    """
    if classes is None:
        classes = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    logger.info("Confusion matrix plot displayed")

def main():
    """Main function to run the kernel approximation model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Kernel Approximation sentiment analysis model")
    
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=20000,
        help="Number of samples to use for training"
    )
    
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="Verbosity mode"
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
        # Load data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data(sample_size=args.sample_size)
        
        # Process data
        logger.info("Processing data...")
        X = data_processor.vectorize_texts(df['text_clean'].tolist())
        y = (df['target'] == 4).astype(np.int32).values  # Convert 0/4 to 0/1
        
        # Define model parameters
        model_params = {
            "model_type": "Kernel Approximation with RBF",
            "gamma": 0.1,
            "n_components": 100,
            "random_state": 42,
            "learning_rate": "optimal",
            "max_iter": 1000,
            "vectorizer": "TF-IDF",
            "sample_size": args.sample_size
        }
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Create and train model
        logger.info("Creating and training model...")
        signal.alarm(args.timeout)
        try:
            # Create kernel approximation pipeline
            model = Pipeline([
                ('kernel', RBFSampler(gamma=model_params["gamma"], 
                                    n_components=model_params["n_components"], 
                                    random_state=model_params["random_state"])),
                ('sgd', SGDClassifier(max_iter=model_params["max_iter"], 
                                    learning_rate=model_params["learning_rate"],
                                    random_state=model_params["random_state"]))
            ])
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            signal.alarm(0)  # Cancel the timeout
        except TimeoutError:
            logger.warning(f"Model training timed out after {args.timeout} seconds")
            raise
        
        # Evaluate model
        logger.info("Evaluating model...")
        signal.alarm(args.timeout)
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            logger.info("\nEvaluation Results:")
            logger.info(f"Model: Kernel Approximation with RBF")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Plot confusion matrix
            plot_confusion_matrix(cm)
            
            signal.alarm(0)  # Cancel the timeout
        except TimeoutError:
            logger.warning(f"Model evaluation timed out after {args.timeout} seconds")
            raise
        
        # Test with example texts
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
                # Process the text
                processed_text = data_processor.preprocess_text(text)
                vectorized_text = data_processor.vectorize_texts([processed_text])
                
                # Predict
                predicted_proba = model.predict_proba(vectorized_text)
                predicted_class = model.predict(vectorized_text)[0]
                confidence = predicted_proba[0][predicted_class] if predicted_proba is not None else 0.5
                
                sentiment = "Positive" if predicted_class == 1 else "Negative"
                
                example_predictions[text] = {
                    "label": sentiment,
                    "confidence": float(confidence)
                }
                
                logger.info(f"\nText: '{text}'")
                logger.info(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
                
                signal.alarm(0)  # Cancel the timeout
            except TimeoutError:
                logger.warning(f"Prediction for '{text}' timed out")
                example_predictions[text] = {
                    "label": "Unknown (timeout)",
                    "confidence": 0.0
                }
            except Exception as e:
                logger.warning(f"Error predicting for '{text}': {str(e)}")
                example_predictions[text] = {
                    "label": "Error",
                    "confidence": 0.0
                }
        
        # Save model results
        logger.info("Saving model results...")
        save_model_results(
            model_name="KernelApproximationModel",
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