"""
Script for training and evaluating the LSTM sentiment analysis model.
"""

import logging
import os
import sys
import argparse
import time
import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.data_processor import DataProcessor
from utils.lstm_classifier import LSTMSentimentClassifier
from utils.results_tracker import save_model_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs(os.path.join(src_dir, 'models'), exist_ok=True)

def plot_confusion_matrix(cm, classes=None):
    """Plot confusion matrix."""
    if classes is None:
        classes = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the LSTM model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train/Test LSTM sentiment analysis model")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'test'],
        default='train',
        help="Whether to train or test the model"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
        help="Number of samples to use"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training/testing"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimization"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and process data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data(sample_size=args.sample_size)
        
        # Convert labels from 0/4 to 0/1
        texts = df['text_clean'].tolist()
        labels = (df['target'] == 4).astype(int).tolist()
        
        # Initialize model
        model = LSTMSentimentClassifier(
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Load existing model if testing
        if args.mode == 'test':
            model_path = os.path.join(src_dir, 'models', 'lstm_model.pt')
            if os.path.exists(model_path):
                logger.info("Loading existing model...")
                model.load_model(model_path)
            else:
                logger.error(f"No model found at {model_path}")
                return
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        if args.mode == 'train':
            # Train model
            logger.info("Training model...")
            start_time = time.time()
            history = model.train(train_texts, train_labels, test_texts, test_labels)
            training_time = time.time() - start_time
            
            # Save model
            model_path = os.path.join(src_dir, 'models', 'lstm_model.pt')
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        test_predictions = model.predict(test_texts)
        test_pred_labels = [1 if pred['label'] == 'Positive' else 0 for pred in test_predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_pred_labels)
        precision = precision_score(test_labels, test_pred_labels)
        recall = recall_score(test_labels, test_pred_labels)
        f1 = f1_score(test_labels, test_pred_labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info("\nEvaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Test with example texts
        example_texts = [
            "This product is amazing! I love it so much.",
            "Terrible experience, would not recommend.",
            "It's okay, not great but not bad either.",
            "The customer service was outstanding!",
            "Disappointed with the quality, not worth the price."
        ]
        
        logger.info("\nTesting model with example texts...")
        example_predictions = {}
        predictions = model.predict(example_texts)
        
        for text, pred in zip(example_texts, predictions):
            example_predictions[text] = pred
            logger.info(f"\nText: '{text}'")
            logger.info(f"Prediction: {pred['label']} (confidence: {pred['confidence']:.4f})")
        
        # Save results
        model_params = {
            'model_type': 'LSTM',
            'embedding_dim': model.model.embedding.embedding_dim,
            'hidden_dim': model.model.lstm.hidden_size,
            'num_layers': model.model.lstm.num_layers,
            'dropout': model.dropout,
            'bidirectional': model.model.lstm.bidirectional,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'sample_size': args.sample_size
        }
        
        if args.mode == 'train':
            model_params['training_time'] = training_time
        
        save_model_results(
            model_name="LSTMModel",
            metrics=metrics,
            parameters=model_params,
            example_predictions=example_predictions
        )
        
        logger.info(f"\n{args.mode.capitalize()} completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 