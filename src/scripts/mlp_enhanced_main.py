"""
Enhanced main script for sentiment analysis project.
Implements an improved pipeline with enhanced model architecture and data processing.
"""

import logging
import os
import sys
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.data_processor import DataProcessor
from utils.model import EnhancedSentimentModel, plot_training_history, plot_confusion_matrix
from utils.results_tracker import save_model_results, save_report_to_file

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs(os.path.join(src_dir, 'models'), exist_ok=True)

def sparse_to_dense(sparse_matrix):
    """Convert sparse matrix to dense numpy array."""
    if sp.issparse(sparse_matrix):
        return sparse_matrix.toarray()
    return sparse_matrix

def create_enhanced_model(input_dim):
    """Create an enhanced sentiment analysis model with more layers."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Define custom metrics
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1_score')
    
    # Compile the model with a slightly lower learning rate for better convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', precision, recall, f1_score]
    )
    
    return model

def main():
    """Main function to train and evaluate the enhanced model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train enhanced sentiment analysis model")
    
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
        help="Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data(sample_size=args.sample_size)
        
        logger.info("Preparing data for training...")
        # Improve feature extraction with more n-grams and features
        data = data_processor.prepare_data(
            df,
            text_column='text_clean',
            label_column='target',
            test_size=0.2,
            max_features=8000,  # Increased from 5000
            ngram_range=(1, 3),  # Include trigrams
            min_df=3,
            max_df=0.9
        )
        
        # Convert sparse matrices to dense arrays
        logger.info("Converting sparse matrices to dense arrays...")
        X_train = sparse_to_dense(data['X_train'])
        X_test = sparse_to_dense(data['X_test'])
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Convert target values from 0/4 to 0/1 for binary classification
        y_train = (y_train == 4).astype(np.int32)
        y_test = (y_test == 4).astype(np.int32)
        
        # Create a smaller validation set from the training data
        logger.info("Creating validation set...")
        val_size = int(X_train.shape[0] * 0.15)  # Reduced validation size
        indices = np.random.permutation(X_train.shape[0])
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Create model
        logger.info("Creating enhanced model...")
        model = create_enhanced_model(input_dim=X_train.shape[1])
        
        # Early stopping callback with more patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        logger.info("Training enhanced model...")
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=args.verbose
        )
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history)
        
        # Evaluate the model
        logger.info("Evaluating enhanced model...")
        evaluation = model.evaluate(X_test, y_test, verbose=1)
        
        # Get metrics
        metrics = {
            'accuracy': float(evaluation[1]),
            'precision': float(evaluation[2]),
            'recall': float(evaluation[3]),
            'f1_score': float(evaluation[4])
        }
        
        logger.info("\nEvaluation Results:")
        logger.info(f"Model: TensorFlow Deep Neural Network (Enhanced)")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Get predictions for confusion matrix
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        logger.info("Plotting confusion matrix...")
        plot_confusion_matrix(cm)
        
        # Save model
        model_path = os.path.join(src_dir, 'models', 'enhanced_sentiment_model')
        logger.info(f"Saving enhanced model to {model_path}...")
        model.save(model_path)
        
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
        
        for text in examples:
            # Process the text
            vectorized_text = data_processor.preprocess_new_text(text)
            
            # Convert to dense if needed
            if sp.issparse(vectorized_text):
                vectorized_text = vectorized_text.toarray()
                
            # Predict
            prediction = model.predict(vectorized_text)
            pred_class = np.argmax(prediction)
            confidence = prediction[0][pred_class]
            sentiment = "Positive" if pred_class == 1 else "Negative"
            
            # Save prediction
            example_predictions[text] = {
                "label": sentiment,
                "confidence": float(confidence)
            }
            
            logger.info(f"\nText: '{text}'")
            logger.info(f"Predicted: {sentiment} (confidence: {confidence:.4f})")
        
        # Define model parameters
        model_params = {
            "model_type": "Enhanced Neural Network",
            "input_dim": X_train.shape[1],
            "hidden_layers": [128, 64, 32, 16],
            "dropout_rates": [0.3, 0.3, 0.2, 0.2],
            "batch_normalization": True,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
            "max_features": 8000,
            "ngram_range": "(1, 3)",
            "min_df": 3,
            "max_df": 0.9
        }

        # Save model results
        logger.info("Saving model results...")
        save_model_results(
            model_name="EnhancedModel",
            metrics=metrics,
            parameters=model_params,
            example_predictions=example_predictions
        )
        
        # Generate and save a report
        save_report_to_file()
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 