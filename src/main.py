"""
Main script for sentiment analysis project.
Implements the complete pipeline from data loading to model evaluation.
"""

import logging
import os
from dotenv import load_dotenv
from data_processor import DataProcessor
from model import create_model, train_model, evaluate_model, plot_training_history, plot_confusion_matrix

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the sentiment analysis pipeline."""
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Load and prepare data
        logger.info("Loading data...")
        df = data_processor.load_data(sample_size=10000)  # Start with 10k samples
        
        logger.info("Preparing data for training...")
        data = data_processor.prepare_data(
            df,
            text_column='text_clean',
            label_column='target',
            test_size=0.2,
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Create a smaller validation set from the training data
        X_train, X_val, y_train, y_val = data_processor.prepare_data(
            df.iloc[:len(X_train)],
            text_column='text_clean',
            label_column='target',
            test_size=0.2
        ).values()  # Extract values from dict
        
        # Create and train model
        logger.info("Creating model...")
        model = create_model(input_dim=X_train.shape[1])
        
        logger.info("Training model...")
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=10,
            batch_size=32
        )
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = evaluate_model(model, X_test, y_test)
        
        # Print classification report
        logger.info("\nClassification Report:")
        print(evaluation_results['classification_report'])
        
        # Plot confusion matrix
        logger.info("Plotting confusion matrix...")
        plot_confusion_matrix(evaluation_results['confusion_matrix'])
        
        # Test a custom prediction
        test_text = "This movie was absolutely amazing! I loved every minute of it."
        logger.info(f"\nTesting prediction with text: '{test_text}'")
        
        vectorized_text = data_processor.preprocess_new_text(test_text)
        prediction = model.predict(vectorized_text)
        sentiment = "Positive" if prediction.argmax() == 4 else "Negative"
        logger.info(f"Predicted sentiment: {sentiment} (Raw prediction: {prediction})")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 