"""
Main script for sentiment analysis project.
Tests data loading and preprocessing functionality.
"""

import logging
from data_processor import load_data, prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test data loading and preprocessing."""
    try:
        # Load a sample of the data (e.g., 1000 records)
        logger.info("Loading data...")
        df = load_data(sample_size=1000)
        
        # Display basic information about the dataset
        logger.info("\nDataset Info:")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nColumns:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        # Prepare data for training
        logger.info("\nPreparing data for training...")
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            text_column='text',
            label_column='sentiment',
            test_size=0.2,
            max_features=5000
        )
        
        # Display information about the prepared data
        logger.info("\nPrepared Data Info:")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 