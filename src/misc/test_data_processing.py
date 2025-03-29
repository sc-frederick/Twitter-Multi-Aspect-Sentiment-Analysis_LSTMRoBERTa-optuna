"""
Test script to verify data processing functionality.
"""

import logging
import os
from data_processor import load_data, DATABASE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test data loading and processing."""
    try:
        # Load a small sample of data
        logger.info("Loading data sample...")
        df = load_data(sample_size=5)
        
        # Display database file path
        logger.info(f"\nDatabase file path: {DATABASE_PATH}")
        logger.info(f"Database exists: {os.path.exists(DATABASE_PATH)}")
        
        # Display basic information about the dataset
        logger.info("\nDataset Info:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Display columns
        logger.info("\nColumns:")
        for col in df.columns:
            logger.info(f"- {col}: {df[col].dtype}")
        
        # Display sample of data
        logger.info("\nSample of data (first 5 rows):")
        print(df[['target', 'text', 'text_clean', 'text_length', 'text_clean_length']].head())
        
        # Display value counts for the target
        logger.info("\nTarget distribution:")
        print(df['target'].value_counts())
        
        # Display basic statistics
        logger.info("\nBasic Statistics:")
        print(df[['text_length', 'text_clean_length']].describe())
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 