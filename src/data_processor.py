"""
Data processing module for sentiment analysis project.
Handles data loading, preprocessing, and feature extraction.
"""

import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def load_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and sample the dataset from Kaggle.
    
    Args:
        sample_size: Number of samples to load. If None, loads full dataset.
        
    Returns:
        DataFrame containing the loaded data.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("zphudzz/tweets-clean-posneg-v1")
        logger.info(f"Dataset downloaded to: {path}")
        
        # Assuming the CSV file is in the downloaded directory
        csv_path = os.path.join(path[0], 'tweets.csv')
        df = pd.read_csv(csv_path)
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records from the dataset")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text: Input text to preprocess.
        
    Returns:
        Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def prepare_data(df: pd.DataFrame, 
                text_column: str = 'text',
                label_column: str = 'sentiment',
                test_size: float = 0.2,
                max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training by preprocessing text and creating TF-IDF features.
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text
        label_column: Name of the column containing labels
        test_size: Proportion of data to use for testing
        max_features: Maximum number of features for TF-IDF vectorization
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Preprocess text
    logger.info("Preprocessing text...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df[label_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f"Data split complete. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test 