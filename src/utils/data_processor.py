# src/utils/data_processor.py
"""
Data processing module for sentiment analysis project.
Handles data loading, preprocessing, and feature extraction.
Loads train.csv and test.csv separately.
"""

import os
from typing import Tuple, Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Note: TF-IDF vectorization is kept but might not be used by RoBERTa models
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import logging
import sqlite3 # Kept for potential future use, but primary loading is CSV
import html
from pathlib import Path
from dotenv import load_dotenv
import contractions
import kagglehub # Use kagglehub for downloading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Define dataset details (Update this if the dataset name/path changes on Kaggle)
KAGGLE_DATASET_OWNER = "kaggle"
KAGGLE_DATASET_NAME = "tweet-sentiment-extraction"
TRAIN_CSV_NAME = "train.csv"
TEST_CSV_NAME = "test.csv"

# Get downloads directory path (or a dedicated data directory)
# Using a 'data' subdirectory within the project source seems more standard
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, "tweets_dataset.db") # Keep DB path for reference

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK data (punkt, stopwords, wordnet)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK data downloaded.")


class TextPreprocessor:
    """Class for text preprocessing and cleaning operations."""

    def __init__(self,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False, # Keep hashtag text, remove '#'
                 fix_contractions: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = True, # Keep numbers? Might be relevant. Set to False if needed.
                 remove_stopwords: bool = True, # Often less critical for transformers
                 lemmatize: bool = True, # Stemming/Lemmatization less critical for transformers
                 handle_negation: bool = False, # Transformers handle negation contextually
                 html_decode: bool = True):
        """
        Initialize with preprocessing options, defaults adjusted for transformer models.
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.fix_contractions = fix_contractions
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.handle_negation = handle_negation
        self.html_decode = html_decode

        # Initialize NLTK components if needed
        self.stop_words = set()
        self.lemmatizer = None
        if self.remove_stopwords or self.handle_negation:
            self.stop_words = set(stopwords.words('english'))
            self.negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}
            if self.handle_negation:
                self.stop_words = self.stop_words.difference(self.negation_words)
            else:
                # Keep negation words if not specifically handled but stopwords are removed
                self.stop_words = self.stop_words.difference(self.negation_words)

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

        logger.info(f"TextPreprocessor initialized with options: {self.__dict__}")


    def clean_text(self, text: str) -> str:
        """
        Clean text using multiple preprocessing techniques. Optimized for transformers.
        Focuses on removing noise like URLs, mentions, HTML, fixing contractions.
        Less aggressive tokenization/stopword removal/lemmatization by default.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # HTML decode first
        if self.html_decode:
            text = html.unescape(text)

        # Fix contractions before potential lowercasing or punctuation removal
        if self.fix_contractions:
            text = contractions.fix(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions (@username)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)

        # Remove hashtags or just the # symbol
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text) # Remove entire hashtag
        else:
            text = re.sub(r'#', '', text) # Keep the text, remove only '#'

        # Remove numbers (optional, might be useful context)
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation (optional, transformers can handle some punctuation)
        if self.remove_punctuation:
            # Keep basic sentence structure punctuation if needed?
            # For now, remove all as per original setting
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Optional: Tokenization, Stopword removal, Lemmatization
        # These are often skipped or handled differently for transformers
        if self.remove_stopwords or self.lemmatize or self.handle_negation:
             tokens = word_tokenize(text)
             if self.handle_negation:
                 tokens = self._handle_text_negation(tokens) # Use helper method
             if self.remove_stopwords:
                 tokens = [token for token in tokens if token not in self.stop_words]
             if self.lemmatize and self.lemmatizer:
                 tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
             text = ' '.join(tokens)

        return text

    def _handle_text_negation(self, tokens: List[str]) -> List[str]:
        """Internal helper for negation handling (if enabled)."""
        negated = False
        processed_tokens = []
        punctuation_marks = {'.', '!', '?', ';', ':', ','} # Define punctuation that resets negation

        for token in tokens:
            if token in self.negation_words:
                negated = True
                processed_tokens.append(token)
            elif token in punctuation_marks:
                negated = False # Reset negation at punctuation
                processed_tokens.append(token)
            elif negated:
                processed_tokens.append(f"NEG_{token}")
            else:
                processed_tokens.append(token)
        return processed_tokens

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Apply text cleaning to a DataFrame column.

        Args:
            df: Input DataFrame.
            text_column: Name of the column containing text to clean.

        Returns:
            DataFrame with a new 'text_clean' column.
        """
        if text_column not in df.columns:
             logger.error(f"Text column '{text_column}' not found in DataFrame.")
             raise ValueError(f"Missing required text column: {text_column}")

        # Handle potential NaN values in the text column before applying cleaning
        df[text_column] = df[text_column].fillna('') # Replace NaN with empty string

        logger.info(f"Applying text cleaning to '{text_column}' column...")
        df['text_clean'] = df[text_column].apply(self.clean_text)
        logger.info("Text cleaning complete.")
        return df


class DataManager:
    """Class for data loading, saving, and potential database operations."""

    def __init__(self, data_dir: str = DATA_DIR, database_path: str = DATABASE_PATH):
        """
        Initialize with data directory and database path.

        Args:
            data_dir: Directory to store downloaded data.
            database_path: Path to SQLite database (optional).
        """
        self.data_dir = data_dir
        self.database_path = database_path
        self.text_preprocessor = TextPreprocessor() # Instantiate preprocessor here

    def download_and_get_path(self) -> str:
        """Downloads the dataset using kagglehub and returns the path."""
        logger.info(f"Downloading dataset '{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}' to '{self.data_dir}'...")
        try:
            # kagglehub.dataset_download returns the path to the downloaded files
            # Removed the unsupported 'force' argument
            dataset_path = kagglehub.dataset_download(
                f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}",
                path=self.data_dir
            )
            logger.info(f"Dataset downloaded/verified at: {dataset_path}")
            return str(dataset_path) # Ensure it's a string
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}", exc_info=True)
            # Attempt to provide more specific guidance based on common errors
            if "403 Client Error" in str(e) or "Forbidden" in str(e):
                 logger.error("Ensure your Kaggle API token (kaggle.json) is correctly placed (e.g., ~/.kaggle/kaggle.json) and has the necessary permissions.")
            elif "404 Client Error" in str(e) or "Not Found" in str(e):
                 logger.error(f"Dataset '{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}' not found. Check owner/name.")
            raise # Re-raise the exception after logging

    def load_and_preprocess_csv(self, file_path: str, text_column: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Loads a CSV, preprocesses text, and optionally samples."""
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found at: {file_path}")
            raise FileNotFoundError(f"Required CSV file missing: {file_path}")

        logger.info(f"Loading CSV: {file_path}")
        try:
            df = pd.read_csv(file_path)
            # Handle potential byte decoding issues if they arise
        except UnicodeDecodeError:
             logger.warning(f"UnicodeDecodeError reading {file_path}. Trying with 'latin-1' encoding.")
             try:
                 df = pd.read_csv(file_path, encoding='latin-1')
             except Exception as e:
                 logger.error(f"Failed to read {file_path} even with latin-1 encoding: {e}")
                 raise
        except Exception as e:
             logger.error(f"Error loading CSV {file_path}: {e}")
             raise

        logger.info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}. Columns: {df.columns.tolist()}")

        # Preprocess the text
        df = self.text_preprocessor.preprocess_dataframe(df, text_column=text_column)

        # --- Sentiment Mapping (Specific to this dataset) ---
        # Map 'sentiment' column (neutral, positive, negative) to numerical labels if it exists
        if 'sentiment' in df.columns:
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            # Create a 'label' column based on the mapping
            df['label'] = df['sentiment'].map(sentiment_map)
            # Handle cases where sentiment might be missing or different
            # Fill NaN labels potentially created by map with a default (e.g., neutral) or drop them
            original_len = len(df)
            df.dropna(subset=['label'], inplace=True)
            if len(df) < original_len:
                 logger.warning(f"Dropped {original_len - len(df)} rows with missing/unmappable sentiment.")
            df['label'] = df['label'].astype(int)
            logger.info("Mapped 'sentiment' to numerical 'label' column (0: neg, 1: neu, 2: pos).")
        else:
             # If there's no 'sentiment' column, we might not need labels (e.g., for test set prediction)
             # Or, if labels are expected, this indicates a problem.
             logger.warning(f"'sentiment' column not found in {os.path.basename(file_path)}. Cannot create numerical 'label'.")
             # Add a placeholder label column if downstream code expects it for the test set
             if 'label' not in df.columns:
                 df['label'] = -1 # Or some other indicator

        # Sample if requested
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} records from {os.path.basename(file_path)}...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled DataFrame size: {len(df)}")

        return df

    def load_data(self, sample_size_train: Optional[int] = None, sample_size_test: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and preprocesses train.csv and test.csv from the Kaggle dataset.

        Args:
            sample_size_train: Number of samples to load from train.csv. If None, loads full dataset.
            sample_size_test: Number of samples to load from test.csv. If None, loads full dataset.

        Returns:
            Tuple containing (train_dataframe, test_dataframe)
        """
        try:
            dataset_dir = self.download_and_get_path()
            train_path = os.path.join(dataset_dir, TRAIN_CSV_NAME)
            test_path = os.path.join(dataset_dir, TEST_CSV_NAME)

            # Load and preprocess train data
            # The text column in this dataset is simply 'text'
            train_df = self.load_and_preprocess_csv(train_path, text_column='text', sample_size=sample_size_train)

            # Load and preprocess test data
            test_df = self.load_and_preprocess_csv(test_path, text_column='text', sample_size=sample_size_test)

            # --- Column Check and Alignment (Optional but Recommended) ---
            # Ensure essential columns ('text', 'text_clean', 'label') exist in both
            required_cols = ['text', 'text_clean', 'label'] # Adjust if test set lacks labels
            missing_train = [col for col in required_cols if col not in train_df.columns]
            # Test might not have labels, or might have placeholder -1
            missing_test = [col for col in required_cols if col not in test_df.columns and col != 'label']

            if missing_train:
                 logger.error(f"Train DataFrame missing required columns: {missing_train}")
                 raise ValueError("Train data loading failed due to missing columns.")
            if missing_test:
                 logger.error(f"Test DataFrame missing required columns: {missing_test}")
                 raise ValueError("Test data loading failed due to missing columns.")

            logger.info(f"Successfully loaded and preprocessed train ({len(train_df)} rows) and test ({len(test_df)} rows) data.")
            return train_df, test_df

        except FileNotFoundError as e:
            logger.error(f"Data loading failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            raise

    # --- TF-IDF Methods (Kept for potential non-transformer use cases) ---

    def prepare_data_tfidf(self,
                    df_train: pd.DataFrame,
                    df_test: pd.DataFrame,
                    text_column: str = 'text_clean',
                    label_column: str = 'label', # Using 'label' now
                    max_features: int = 5000,
                    ngram_range: Tuple[int, int] = (1, 2),
                    min_df: int = 2,
                    max_df: float = 0.95) -> Dict[str, Union[np.ndarray, TfidfVectorizer]]:
        """
        Prepare data for training using TF-IDF features. Fits on train, transforms both.

        Args:
            df_train: Training DataFrame
            df_test: Testing DataFrame
            text_column: Name of the column containing preprocessed text
            label_column: Name of the column containing labels
            max_features: Maximum number of features for TF-IDF vectorization
            ngram_range: Range of n-grams to include
            min_df: Minimum document frequency
            max_df: Maximum document frequency

        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, and vectorizer
        """
        logger.info("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )

        # Fit on training data ONLY
        X_train = vectorizer.fit_transform(df_train[text_column])
        y_train = df_train[label_column].values

        # Transform test data
        X_test = vectorizer.transform(df_test[text_column])
        y_test = df_test[label_column].values # Assumes test has labels for evaluation

        logger.info(f"TF-IDF Data prepared. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'vectorizer': vectorizer
        }

    def preprocess_new_text_tfidf(self, text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
        """
        Preprocess new text for prediction using a fitted TF-IDF vectorizer.

        Args:
            text: Input text to preprocess
            vectorizer: Fitted TfidfVectorizer instance

        Returns:
            Vectorized text ready for model prediction
        """
        if vectorizer is None:
            raise ValueError("Vectorizer not provided or not fitted.")

        # Clean and preprocess text using the same preprocessor instance
        processed_text = self.text_preprocessor.clean_text(text)

        # Vectorize
        return vectorizer.transform([processed_text])

# --- DataProcessor Class (Wrapper) ---
# This class remains largely the same, but its load_data method is now handled by DataManager
class DataProcessor:
    """Class for orchestrating sentiment analysis data processing."""

    def __init__(self, data_dir: str = DATA_DIR, database_path: str = DATABASE_PATH):
        """
        Initialize with data directory and database path.

        Args:
            data_dir: Directory to store downloaded data.
            database_path: Path to SQLite database (optional).
        """
        self.data_manager = DataManager(data_dir=data_dir, database_path=database_path)
        self.vectorizer = None # For TF-IDF

    def load_data(self, sample_size_train: Optional[int] = None, sample_size_test: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads train and test data using the DataManager.

        Args:
            sample_size_train: Number of training samples to load.
            sample_size_test: Number of test samples to load.

        Returns:
            Tuple containing (train_dataframe, test_dataframe)
        """
        return self.data_manager.load_data(sample_size_train=sample_size_train, sample_size_test=sample_size_test)

    def prepare_data_tfidf(self, *args, **kwargs):
        """Prepares data using TF-IDF via DataManager."""
        results = self.data_manager.prepare_data_tfidf(*args, **kwargs)
        self.vectorizer = results['vectorizer'] # Store the fitted vectorizer
        return results

    def preprocess_new_text_tfidf(self, text: str) -> np.ndarray:
        """Preprocesses new text using the fitted TF-IDF vectorizer."""
        if self.vectorizer is None:
             raise ValueError("TF-IDF Vectorizer not initialized. Call prepare_data_tfidf first.")
        return self.data_manager.preprocess_new_text_tfidf(text, self.vectorizer)

    def get_text_preprocessor(self) -> TextPreprocessor:
         """Returns the TextPreprocessor instance."""
         return self.data_manager.text_preprocessor

# Example Usage (for testing the module directly)
if __name__ == "__main__":
    logger.info("--- Testing DataProcessor ---")
    processor = DataProcessor()
    try:
        # Load a small sample
        train_df_sample, test_df_sample = processor.load_data(sample_size_train=100, sample_size_test=50)
        logger.info("\n--- Train DataFrame Sample Head ---")
        print(train_df_sample.head())
        logger.info("\n--- Test DataFrame Sample Head ---")
        print(test_df_sample.head())

        logger.info("\n--- Train DataFrame Info ---")
        train_df_sample.info()
        logger.info("\n--- Test DataFrame Info ---")
        test_df_sample.info()

        # Check label distribution in train sample
        if 'label' in train_df_sample.columns:
            logger.info("\n--- Train Sample Label Distribution ---")
            print(train_df_sample['label'].value_counts())
        else:
            logger.warning("Label column not found in train sample.")

        # Test preprocessing a single text
        preprocessor = processor.get_text_preprocessor()
        sample_text = "This is a test tweet! Check out https://example.com @user #testing :) It's not bad, is it?"
        cleaned_text = preprocessor.clean_text(sample_text)
        logger.info(f"\n--- Text Preprocessing Example ---")
        logger.info(f"Original: {sample_text}")
        logger.info(f"Cleaned:  {cleaned_text}")

        # Test TF-IDF preparation (optional)
        # logger.info("\n--- Testing TF-IDF Preparation ---")
        # tfidf_data = processor.prepare_data_tfidf(train_df_sample, test_df_sample)
        # logger.info(f"TF-IDF Shapes: Train={tfidf_data['X_train'].shape}, Test={tfidf_data['X_test'].shape}")
        # logger.info(f"Vectorizer features: {len(tfidf_data['vectorizer'].get_feature_names_out())}")

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
