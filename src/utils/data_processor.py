# src/utils/data_processor.py
"""
Data processing module for sentiment analysis project.
Handles data loading, preprocessing, and feature extraction.
Loads train.csv and test.csv directly from the local 'src/data' directory.
Includes stricter handling of sentiment labels in train.csv.
"""

import os
from typing import Tuple, Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import logging
import sqlite3
import html
from pathlib import Path
from dotenv import load_dotenv
import contractions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
TRAIN_CSV_NAME = "train.csv"
TEST_CSV_NAME = "test.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, "tweets_dataset.db")

# Download required NLTK data (same as before)
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
    # (Content of TextPreprocessor remains the same as the previous version)
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
        self.stop_words = set()
        self.lemmatizer = None
        if self.remove_stopwords or self.handle_negation:
            self.stop_words = set(stopwords.words('english'))
            self.negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}
            if self.handle_negation:
                self.stop_words = self.stop_words.difference(self.negation_words)
            else:
                self.stop_words = self.stop_words.difference(self.negation_words)
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        logger.info(f"TextPreprocessor initialized with options: {self.__dict__}")

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str): return ""
        # --- Handle potential NaNs in text explicitly ---
        text = str(text) # Convert potential non-strings (like NaN float) to string
        if pd.isna(text): return "" # Return empty if it was NaN
        # --- End NaN handling ---
        if self.html_decode: text = html.unescape(text)
        if self.fix_contractions: text = contractions.fix(text)
        if self.lowercase: text = text.lower()
        if self.remove_urls: text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        if self.remove_mentions: text = re.sub(r'@\w+', '', text)
        if self.remove_hashtags: text = re.sub(r'#\w+', '', text)
        else: text = re.sub(r'#', '', text)
        if self.remove_numbers: text = re.sub(r'\d+', '', text)
        if self.remove_punctuation: text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        if self.remove_stopwords or self.lemmatize or self.handle_negation:
             tokens = word_tokenize(text)
             if self.handle_negation: tokens = self._handle_text_negation(tokens)
             if self.remove_stopwords: tokens = [token for token in tokens if token not in self.stop_words]
             if self.lemmatize and self.lemmatizer: tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
             text = ' '.join(tokens)
        return text

    def _handle_text_negation(self, tokens: List[str]) -> List[str]:
        negated = False
        processed_tokens = []
        punctuation_marks = {'.', '!', '?', ';', ':', ','}
        for token in tokens:
            if token in self.negation_words: negated = True; processed_tokens.append(token)
            elif token in punctuation_marks: negated = False; processed_tokens.append(token)
            elif negated: processed_tokens.append(f"NEG_{token}")
            else: processed_tokens.append(token)
        return processed_tokens

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        if text_column not in df.columns:
             logger.error(f"Text column '{text_column}' not found in DataFrame.")
             raise ValueError(f"Missing required text column: {text_column}")
        # --- Apply NaN handling within clean_text ---
        # df[text_column] = df[text_column].fillna('') # Removed explicit fillna here
        logger.info(f"Applying text cleaning to '{text_column}' column...")
        df['text_clean'] = df[text_column].apply(self.clean_text)
        # --- Drop rows where text became empty after cleaning (optional but good practice) ---
        original_len = len(df)
        df = df[df['text_clean'] != ''].copy()
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows with empty 'text_clean' after preprocessing.")
        # --- End drop empty text ---
        logger.info("Text cleaning complete.")
        return df


class DataManager:
    """Class for data loading, saving, and potential database operations."""

    def __init__(self, data_dir: str = DATA_DIR, database_path: str = DATABASE_PATH):
        self.data_dir = data_dir
        self.database_path = database_path
        self.text_preprocessor = TextPreprocessor()

    def load_and_preprocess_csv(self, file_path: str, text_column: str, is_train_data: bool, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Loads a CSV, preprocesses text, handles labels robustly, and optionally samples.
        """
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found at: {file_path}")
            raise FileNotFoundError(f"Required CSV file missing: {file_path}")

        logger.info(f"Loading CSV: {file_path}")
        try:
            # --- Explicitly set dtype for sentiment to string ---
            df = pd.read_csv(file_path, dtype={'sentiment': str})
        except UnicodeDecodeError:
             logger.warning(f"UnicodeDecodeError reading {file_path}. Trying with 'latin-1' encoding.")
             try: df = pd.read_csv(file_path, encoding='latin-1', dtype={'sentiment': str})
             except Exception as e: logger.error(f"Failed to read {file_path} even with latin-1 encoding: {e}"); raise
        except Exception as e: logger.error(f"Error loading CSV {file_path}: {e}"); raise

        logger.info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}. Columns: {df.columns.tolist()}")

        # --- Handle NaNs in text column BEFORE preprocessing ---
        if text_column in df.columns:
            original_len = len(df)
            df.dropna(subset=[text_column], inplace=True)
            if len(df) < original_len:
                logger.warning(f"Dropped {original_len - len(df)} rows with missing '{text_column}' values.")
        else:
             logger.error(f"Text column '{text_column}' not found before preprocessing.")
             raise ValueError(f"Missing required text column: {text_column}")

        # Preprocess the text
        df = self.text_preprocessor.preprocess_dataframe(df, text_column=text_column)

        # --- Stricter Sentiment Label Handling ---
        if 'sentiment' in df.columns:
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            valid_sentiments = list(sentiment_map.keys())

            # Convert potential non-strings in sentiment to strings, fill NaNs
            df['sentiment'] = df['sentiment'].astype(str).fillna('neutral') # Fill NaNs *before* checking validity

            # Identify rows with invalid sentiment strings
            invalid_sentiment_mask = ~df['sentiment'].isin(valid_sentiments)
            num_invalid = invalid_sentiment_mask.sum()

            if num_invalid > 0:
                logger.warning(f"Found {num_invalid} rows with unexpected sentiment values in '{os.path.basename(file_path)}'. Examples: {df.loc[invalid_sentiment_mask, 'sentiment'].unique()[:5]}.")

            if is_train_data:
                # For training data, KEEP ONLY valid sentiments
                df = df[~invalid_sentiment_mask].copy()
                if num_invalid > 0:
                    logger.warning(f"Dropped {num_invalid} rows with invalid sentiment from training data.")
                # Map the remaining valid sentiments
                df['label'] = df['sentiment'].map(sentiment_map)
                # Ensure the label column is integer
                df['label'] = df['label'].astype(int)
                logger.info("Mapped 'sentiment' to numerical 'label' (0, 1, 2) for training data.")
            else:
                # For test data, map valid ones, assign -1 to invalid ones
                df['label'] = df['sentiment'].map(sentiment_map)
                df.loc[invalid_sentiment_mask, 'label'] = -1 # Assign -1 only to invalid rows
                df['label'] = df['label'].astype(int)
                logger.info("Mapped 'sentiment' to numerical 'label' (0, 1, 2, or -1 for invalid) for test data.")

        elif is_train_data:
            logger.error(f"'sentiment' column is required but missing in training file: {file_path}")
            raise ValueError(f"Training data file {os.path.basename(file_path)} lacks 'sentiment' column.")
        else:
            logger.warning(f"'sentiment' column not found in {os.path.basename(file_path)}. Assigning placeholder label -1.")
            df['label'] = -1

        # Sample if requested
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} records from {os.path.basename(file_path)}...")
            # Ensure sampling happens *after* cleaning and label assignment
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled DataFrame size: {len(df)}")

        # Final check for label range in training data AFTER all processing
        if is_train_data:
             invalid_labels_mask = ~df['label'].isin([0, 1, 2])
             if invalid_labels_mask.any():
                 num_final_invalid = invalid_labels_mask.sum()
                 logger.error(f"Found {num_final_invalid} rows with invalid labels (outside [0, 1, 2]) in final training data. This should not happen.")
                 logger.error(f"Invalid label values found: {df.loc[invalid_labels_mask, 'label'].unique()}")
                 # Drop these problematic rows definitively
                 df = df[~invalid_labels_mask].copy()
                 logger.warning(f"Dropped {num_final_invalid} rows with invalid final labels from training data.")

        return df

    def load_data(self, sample_size_train: Optional[int] = None, sample_size_test: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and preprocesses train.csv and test.csv directly from the local data directory.
        """
        try:
            train_path = os.path.join(self.data_dir, TRAIN_CSV_NAME)
            test_path = os.path.join(self.data_dir, TEST_CSV_NAME)
            logger.info(f"Attempting to load train data from: {train_path}")
            logger.info(f"Attempting to load test data from: {test_path}")

            train_df = self.load_and_preprocess_csv(train_path, text_column='text', is_train_data=True, sample_size=sample_size_train)
            test_df = self.load_and_preprocess_csv(test_path, text_column='text', is_train_data=False, sample_size=sample_size_test)

            # Column Check
            required_cols_train = ['text', 'text_clean', 'label']
            required_cols_test = ['text', 'text_clean', 'label']
            missing_train = [col for col in required_cols_train if col not in train_df.columns]
            missing_test = [col for col in required_cols_test if col not in test_df.columns]
            if missing_train: raise ValueError(f"Train DataFrame missing required columns: {missing_train}")
            if missing_test: raise ValueError(f"Test DataFrame missing required columns: {missing_test}")

            logger.info(f"Successfully loaded and preprocessed train ({len(train_df)} rows) and test ({len(test_df)} rows) data from local files.")
            # --- Add final check log for training labels ---
            if not train_df.empty:
                 logger.info(f"Final Training Data Label Summary: Min={train_df['label'].min()}, Max={train_df['label'].max()}, Counts=\n{train_df['label'].value_counts()}")
            # --- End final check log ---
            return train_df, test_df

        except FileNotFoundError as e:
            logger.error(f"Data loading failed: {e}. Ensure '{TRAIN_CSV_NAME}' and '{TEST_CSV_NAME}' are present in '{self.data_dir}'.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            raise

    # --- TF-IDF Methods (Unchanged) ---
    def prepare_data_tfidf(self, df_train: pd.DataFrame, df_test: pd.DataFrame, text_column: str = 'text_clean', label_column: str = 'label', max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2, max_df: float = 0.95) -> Dict[str, Union[np.ndarray, TfidfVectorizer]]:
        logger.info("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        X_train = vectorizer.fit_transform(df_train[text_column])
        y_train = df_train[label_column].values
        X_test = vectorizer.transform(df_test[text_column])
        y_test = df_test[label_column].values
        logger.info(f"TF-IDF Data prepared. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'vectorizer': vectorizer}

    def preprocess_new_text_tfidf(self, text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
        if vectorizer is None: raise ValueError("Vectorizer not provided or not fitted.")
        processed_text = self.text_preprocessor.clean_text(text)
        return vectorizer.transform([processed_text])

# --- DataProcessor Class (Wrapper - Unchanged) ---
class DataProcessor:
    def __init__(self, data_dir: str = DATA_DIR, database_path: str = DATABASE_PATH):
        self.data_manager = DataManager(data_dir=data_dir, database_path=database_path)
        self.vectorizer = None
    def load_data(self, sample_size_train: Optional[int] = None, sample_size_test: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.data_manager.load_data(sample_size_train=sample_size_train, sample_size_test=sample_size_test)
    def prepare_data_tfidf(self, *args, **kwargs):
        results = self.data_manager.prepare_data_tfidf(*args, **kwargs)
        self.vectorizer = results['vectorizer']
        return results
    def preprocess_new_text_tfidf(self, text: str) -> np.ndarray:
        if self.vectorizer is None: raise ValueError("TF-IDF Vectorizer not initialized. Call prepare_data_tfidf first.")
        return self.data_manager.preprocess_new_text_tfidf(text, self.vectorizer)
    def get_text_preprocessor(self) -> TextPreprocessor:
         return self.data_manager.text_preprocessor

# Example Usage (Unchanged)
if __name__ == "__main__":
    logger.info("--- Testing DataProcessor ---")
    if not os.path.exists(os.path.join(DATA_DIR, TRAIN_CSV_NAME)) or \
       not os.path.exists(os.path.join(DATA_DIR, TEST_CSV_NAME)):
        logger.error(f"Please download '{TRAIN_CSV_NAME}' and '{TEST_CSV_NAME}' from Kaggle")
        logger.error(f"and place them in the '{DATA_DIR}' directory before running this test.")
    else:
        processor = DataProcessor()
        try:
            train_df_sample, test_df_sample = processor.load_data(sample_size_train=1000, sample_size_test=100)
            logger.info("\n--- Train DataFrame Sample Head ---"); print(train_df_sample.head())
            logger.info("\n--- Test DataFrame Sample Head ---"); print(test_df_sample.head())
            logger.info("\n--- Train DataFrame Info ---"); train_df_sample.info()
            logger.info("\n--- Test DataFrame Info ---"); test_df_sample.info()
            if 'label' in train_df_sample.columns:
                logger.info("\n--- Train Sample Label Distribution ---"); print(train_df_sample['label'].value_counts())
                logger.info(f"Min label: {train_df_sample['label'].min()}, Max label: {train_df_sample['label'].max()}")
            if 'label' in test_df_sample.columns:
                logger.info("\n--- Test Sample Label Distribution ---"); print(test_df_sample['label'].value_counts())
            preprocessor = processor.get_text_preprocessor()
            sample_text = "This is a test tweet! Check out https://example.com @user #testing :) It's not bad, is it?"
            cleaned_text = preprocessor.clean_text(sample_text)
            logger.info(f"\n--- Text Preprocessing Example ---")
            logger.info(f"Original: {sample_text}"); logger.info(f"Cleaned:  {cleaned_text}")
        except Exception as e:
            logger.error(f"Error during testing: {e}", exc_info=True)

