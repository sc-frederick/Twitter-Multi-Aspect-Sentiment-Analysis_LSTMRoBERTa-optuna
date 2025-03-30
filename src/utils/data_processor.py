"""
Data processing module for sentiment analysis project.
Handles data loading, preprocessing, and feature extraction.
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
from nltk.stem import WordNetLemmatizer, PorterStemmer
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get downloads directory path
DOWNLOADS_DIR = str(Path.home() / "Downloads")
DATABASE_PATH = os.path.join(DOWNLOADS_DIR, "tweets_dataset.db")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class TextPreprocessor:
    """Class for text preprocessing and cleaning operations."""
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False,
                 fix_contractions: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 handle_negation: bool = True,
                 html_decode: bool = True):
        """
        Initialize with preprocessing options.
        
        Args:
            remove_urls: Whether to remove URLs from text
            remove_mentions: Whether to remove user mentions from text
            remove_hashtags: Whether to remove hashtags from text
            fix_contractions: Whether to expand contractions (e.g., "don't" -> "do not")
            remove_punctuation: Whether to remove punctuation from text
            lowercase: Whether to convert text to lowercase
            remove_numbers: Whether to remove numbers from text
            remove_stopwords: Whether to remove stopwords from text
            lemmatize: Whether to lemmatize words
            handle_negation: Whether to handle negation in text
            html_decode: Whether to decode HTML entities
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
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Keep some stopwords for negation handling
        self.negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}
        if self.handle_negation:
            self.stop_words = self.stop_words.difference(self.negation_words)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using multiple preprocessing techniques.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # HTML decode
        if self.html_decode:
            text = html.unescape(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags or just the # symbol
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = re.sub(r'#', '', text)
        
        # Fix contractions
        if self.fix_contractions:
            text = contractions.fix(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_text_negation(self, tokens: List[str]) -> List[str]:
        """
        Mark words that follow negation words with NEG_ prefix.
        
        Args:
            tokens: List of tokenized words
            
        Returns:
            Tokens with negation handled
        """
        negated = False
        processed_tokens = []
        
        for token in tokens:
            if token in self.negation_words:
                negated = True
                processed_tokens.append(token)
            elif token in {'.', '!', '?', ';', ':', ','}:
                # Reset negation at punctuation
                negated = False
                processed_tokens.append(token)
            elif negated:
                processed_tokens.append(f"NEG_{token}")
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Handle negation
        if self.handle_negation:
            tokens = self.handle_text_negation(tokens)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)


class DataManager:
    """Class for data loading, saving, and database operations."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        """
        Initialize with database path.
        
        Args:
            database_path: Path to SQLite database
        """
        self.database_path = database_path
        
    def setup_database(self) -> None:
        """Create SQLite database and table if they don't exist."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create tweets table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target REAL NOT NULL,
            text TEXT NOT NULL,
            text_clean TEXT,
            text_length INTEGER,
            text_clean_length INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database setup complete at {self.database_path}")
    
    def save_to_database(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to SQLite database.
        
        Args:
            df: DataFrame to save
        """
        conn = sqlite3.connect(self.database_path)
        
        # Save to database
        df.to_sql('tweets', conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"Data saved to database at {self.database_path}")
    
    def load_from_database(self) -> pd.DataFrame:
        """
        Load data from SQLite database.
        
        Returns:
            DataFrame containing the loaded data
        """
        conn = sqlite3.connect(self.database_path)
        df = pd.read_sql_query("SELECT * FROM tweets", conn)
        conn.close()
        return df


class DataProcessor:
    """Class for sentiment analysis data processing."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        """
        Initialize with database path and preprocessing options.
        
        Args:
            database_path: Path to SQLite database
        """
        self.data_manager = DataManager(database_path)
        self.text_preprocessor = TextPreprocessor()
        self.vectorizer = None
    
    def load_data(self, sample_size: Optional[int] = None, force_download: bool = False) -> pd.DataFrame:
        """
        Load data from SQLite database or Kaggle API if not available.
        
        Args:
            sample_size: Number of samples to load. If None, loads full dataset.
            force_download: Whether to force download from Kaggle even if database exists.
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # Check if database exists and not forcing download
            if os.path.exists(self.data_manager.database_path) and not force_download:
                logger.info("Loading data from existing database...")
                df = self.data_manager.load_from_database()
            else:
                logger.info("Database not found or force download enabled. Downloading from Kaggle...")
                import kagglehub
                
                # Download dataset and get the path
                path = kagglehub.dataset_download("zphudzz/tweets-clean-posneg-v1")
                logger.info(f"Dataset downloaded to: {path}")
                
                # Handle path which might be a list or string
                if isinstance(path, list):
                    path = path[0]
                
                # Log directory contents
                logger.info(f"Contents of directory {path}:")
                for item in os.listdir(path):
                    logger.info(f"- {item}")
                
                # Try different possible CSV filenames
                possible_csv_names = ['tweets.csv', 'data.csv', 'dataset.csv', 'sentiment.csv', 
                                     'final_clean_no_neutral_no_duplicates.csv']
                csv_path = None
                
                for csv_name in possible_csv_names:
                    temp_path = os.path.join(path, csv_name)
                    if os.path.exists(temp_path):
                        csv_path = temp_path
                        break
                
                if not csv_path:
                    # If no exact match, try to find any CSV file
                    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    if csv_files:
                        csv_path = os.path.join(path, csv_files[0])
                
                if not csv_path:
                    raise FileNotFoundError(f"No CSV file found in directory: {path}")
                
                logger.info(f"Reading CSV file from: {csv_path}")
                
                # Read the CSV file
                df = pd.read_csv(csv_path)
                logger.info(f"Successfully read CSV with columns: {df.columns.tolist()}")
                
                # Ensure we have the columns we need
                # If 'target' doesn't exist but 'sentiment' does, rename it
                if 'target' not in df.columns and 'sentiment' in df.columns:
                    df = df.rename(columns={'sentiment': 'target'})
                elif 'target' not in df.columns and 'label' in df.columns:
                    df = df.rename(columns={'label': 'target'})
                
                # If needed columns already exist, use them
                if not all(col in df.columns for col in ['text_clean', 'text_length', 'text_clean_length']):
                    logger.info("Adding missing columns...")
                    
                    # Add missing columns if needed
                    if 'text_clean' not in df.columns:
                        df['text_clean'] = df['text'].apply(self.text_preprocessor.clean_text)
                    
                    if 'text_length' not in df.columns:
                        df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))
                    
                    if 'text_clean_length' not in df.columns:
                        df['text_clean_length'] = df['text_clean'].astype(str).apply(lambda x: len(x.split()))
                
                # Setup database and save data
                self.data_manager.setup_database()
                self.data_manager.save_to_database(df)
                logger.info("Data saved to database for future use")
            
            # Ensure required columns exist
            required_columns = ['target', 'text', 'text_clean', 'text_length', 'text_clean_length']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            if sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} records from the dataset")
            
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    text_column: str = 'text_clean',
                    label_column: str = 'target',
                    test_size: float = 0.2,
                    max_features: int = 5000,
                    ngram_range: Tuple[int, int] = (1, 2),
                    min_df: int = 2,
                    max_df: float = 0.95) -> Dict[str, Union[np.ndarray, TfidfVectorizer]]:
        """
        Prepare data for training by preprocessing text and creating TF-IDF features.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of data to use for testing
            max_features: Maximum number of features for TF-IDF vectorization
            ngram_range: Range of n-grams to include
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, and vectorizer
        """
        # Create TF-IDF features
        logger.info("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[label_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data split complete. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'vectorizer': self.vectorizer
        }
    
    def preprocess_new_text(self, text: str) -> np.ndarray:
        """
        Preprocess new text for prediction.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Vectorized text ready for model prediction
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized. Call prepare_data first.")
        
        # Clean and preprocess text
        processed_text = self.text_preprocessor.preprocess_text(text)
        
        # Vectorize
        return self.vectorizer.transform([processed_text])
        
    def vectorize_texts(self, texts: List[str]) -> np.ndarray:
        """
        Vectorize a list of preprocessed texts using TF-IDF.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Vectorized texts ready for model training or prediction
        """
        if self.vectorizer is None:
            # Initialize vectorizer if it doesn't exist
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            return self.vectorizer.fit_transform(texts)
        else:
            # Use existing vectorizer
            return self.vectorizer.transform(texts)
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply text preprocessing.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        return self.text_preprocessor.preprocess_text(text) 