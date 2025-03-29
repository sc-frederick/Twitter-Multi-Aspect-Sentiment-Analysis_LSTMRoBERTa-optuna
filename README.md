# Sentiment Analysis Project

This project performs advanced sentiment analysis on tweets using state-of-the-art NLP techniques. It incorporates both traditional machine learning approaches and modern transformer-based models to achieve comprehensive multi-aspect sentiment analysis.

## Project Structure

```
.
├── src/
│   ├── data_processor.py     # Data loading and preprocessing
│   ├── model.py              # ML models for sentiment analysis
│   ├── transformers.py       # Transformer-based models (BERT, RoBERTa)
│   ├── aspect_analyzer.py    # Multi-aspect sentiment analysis
│   ├── main.py               # Main execution script
│   └── misc/                 # Miscellaneous helper scripts
├── sentiment_analysis_env/   # Virtual environment (not in repo)
├── models/                   # Saved model checkpoints
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Features

### Advanced Sentiment Analysis Capabilities

This project implements a comprehensive sentiment analysis system with the following capabilities:

1. **Multi-Aspect Sentiment Analysis**
   - Extract sentiment toward specific aspects/topics within tweets
   - Identify both the aspect and the associated sentiment
   - Generate nuanced sentiment profiles rather than binary classifications

2. **Contextual Embeddings with Transformer Models**
   - Fine-tuned BERT, RoBERTa, or XLNet models for sentiment analysis
   - Capture contextual nuances and linguistic subtleties
   - Transfer learning from pre-trained language models

3. **Attention Mechanisms**
   - Focus on relevant parts of tweets for aspect-specific sentiment
   - Visualize attention weights to explain model decisions
   - Improve performance on complex sentiment expressions

### Data Processing Pipeline

The data processing pipeline is implemented using object-oriented programming principles:

- **DataProcessor**: Main class for managing the entire data processing pipeline
- **DataManager**: Handles database operations (loading, saving, and querying data)
- **TextPreprocessor**: Manages text preprocessing with advanced NLP techniques

### Text Processing Techniques

The `TextPreprocessor` class implements several advanced NLP techniques:

1. **Tokenization**: Breaking text into individual tokens using NLTK
2. **Normalization**: Converting text to lowercase, expanding contractions
3. **Cleaning**: Removing URLs, user mentions, hashtags, and special characters
4. **HTML Decoding**: Converting HTML entities to their corresponding characters
5. **Stopword Removal**: Removing common words that don't contribute to sentiment
6. **Lemmatization**: Reducing words to their base form for better analysis
7. **Negation Handling**: Special treatment for negated terms (e.g., "not good" → "NEG_good")
8. **Punctuation & Number Removal**: Stripping punctuation and numbers from text

### Database Management

The `DataManager` class handles persistent storage of the dataset:

1. **SQLite Integration**: Stores data in a SQLite database for faster access
2. **Schema Management**: Defines and creates the database schema
3. **Data Loading**: Retrieves data efficiently from the database
4. **Data Saving**: Stores processed data for future use

### Model Architecture

The project implements multiple model architectures:

1. **Traditional Neural Network**:
   - Dense layers with ReLU activation
   - Dropout for regularization
   - Early stopping to prevent overfitting

2. **Transformer-Based Models**:
   - BERT with fine-tuning for sentiment classification
   - RoBERTa for enhanced language understanding
   - XLNet for capturing bidirectional contexts

3. **Multi-Aspect Models**:
   - Aspect extraction using named entity recognition
   - Aspect-sentiment pair classification
   - Attention mechanisms to focus on specific aspects

## Setup

1. Create a virtual environment:
```bash
python -m venv sentiment_analysis_env
source sentiment_analysis_env/bin/activate  # On Windows: sentiment_analysis_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API:
- Place your Kaggle API credentials in `.env` or `kaggle.json` in the project root
- Make sure `kaggle.json` is in your `.gitignore`

## Usage

### Basic Sentiment Analysis

```python
from src.data_processor import DataProcessor
from src.model import SentimentModel

# Initialize data processor and load data
data_processor = DataProcessor()
df = data_processor.load_data(sample_size=10000)

# Prepare data for training
data = data_processor.prepare_data(df)

# Train basic sentiment model
model = SentimentModel(input_dim=data['X_train'].shape[1])
history = model.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

# Make prediction
text = "This movie was absolutely amazing! I loved every minute of it."
prediction = model.predict(data_processor.preprocess_new_text(text))
```

### Multi-Aspect Sentiment Analysis

```python
from src.aspect_analyzer import AspectSentimentAnalyzer
from src.transformers import BERTSentimentModel

# Initialize aspect analyzer
analyzer = AspectSentimentAnalyzer()

# Load pre-trained BERT model
bert_model = BERTSentimentModel.load("models/bert_sentiment.h5")

# Analyze text with multiple aspects
text = "The interface is beautiful but the performance is terrible."
aspects = analyzer.extract_aspects(text)
# Output: ['interface', 'performance']

sentiment_results = analyzer.analyze_aspects(text, aspects, bert_model)
# Output: {'interface': 'positive', 'performance': 'negative'}
```

## Running the Project

To run the complete pipeline:
```bash
python src/main.py
```

This will:
1. Load the dataset from Kaggle or existing SQLite database
2. Preprocess the text data
3. Train both traditional and transformer-based models
4. Perform multi-aspect sentiment analysis
5. Evaluate and visualize the results

## Implementation Details

### Multi-Aspect Sentiment Analysis

The multi-aspect sentiment analysis works by:
1. Extracting candidate aspects using named entity recognition and dependency parsing
2. Identifying relationships between aspects and sentiment expressions
3. Applying attention mechanisms to focus on relevant parts of text for each aspect
4. Generating aspect-specific sentiment labels

### Transformer Models

The transformer-based models incorporate:
1. Pre-trained language models as the foundation
2. Fine-tuning on domain-specific data for sentiment analysis
3. Custom classification layers for sentiment prediction
4. Attention visualization for model explainability 