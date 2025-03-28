# Sentiment Analysis Project

This project performs sentiment analysis on tweets using machine learning techniques. It includes data preprocessing, feature extraction, and model training capabilities.

## Project Structure

```
.
├── src/
│   ├── data_processor.py  # Data loading and preprocessing
│   └── main.py           # Main execution script
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API:
- Place your Kaggle API credentials in `kaggle.json` in the project root
- Make sure `kaggle.json` is in your `.gitignore`

## Usage

Run the main script to test data loading and preprocessing:
```bash
python src/main.py
```

## Features

- Data loading from Kaggle with sampling capability
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF feature extraction
- Train-test split functionality
- Comprehensive logging

## Data Processing Pipeline

1. Load data from Kaggle
2. Preprocess text:
   - Convert to lowercase
   - Tokenize
   - Remove stopwords
   - Lemmatize
3. Create TF-IDF features
4. Split data into training and test sets

## Next Steps

- Implement model training with TensorFlow
- Add model evaluation metrics
- Create visualization utilities
- Add cross-validation
- Implement model saving and loading 