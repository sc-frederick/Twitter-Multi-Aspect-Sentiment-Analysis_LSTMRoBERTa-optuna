# Sentiment Analysis Project - Source Code

This directory contains the source code for the sentiment analysis project. The code is organized using object-oriented programming (OOP) principles for better maintainability and extensibility.

## Project Structure

The project is structured into several main components:

### Data Processing

- `DataProcessor`: Main class for managing the entire data processing pipeline
- `DataManager`: Handles database operations (loading, saving, and querying data)
- `TextPreprocessor`: Manages text preprocessing with advanced NLP techniques

### Model

- `SentimentModel`: OOP wrapper for the TensorFlow sentiment analysis model
- Functional API for compatibility with existing code

## Text Processing Features

The `TextPreprocessor` class implements several advanced NLP techniques:

1. **Tokenization**: Breaking text into individual tokens using NLTK
2. **Normalization**: Converting text to lowercase, expanding contractions
3. **Cleaning**: Removing URLs, user mentions, hashtags, and special characters
4. **HTML Decoding**: Converting HTML entities to their corresponding characters
5. **Stopword Removal**: Removing common words that don't contribute to sentiment
6. **Lemmatization**: Reducing words to their base form for better analysis
7. **Negation Handling**: Special treatment for negated terms (e.g., "not good" → "NEG_good")
8. **Punctuation & Number Removal**: Stripping punctuation and numbers from text

## Database Management

The `DataManager` class handles persistent storage of the dataset:

1. **SQLite Integration**: Stores data in a SQLite database for faster access
2. **Schema Management**: Defines and creates the database schema
3. **Data Loading**: Retrieves data efficiently from the database
4. **Data Saving**: Stores processed data for future use

## Usage

The main processing flow is:

1. Initialize the `DataProcessor`:
   ```python
   data_processor = DataProcessor()
   ```

2. Load data (downloads from Kaggle if necessary):
   ```python
   df = data_processor.load_data(sample_size=10000)
   ```

3. Prepare data for training:
   ```python
   data = data_processor.prepare_data(df, text_column='text_clean', label_column='target')
   ```

4. Access prepared data:
   ```python
   X_train = data['X_train']
   X_test = data['X_test']
   y_train = data['y_train']
   y_test = data['y_test']
   ```

5. Create and train model:
   ```python
   model = SentimentModel(input_dim=X_train.shape[1])
   history = model.train(X_train, y_train, X_val, y_val)
   ```

6. Evaluate model:
   ```python
   results = model.evaluate(X_test, y_test)
   ```

7. Make predictions on new text:
   ```python
   text_vector = data_processor.preprocess_new_text("This is great!")
   prediction = model.predict(text_vector)
   ``` 