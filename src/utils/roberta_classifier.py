"""
Transformer-based models for sentiment analysis.
Implements RoBERTa and other transformer models.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import transformers
from transformers import TFRobertaModel, RobertaTokenizer, TFRobertaForSequenceClassification
from transformers import TFBertModel, BertTokenizer
from typing import Dict, List, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set transformers logging level to reduce verbosity
transformers.logging.set_verbosity_error()

class RobertaClassifier:
    """Sentiment classifier using the RoBERTa model."""
    
    def __init__(self, 
                 model_name: str = 'roberta-base',
                 num_classes: int = 2,
                 max_length: int = 128,
                 learning_rate: float = 2e-5):
        """
        Initialize the RoBERTa classifier.
        
        Args:
            model_name: Name of the pre-trained RoBERTa model
            num_classes: Number of output classes
            max_length: Maximum sequence length for tokenization
            learning_rate: Learning rate for fine-tuning
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.learning_rate = learning_rate
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Build a RoBERTa model for sequence classification.
        
        Returns:
            TensorFlow model with RoBERTa base and classification head
        """
        # Create pretrained model with classification head
        model = TFRobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def prepare_inputs(self, texts: List[str]) -> Dict[str, tf.Tensor]:
        """
        Tokenize and prepare inputs for the model.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary of model inputs
        """
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Convert to tensor dict to ensure compatibility
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def train(self, 
              train_texts: List[str],
              train_labels: np.ndarray,
              val_texts: List[str],
              val_labels: np.ndarray,
              epochs: int = 3,
              batch_size: int = 16) -> tf.keras.callbacks.History:
        """
        Fine-tune the RoBERTa model on training data.
        
        Args:
            train_texts: List of training text strings
            train_labels: Training labels
            val_texts: List of validation text strings
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        logger.info(f"Preparing training data with {len(train_texts)} examples...")
        train_inputs = self.prepare_inputs(train_texts)
        
        logger.info(f"Preparing validation data with {len(val_texts)} examples...")
        val_inputs = self.prepare_inputs(val_texts)
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        # Create TF datasets to improve compatibility
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_inputs),
            train_labels
        )).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_inputs),
            val_labels
        )).batch(batch_size)
        
        logger.info("Fine-tuning RoBERTa model...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        return history
    
    def evaluate(self, 
                test_texts: List[str],
                test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: List of test text strings
            test_labels: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Preparing test data with {len(test_texts)} examples...")
        test_inputs = self.prepare_inputs(test_texts)
        
        # Create TF dataset for evaluation with smaller batches
        batch_size = min(32, len(test_texts))  # Ensure batch size isn't larger than dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_inputs),
            test_labels
        )).batch(batch_size)
        
        logger.info("Evaluating model...")
        try:
            loss, accuracy = self.model.evaluate(test_dataset)
            
            # Get predictions in smaller batches to avoid memory issues
            predictions = []
            for batch in test_dataset:
                try:
                    batch_pred = self.model.predict(batch[0], verbose=0)
                    predictions.append(batch_pred.logits)
                except Exception as e:
                    logger.error(f"Error during batch prediction: {e}")
                    continue
            
            if not predictions:
                logger.warning("No predictions were made successfully")
                return {
                    'accuracy': accuracy,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
                
            logits = np.vstack(predictions)
            y_pred = np.argmax(logits, axis=1)
            
            # Calculate metrics
            report = classification_report(test_labels[:len(y_pred)], y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(test_labels[:len(y_pred)], y_pred)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment of a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        try:
            # Prepare input
            inputs = self.prepare_inputs([text])
            
            # Create a dataset for prediction
            dataset = tf.data.Dataset.from_tensor_slices((dict(inputs),)).batch(1)
            
            # Get prediction
            predictions = self.model.predict(dataset, verbose=0)
            logits = predictions.logits[0]
            
            # Convert logits to probabilities
            probabilities = tf.nn.softmax(logits).numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0, 0.5  # Return default values
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Directory path to save the model
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RobertaClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Directory path to the saved model
            
        Returns:
            Loaded RobertaClassifier instance
        """
        instance = cls()
        instance.model = TFRobertaForSequenceClassification.from_pretrained(filepath)
        instance.tokenizer = RobertaTokenizer.from_pretrained(filepath)
        logger.info(f"Model loaded from {filepath}")
        return instance


class CustomRoBERTaClassifier:
    """Custom RoBERTa model with custom classification head."""
    
    def __init__(self, 
                 model_name: str = 'roberta-base',
                 num_classes: int = 2,
                 max_length: int = 128,
                 learning_rate: float = 2e-5):
        """
        Initialize the custom RoBERTa classifier.
        
        Args:
            model_name: Name of the pre-trained RoBERTa model
            num_classes: Number of output classes
            max_length: Maximum sequence length for tokenization
            learning_rate: Learning rate for fine-tuning
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.learning_rate = learning_rate
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Build a custom RoBERTa model with classification head.
        
        Returns:
            TensorFlow model with RoBERTa base and custom classification head
        """
        # RoBERTa as base model
        base_model = TFRobertaModel.from_pretrained(self.model_name)
        
        # Input layers
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # Get RoBERTa embeddings
        embeddings = base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        # Use CLS token embedding for classification
        cls_embedding = embeddings[:, 0, :]
        
        # Classification head
        x = Dropout(0.2)(cls_embedding)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        
        # Freeze base model layers (optional)
        # for layer in base_model.layers:
        #     layer.trainable = False
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def prepare_inputs(self, texts: List[str]) -> Dict[str, tf.Tensor]:
        """
        Tokenize and prepare inputs for the model.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary of model inputs
        """
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def train(self, 
              train_texts: List[str],
              train_labels: np.ndarray,
              val_texts: List[str],
              val_labels: np.ndarray,
              epochs: int = 3,
              batch_size: int = 16) -> tf.keras.callbacks.History:
        """
        Fine-tune the RoBERTa model on training data.
        
        Args:
            train_texts: List of training text strings
            train_labels: Training labels
            val_texts: List of validation text strings
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        logger.info(f"Preparing training data with {len(train_texts)} examples...")
        train_inputs = self.prepare_inputs(train_texts)
        
        logger.info(f"Preparing validation data with {len(val_texts)} examples...")
        val_inputs = self.prepare_inputs(val_texts)
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        logger.info("Fine-tuning RoBERTa model...")
        history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history
    
    def evaluate(self, 
                test_texts: List[str],
                test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: List of test text strings
            test_labels: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Preparing test data with {len(test_texts)} examples...")
        test_inputs = self.prepare_inputs(test_texts)
        
        logger.info("Evaluating model...")
        loss, accuracy = self.model.evaluate(test_inputs, test_labels)
        
        # Get predictions
        predictions = self.model.predict(test_inputs)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        report = classification_report(test_labels, y_pred, output_dict=True)
        cm = confusion_matrix(test_labels, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment of a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Prepare input
        inputs = self.prepare_inputs([text])
        
        # Get prediction
        probabilities = self.model.predict(inputs)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Directory path to save the model
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(filepath, 'tokenizer'))
        
        # Save model weights
        self.model.save_weights(os.path.join(filepath, 'model_weights.h5'))
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, model_name: str = 'roberta-base') -> 'CustomRoBERTaClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Directory path to the saved model
            model_name: Name of the pre-trained RoBERTa model
            
        Returns:
            Loaded CustomRoBERTaClassifier instance
        """
        instance = cls(model_name=model_name)
        
        # Load tokenizer
        instance.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(filepath, 'tokenizer'))
        
        # Load model weights
        instance.model.load_weights(os.path.join(filepath, 'model_weights.h5'))
        
        logger.info(f"Model loaded from {filepath}")
        return instance 


class RoBERTaSentimentClassifier:
    """Sentiment classifier using RoBERTa model with a consistent interface."""
    
    def __init__(self, 
                 pretrained_model: str = 'distilroberta-base',
                 max_length: int = 128,
                 num_classes: int = 2):
        """
        Initialize the RoBERTa sentiment classifier.
        
        Args:
            pretrained_model: Name of the pre-trained RoBERTa model
            max_length: Maximum sequence length for tokenization
            num_classes: Number of output classes
        """
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Create the underlying RoBERTa classifier
        self.classifier = RobertaClassifier(
            model_name=pretrained_model,
            num_classes=num_classes,
            max_length=max_length
        )
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 3,
              batch_size: int = 16,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train the model on the given data.
        
        Args:
            X_train: Training text data as numpy array
            y_train: Training labels
            X_val: Validation text data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        # Convert verbose int to bool for transformers
        verbose_bool = verbose > 0
        
        # Train using the underlying classifier
        history = self.classifier.train(
            train_texts=X_train.tolist(),
            train_labels=y_train,
            val_texts=X_val.tolist(),
            val_labels=y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Evaluate using the underlying classifier
        results = self.classifier.evaluate(
            test_texts=X_test.tolist(),
            test_labels=y_test
        )
        
        # Extract and format metrics to match the expected interface
        metrics = {
            'accuracy': results['accuracy'],
            'precision': results['classification_report']['weighted avg']['precision'],
            'recall': results['classification_report']['weighted avg']['recall'],
            'f1_score': results['classification_report']['weighted avg']['f1-score']
        }
        
        return metrics
    
    def predict(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with class and confidence
        """
        # Get prediction from underlying classifier
        predicted_class, confidence = self.classifier.predict(text)
        
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple texts.
        
        Args:
            X: Array of text strings
            
        Returns:
            Array of predicted classes
        """
        predictions = []
        
        # Convert to list if it's a numpy array
        texts = X.tolist() if isinstance(X, np.ndarray) else X
        
        # Prepare inputs for batch prediction
        inputs = self.classifier.prepare_inputs(texts)
        
        # Get predictions
        predictions = self.classifier.model.predict(inputs)
        logits = predictions.logits
        return np.argmax(logits, axis=1)
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.classifier.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'RoBERTaSentimentClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded RoBERTaSentimentClassifier
        """
        instance = cls()
        instance.classifier = RobertaClassifier.load(filepath)
        return instance


def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history: Training history object from model.fit()
    """
    import matplotlib.pyplot as plt
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    # plt.show()  # Comment out to prevent displaying/saving
    plt.close(fig)  # Close the figure to free memory
    
    logger.info("Training history plot created (display disabled)")


def plot_confusion_matrix(cm, classes=None, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names
        normalize: Whether to normalize the values
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if classes is None:
        classes = ['Negative', 'Positive']
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d" if not normalize else ".2f",
                cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.show()  # Comment out to prevent displaying/saving
    plt.close(fig)  # Close the figure to free memory
    
    logger.info("Confusion matrix plot created (display disabled)") 