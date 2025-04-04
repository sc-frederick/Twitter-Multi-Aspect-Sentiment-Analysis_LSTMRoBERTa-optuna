"""
LSTM-based sentiment classifier using word embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from transformers import AutoTokenizer
import torch.nn.functional as F

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            labels: List of sentiment labels (0 for negative, 1 for positive)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and pad/truncate to max_length
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultiHeadAttention(nn.Module):
    """Multi-head attention module for improved feature extraction."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize the multi-head attention module.
        
        Args:
            hidden_dim: Dimension of hidden state
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Attention mask of shape [batch_size, seq_len]
        
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output(context)
        
        return output

class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier with enhanced architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize the LSTM classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_attention_heads: Number of attention heads
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Layer normalization for embeddings
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.emb_layer_norm = nn.LayerNorm(embedding_dim)
        
        self.use_residual = use_residual
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Account for bidirectional in output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Enhanced attention mechanism
        self.attention = MultiHeadAttention(
            hidden_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Layer normalization before classification
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Improved classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.LayerNorm(lstm_output_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.LayerNorm(lstm_output_dim // 4) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, 2)  # 2 classes: negative and positive
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
        
        Returns:
            Tensor of logits for each class
        """
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply layer normalization to embeddings if enabled
        if self.use_layer_norm:
            embedded = self.emb_layer_norm(embedded)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Save original LSTM output for residual connection
        lstm_original = lstm_out
        
        # Apply multi-head attention
        attended = self.attention(lstm_out, attention_mask)
        
        # Apply residual connection if enabled
        if self.use_residual:
            attended = attended + lstm_original
        
        # Apply layer normalization before classification if enabled
        if self.use_layer_norm:
            attended = self.final_layer_norm(attended)
        
        # Global max pooling
        pooled, _ = torch.max(attended * attention_mask.unsqueeze(-1), dim=1)
        
        # Classify
        logits = self.classifier(pooled)  # [batch_size, 2]
        
        return logits

class LSTMSentimentClassifier:
    """Wrapper class for LSTM sentiment classification."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        max_length: int = 128,
        num_epochs: int = 3,
        device: str = None,
        num_attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
        scheduler_warmup_steps: int = 500
    ):
        """
        Initialize the classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
            num_epochs: Number of training epochs
            device: Device to use for training (cpu/cuda)
            num_attention_heads: Number of attention heads
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            weight_decay: Weight decay for regularization
            use_scheduler: Whether to use learning rate scheduler
            scheduler_warmup_steps: Number of warmup steps for scheduler
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_attention_heads = num_attention_heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_warmup_steps = scheduler_warmup_steps
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize model
        self.model = LSTMClassifier(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler if enabled
        self.scheduler = None
        if use_scheduler:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=1000000  # Will be updated during training
            )
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def train(self, train_texts: List[str], train_labels: List[int], 
             val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
        
        Returns:
            Dictionary containing training history
        """
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            # Calculate training metrics
            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_texts is not None and val_labels is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logging.info(f'Epoch {epoch+1}/{self.num_epochs}:')
                logging.info(f'Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}')
                logging.info(f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
            else:
                logging.info(f'Epoch {epoch+1}/{self.num_epochs}:')
                logging.info(f'Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}')
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            List of dictionaries containing predictions and confidence scores
        """
        self.model.eval()
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                
                for probs in probabilities:
                    pred_class = torch.argmax(probs).item()
                    confidence = probs[pred_class].item()
                    
                    predictions.append({
                        'label': 'Positive' if pred_class == 1 else 'Negative',
                        'confidence': confidence
                    })
        
        return predictions
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 