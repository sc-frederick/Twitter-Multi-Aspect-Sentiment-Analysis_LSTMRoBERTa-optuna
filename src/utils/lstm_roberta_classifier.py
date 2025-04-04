"""
Hybrid LSTM-RoBERTa classifier for sentiment analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            labels: List of sentiment labels (0 for negative, 1 for positive)
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

class LSTMRoBERTaClassifier(nn.Module):
    """Hybrid LSTM-RoBERTa model for sentiment classification."""
    
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_roberta: bool = True,
        bidirectional: bool = True,
        num_attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_pooler_output: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            roberta_model: Name/path of the RoBERTa model to use
            hidden_dim: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            freeze_roberta: Whether to freeze RoBERTa parameters
            bidirectional: Whether to use bidirectional LSTM
            num_attention_heads: Number of attention heads
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            use_pooler_output: Whether to use RoBERTa pooler output alongside sequence outputs
        """
        super().__init__()
        
        # Load RoBERTa model
        self.roberta = AutoModel.from_pretrained(roberta_model)
        
        # Freeze RoBERTa parameters if specified
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_pooler_output = use_pooler_output
        
        # Layer normalization for RoBERTa outputs
        if use_layer_norm:
            self.roberta_layer_norm = nn.LayerNorm(self.roberta.config.hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            self.roberta.config.hidden_size,  # RoBERTa hidden size as input
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
        
        # Layer normalization for LSTM outputs
        if use_layer_norm:
            self.lstm_layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Additional feature dimension if using pooler output
        additional_dim = self.roberta.config.hidden_size if use_pooler_output else 0
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim + additional_dim, (lstm_output_dim + additional_dim) // 2),
            nn.LayerNorm((lstm_output_dim + additional_dim) // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((lstm_output_dim + additional_dim) // 2, (lstm_output_dim + additional_dim) // 4),
            nn.LayerNorm((lstm_output_dim + additional_dim) // 4) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((lstm_output_dim + additional_dim) // 4, 2)  # 2 classes: negative and positive
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
        # Get RoBERTa embeddings
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence outputs from RoBERTa
        sequence_output = roberta_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Save pooler output if needed
        pooler_output = roberta_output.pooler_output if self.use_pooler_output else None
        
        # Apply layer normalization to RoBERTa outputs if enabled
        if self.use_layer_norm:
            sequence_output = self.roberta_layer_norm(sequence_output)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(sequence_output)  # [batch_size, seq_len, hidden_dim*2]
        
        # Save original LSTM output for residual connection
        lstm_original = lstm_out
        
        # Apply multi-head attention
        attended = self.attention(lstm_out, attention_mask)
        
        # Apply residual connection if enabled
        if self.use_residual:
            attended = attended + lstm_original
        
        # Apply layer normalization to LSTM outputs if enabled
        if self.use_layer_norm:
            attended = self.lstm_layer_norm(attended)
        
        # Global max pooling for sequence representation
        pooled, _ = torch.max(attended * attention_mask.unsqueeze(-1), dim=1)
        
        # Concatenate with RoBERTa pooler output if enabled
        if self.use_pooler_output:
            pooled = torch.cat([pooled, pooler_output], dim=1)
        
        # Classify
        logits = self.classifier(pooled)  # [batch_size, 2]
        
        return logits

class HybridSentimentClassifier:
    """Wrapper class for hybrid LSTM-RoBERTa sentiment classification."""
    
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 128,
        num_epochs: int = 3,
        freeze_roberta: bool = True,
        device: str = None,
        num_attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_pooler_output: bool = False,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
        scheduler_warmup_steps: int = 500,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize the classifier.
        
        Args:
            roberta_model: Name/path of the RoBERTa model to use
            hidden_dim: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
            num_epochs: Number of training epochs
            freeze_roberta: Whether to freeze RoBERTa parameters
            device: Device to use for training (cpu/cuda)
            num_attention_heads: Number of attention heads
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            use_pooler_output: Whether to use RoBERTa pooler output
            weight_decay: Weight decay for regularization
            use_scheduler: Whether to use learning rate scheduler
            scheduler_warmup_steps: Number of warmup steps for scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for gradient clipping
        """
        self.roberta_model = roberta_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.freeze_roberta = freeze_roberta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_attention_heads = num_attention_heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_pooler_output = use_pooler_output
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.model = LSTMRoBERTaClassifier(
            roberta_model=roberta_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            freeze_roberta=freeze_roberta,
            num_attention_heads=num_attention_heads,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            use_pooler_output=use_pooler_output
        ).to(self.device)
        
        # Initialize optimizer with weight decay and parameter grouping
        # Different learning rates for RoBERTa and other components
        if not freeze_roberta:
            roberta_params = [p for n, p in self.model.named_parameters() if "roberta" in n]
            other_params = [p for n, p in self.model.named_parameters() if "roberta" not in n]
            
            self.optimizer = torch.optim.AdamW([
                {'params': roberta_params, 'lr': learning_rate * 0.1},  # Lower LR for RoBERTa
                {'params': other_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
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
        
        # Update scheduler total steps if enabled
        if self.use_scheduler:
            from transformers import get_linear_schedule_with_warmup
            total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.scheduler_warmup_steps,
                num_training_steps=total_steps
            )
        
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
            
            # For gradient accumulation
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update total loss and accuracy metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Gradient accumulation update
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step if enabled
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
            
            # Process any remaining batches that didn't get included in the last accumulation step
            if len(train_loader) % self.gradient_accumulation_steps != 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step if enabled
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Calculate average loss and accuracy
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            # Log training metrics
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            
            # Evaluate on validation set if provided
            if val_texts is not None and val_labels is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"train_loss={avg_loss:.4f}, train_acc={accuracy:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"train_loss={avg_loss:.4f}, train_acc={accuracy:.4f}"
                )
        
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
                'roberta_model': self.roberta_model,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'freeze_roberta': self.freeze_roberta
            }
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 