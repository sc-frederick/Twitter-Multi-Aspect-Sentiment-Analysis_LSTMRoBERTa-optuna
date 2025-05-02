# src/utils/han_roberta_classifier.py
"""
Hierarchical Attention Network (HAN) combined with RoBERTa embeddings
for sentiment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import List, Tuple, Dict, Any
import logging
import os
import numpy as np

# Re-use the SentimentDataset from the LSTM version as input format is compatible
# (Requires input_ids and attention_mask)
try:
    # Attempt to import from the existing lstm_roberta_classifier file
    from .lstm_roberta_classifier import SentimentDataset
except ImportError:
    # Fallback or define SentimentDataset here if needed, ensuring compatibility
    # For simplicity, assuming lstm_roberta_classifier.py is in the same directory
    logging.error("Could not import SentimentDataset. Ensure lstm_roberta_classifier.py is accessible.")
    # Define a basic compatible Dataset if import fails (example structure)
    class SentimentDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self) -> int: return len(self.texts)
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            text = str(self.texts[idx]); label = self.labels[idx]
            encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label, dtype=torch.long)}


class AttentionWithContext(nn.Module):
    """
    Simple Attention mechanism with context vector.
    Used for both word and sentence level attention in HAN.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Context vector (learnable)
        self.context_vector = nn.Parameter(torch.Tensor(hidden_dim))
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        # Initialize context vector
        nn.init.uniform_(self.context_vector, -0.1, 0.1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            Tuple of (context_vector, attention_weights)
            context_vector: Tensor of shape [batch_size, hidden_dim]
            attention_weights: Tensor of shape [batch_size, seq_len]
        """
        # Project hidden states: [batch_size, seq_len, hidden_dim]
        projected_hidden = self.tanh(self.linear(hidden_states))

        # Compute attention scores (dot product with context vector)
        # context_vector shape: [hidden_dim] -> unsqueeze for broadcasting
        # scores shape: [batch_size, seq_len]
        scores = torch.matmul(projected_hidden, self.context_vector.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

        # Apply mask (set attention scores for padding tokens to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        # Compute attention weights (softmax)
        # attention_weights shape: [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=1)

        # Compute context vector (weighted sum of hidden states)
        # attention_weights shape: [batch_size, 1, seq_len] for broadcasting
        # context_vector shape: [batch_size, hidden_dim]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)

        return context_vector, attention_weights


class HANRoBERTa(nn.Module):
    """
    Hierarchical Attention Network using RoBERTa embeddings.
    Note: This implementation applies attention directly over RoBERTa's token embeddings.
    It doesn't explicitly model sentences hierarchically in the traditional HAN sense,
    but uses a similar attention mechanism over the sequence.
    A true hierarchical implementation would require sentence segmentation and potentially
    more complex handling of RoBERTa outputs per sentence.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        hidden_dim: int = 256, # Dimension for the GRU/LSTM layer after RoBERTa
        num_layers: int = 1,   # Number of GRU/LSTM layers
        dropout: float = 0.3,
        freeze_roberta: bool = True,
        bidirectional: bool = True,
        use_gru: bool = True, # Use GRU instead of LSTM
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.roberta = AutoModel.from_pretrained(roberta_model)
        roberta_hidden_size = self.roberta.config.hidden_size

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.roberta_layer_norm = nn.LayerNorm(roberta_hidden_size)

        # Recurrent layer (GRU or LSTM) to process RoBERTa sequence outputs
        rnn_input_size = roberta_hidden_size
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if use_gru:
            self.rnn = nn.GRU(
                rnn_input_size, hidden_dim, num_layers=num_layers,
                bidirectional=bidirectional, batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.LSTM(
                rnn_input_size, hidden_dim, num_layers=num_layers,
                bidirectional=bidirectional, batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        if use_layer_norm:
            self.rnn_layer_norm = nn.LayerNorm(rnn_output_dim)

        # Attention layer (applied over RNN outputs)
        self.attention = AttentionWithContext(rnn_output_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, rnn_output_dim // 2),
            nn.LayerNorm(rnn_output_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim // 2, 2) # Binary classification (Negative/Positive)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, num_classes]
        """
        # Get RoBERTa embeddings
        # output is a BaseModelOutputWithPoolingAndCrossAttentions object
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use last_hidden_state: [batch_size, seq_len, roberta_hidden_size]
        sequence_output = roberta_output.last_hidden_state

        if self.use_layer_norm:
            sequence_output = self.roberta_layer_norm(sequence_output)

        # Pass through RNN
        # rnn_output: [batch_size, seq_len, rnn_output_dim]
        rnn_output, _ = self.rnn(sequence_output)

        if self.use_layer_norm:
             rnn_output = self.rnn_layer_norm(rnn_output)

        # Apply attention over RNN outputs
        # Use the original attention_mask from RoBERTa input
        # context_vector: [batch_size, rnn_output_dim]
        # attn_weights: [batch_size, seq_len]
        context_vector, attn_weights = self.attention(rnn_output, mask=attention_mask)

        # Classify the context vector
        logits = self.classifier(context_vector)

        return logits


class HANRoBERTaSentimentClassifier:
    """
    Wrapper class for HAN-RoBERTa sentiment classification.
    Mirrors the structure of HybridSentimentClassifier for consistency.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 128,
        num_epochs: int = 3,
        freeze_roberta: bool = True,
        device: str = None,
        use_gru: bool = True,
        use_layer_norm: bool = True,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
        scheduler_warmup_steps: int = 100, # Adjusted default
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0
    ):
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
        self.use_gru = use_gru
        self.use_layer_norm = use_layer_norm
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.model = HANRoBERTa(
            roberta_model=roberta_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            freeze_roberta=freeze_roberta,
            use_gru=use_gru,
            use_layer_norm=use_layer_norm
        ).to(self.device)

        # Optimizer setup (similar to LSTM version)
        if not freeze_roberta:
            roberta_params = [p for n, p in self.model.named_parameters() if "roberta" in n and p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() if "roberta" not in n and p.requires_grad]
            self.optimizer = torch.optim.AdamW([
                {'params': roberta_params, 'lr': learning_rate * 0.1}, # Lower LR for RoBERTa
                {'params': other_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate, weight_decay=weight_decay
            )

        self.scheduler = None # Will be initialized in train() if use_scheduler is True
        self.criterion = nn.CrossEntropyLoss() # Consider label smoothing if needed

        logging.info(f"HAN-RoBERTa Classifier initialized on device: {self.device}")
        logging.info(f"Freeze RoBERTa: {self.freeze_roberta}")


    def train(self, train_texts: List[str], train_labels: List[int],
             val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, List[float]]:
        """Train the model."""
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=min(4, os.cpu_count()))

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=min(4, os.cpu_count()))

        # Initialize scheduler if enabled
        if self.use_scheduler:
            total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.scheduler_warmup_steps,
                num_training_steps=total_steps
            )
            logging.info(f"Scheduler initialized with {total_steps} total steps and {self.scheduler_warmup_steps} warmup steps.")


        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        logger = logging.getLogger(__name__) # Ensure logger is accessible

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            self.optimizer.zero_grad() # Zero gradients at the start of epoch

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps # Unscale for logging

                # Accuracy calculation
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # Optimizer step
                    self.optimizer.step()
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                    # Zero gradients
                    self.optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_samples
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)

            log_message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"

            # Validation step
            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                log_message += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"

            logger.info(log_message)

        return history


    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy


    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on new texts."""
        self.model.eval()
        # Use the same SentimentDataset structure
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=min(4, os.cpu_count()))

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)

                preds = torch.argmax(probabilities, dim=1)
                confs = probabilities[torch.arange(len(preds)), preds]

                for i in range(len(preds)):
                    pred_class = preds[i].item()
                    confidence = confs[i].item()
                    predictions.append({
                        'label': 'Positive' if pred_class == 1 else 'Negative',
                        'confidence': confidence
                    })
        return predictions


    def save_model(self, path: str):
        """Save the model and optimizer state."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # Save model state, optimizer state, and potentially scheduler state
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.scheduler:
                 save_dict['scheduler_state_dict'] = self.scheduler.state_dict()

            # Include minimal config needed to reconstruct the model architecture
            save_dict['model_config'] = {
                'roberta_model': self.roberta_model,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'freeze_roberta': self.freeze_roberta, # Important for reloading optimizer correctly
                'use_gru': self.use_gru,
                'use_layer_norm': self.use_layer_norm,
                # Add other architectural params if they vary
            }
            torch.save(save_dict, path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to save model to {path}: {e}", exc_info=True)


    def load_model(self, path: str):
        """Load the model and optimizer state."""
        if not os.path.exists(path):
            logging.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model checkpoint found at {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            # Important: Ensure the optimizer was created *before* calling load_model
            # This requires the freeze_roberta state to be correctly set during init
            # based on the config used for the *saved* model.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state if available and needed
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            logging.info(f"Model and optimizer state loaded successfully from {path}")

        except KeyError as e:
             logging.error(f"Missing key in checkpoint {path}: {e}. Checkpoint might be incompatible or incomplete.", exc_info=True)
             raise
        except Exception as e:
            logging.error(f"Failed to load model from {path}: {e}", exc_info=True)
            raise
