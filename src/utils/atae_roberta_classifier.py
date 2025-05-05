# src/utils/atae_roberta_classifier.py
"""
Attention-based LSTM with RoBERTa Embeddings (ATAE-RoBERTa adaptation)
for sentiment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Ensure AdamW is imported correctly
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import numpy as np

# Configure logging if not already done elsewhere
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Re-use the SentimentDataset from the LSTM version as input format is compatible
try:
    from .lstm_roberta_classifier import SentimentDataset # Use the updated one
except ImportError:
    logger.error("Could not import SentimentDataset. Ensure lstm_roberta_classifier.py is accessible.")
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
            # --- Add label check ---
            if not (0 <= label <= 2):
                 logger.warning(f"Invalid label found at index {idx}: {label}. Replacing with 1 (Neutral). Text: {text[:100]}...")
                 label = 1
            return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label, dtype=torch.long)}

class Attention(nn.Module):
    """
    Simple attention mechanism using a learnable context vector.
    Calculates attention weights over a sequence of hidden states.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)
        # Learnable context vector (query for attention)
        self.context_vector = nn.Parameter(torch.Tensor(hidden_dim))
        nn.init.uniform_(self.context_vector, -0.1, 0.1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1) # Softmax over seq_len dim

    def forward(self, lstm_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: Output from LSTM layer [batch_size, seq_len, hidden_dim]
            attention_mask: Mask for padding tokens [batch_size, seq_len] (1 for real, 0 for padding)

        Returns:
            Tuple of (context_vector, attention_weights)
            context_vector: Weighted sum of LSTM outputs [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        projected_output = self.tanh(self.attention_layer(lstm_output))
        scores = torch.bmm(projected_output, self.context_vector.unsqueeze(0).repeat(projected_output.size(0), 1).unsqueeze(2)).squeeze(2)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -float('inf'))
        attention_weights = self.softmax(scores)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights


class ATAERoBERTa(nn.Module):
    """
    ATAE-RoBERTa model adapted for sentence-level sentiment classification.
    Uses RoBERTa embeddings -> BiLSTM -> Attention -> Classifier.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        freeze_roberta: bool = True,
        bidirectional_lstm: bool = True,
        use_layer_norm: bool = True,
        num_classes: int = 3 # Added num_classes
    ):
        super().__init__()
        self.num_classes = num_classes # Store num_classes

        self.roberta = AutoModel.from_pretrained(roberta_model)
        roberta_hidden_size = self.roberta.config.hidden_size

        if freeze_roberta:
            logger.info("Freezing RoBERTa parameters.")
            for param in self.roberta.parameters():
                param.requires_grad = False
        else:
            logger.info("RoBERTa parameters will be fine-tuned (not frozen).")

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.roberta_layer_norm = nn.LayerNorm(roberta_hidden_size)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=roberta_hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional_lstm,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional_lstm else lstm_hidden_dim

        if use_layer_norm:
            self.lstm_layer_norm = nn.LayerNorm(lstm_output_dim)

        # Attention Layer
        self.attention = Attention(lstm_output_dim)

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.LayerNorm(lstm_output_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            # --- Ensure final layer outputs num_classes ---
            nn.Linear(lstm_output_dim // 2, self.num_classes)
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
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = roberta_output.last_hidden_state
        if self.use_layer_norm and hasattr(self, 'roberta_layer_norm'):
            sequence_output = self.roberta_layer_norm(sequence_output)
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        if self.use_layer_norm and hasattr(self, 'lstm_layer_norm'):
            lstm_output = self.lstm_layer_norm(lstm_output)
        context_vector, attn_weights = self.attention(lstm_output, attention_mask=attention_mask)
        logits = self.classifier(context_vector)
        return logits


class ATAERoBERTaSentimentClassifier:
    """
    Wrapper class for ATAE-RoBERTa sentiment classification.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 128,
        num_epochs: int = 3,
        freeze_roberta: bool = True,
        device: str = None,
        bidirectional_lstm: bool = True,
        use_layer_norm: bool = True,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
        scheduler_warmup_steps: int = 100,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        num_classes: int = 3 # Added num_classes
    ):
        self.roberta_model = roberta_model
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.freeze_roberta = freeze_roberta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bidirectional_lstm = bidirectional_lstm
        self.use_layer_norm = use_layer_norm
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.num_classes = num_classes # Store num_classes

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.model = ATAERoBERTa(
            roberta_model=roberta_model,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            freeze_roberta=self.freeze_roberta,
            bidirectional_lstm=bidirectional_lstm,
            use_layer_norm=use_layer_norm,
            num_classes=self.num_classes # Pass num_classes
        ).to(self.device)

        # --- MODIFIED OPTIMIZER INITIALIZATION (Re-used) ---
        if not self.freeze_roberta:
            roberta_params = []
            other_params = []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if "roberta" in n: roberta_params.append(p)
                    else: other_params.append(p)
            if not roberta_params:
                 logger.warning("Differential LR enabled, but no RoBERTa parameters found requiring gradients.")
                 self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
                 logger.info("Optimizer configured with single learning rate (RoBERTa params not trainable).")
            else:
                optimizer_grouped_parameters = [{'params': roberta_params, 'lr': self.learning_rate * 0.1}, {'params': other_params, 'lr': self.learning_rate}]
                self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=self.weight_decay)
                logger.info(f"Optimizer configured with differential learning rates: RoBERTa LR={self.learning_rate * 0.1}, Head LR={self.learning_rate}")
        else:
            self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
            logger.info(f"Optimizer configured with single learning rate: {self.learning_rate} (RoBERTa frozen).")
        # --- END MODIFIED OPTIMIZER INITIALIZATION ---

        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"ATAE-RoBERTa Classifier initialized on device: {self.device}")


    def train(self, train_texts: List[str], train_labels: List[int],
             val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, List[float]]:
        """Train the model."""
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers)

        if self.use_scheduler:
            total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.scheduler_warmup_steps, num_training_steps=total_steps)
            logger.info(f"Scheduler initialized: {total_steps} total steps, {self.scheduler_warmup_steps} warmup.")

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        logger.info(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # --- Debugging: Check labels before loss ---
                if labels.min() < 0 or labels.max() >= self.num_classes:
                     logger.error(f"Epoch {epoch+1}, Batch {batch_idx}: Invalid label detected!")
                     logger.error(f"Labels in batch: {labels.tolist()}")
                     logger.error(f"Min label: {labels.min()}, Max label: {labels.max()}, Num classes: {self.num_classes}")
                     raise ValueError(f"Invalid label found in batch {batch_idx} of epoch {epoch+1}")
                # --- End Debugging ---

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None: self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            log_message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"

            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                log_message += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"

            logger.info(log_message)

        logger.info("Training finished.")
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

                # --- Debugging: Check labels before eval loss ---
                if labels.min() < 0 or labels.max() >= self.num_classes:
                     logger.error(f"Evaluation: Invalid label detected!")
                     logger.error(f"Labels in batch: {labels.tolist()}")
                     logger.error(f"Min label: {labels.min()}, Max label: {labels.max()}, Num classes: {self.num_classes}")
                     continue # Skip batch

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy


    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on new texts."""
        self.model.eval()
        dataset = SentimentDataset(texts, [-1] * len(texts), self.tokenizer, self.max_length) # Placeholder label
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)

        predictions = []
        label_map_inv = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                confs = probabilities[torch.arange(len(preds)), preds]

                for i in range(len(preds)):
                    pred_class_idx = preds[i].item()
                    confidence = confs[i].item()
                    pred_label_str = label_map_inv.get(pred_class_idx, 'Unknown')
                    predictions.append({
                        'label': pred_label_str,
                        'confidence': confidence
                    })
        return predictions


    def save_model(self, path: str):
        """Save the model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
            if self.scheduler: save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
            save_dict['model_config'] = {
                'roberta_model': self.roberta_model, 'lstm_hidden_dim': self.lstm_hidden_dim, 'lstm_layers': self.lstm_layers,
                'dropout': self.dropout, 'freeze_roberta': self.freeze_roberta, 'bidirectional_lstm': self.bidirectional_lstm,
                'use_layer_norm': self.use_layer_norm, 'num_classes': self.num_classes # Save num_classes
            }
            torch.save(save_dict, path)
            logger.info(f"ATAE-RoBERTa model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save ATAE-RoBERTa model to {path}: {e}", exc_info=True)


    def load_model(self, path: str):
        """Load the model and optimizer state."""
        if not os.path.exists(path):
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No ATAE-RoBERTa model checkpoint found at {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            saved_config = checkpoint.get('model_config')
            if not saved_config:
                 logger.warning(f"Checkpoint {path} missing 'model_config'. Attempting load with current defaults.")
                 saved_freeze_roberta = self.freeze_roberta
                 self.num_classes = getattr(self, 'num_classes', 3) # Default if missing
            else:
                 self.roberta_model = saved_config.get('roberta_model', self.roberta_model)
                 self.lstm_hidden_dim = saved_config.get('lstm_hidden_dim', self.lstm_hidden_dim)
                 self.lstm_layers = saved_config.get('lstm_layers', self.lstm_layers)
                 self.dropout = saved_config.get('dropout', self.dropout)
                 self.freeze_roberta = saved_config.get('freeze_roberta', self.freeze_roberta)
                 self.bidirectional_lstm = saved_config.get('bidirectional_lstm', self.bidirectional_lstm)
                 self.use_layer_norm = saved_config.get('use_layer_norm', self.use_layer_norm)
                 self.num_classes = saved_config.get('num_classes', 3) # Load num_classes, default 3
                 saved_freeze_roberta = self.freeze_roberta
                 logger.info("Model parameters updated from saved config.")

                 # Re-create the model architecture with loaded config (including num_classes)
                 self.model = ATAERoBERTa(
                     roberta_model=self.roberta_model, lstm_hidden_dim=self.lstm_hidden_dim, lstm_layers=self.lstm_layers,
                     dropout=self.dropout, freeze_roberta=self.freeze_roberta, bidirectional_lstm=self.bidirectional_lstm,
                     use_layer_norm=self.use_layer_norm, num_classes=self.num_classes # Pass loaded num_classes
                 ).to(self.device)
                 logger.info(f"Model architecture re-created from saved config (num_classes={self.num_classes}).")

            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Re-create optimizer AFTER model is potentially recreated
            if not saved_freeze_roberta:
                roberta_params = [p for n, p in self.model.named_parameters() if "roberta" in n and p.requires_grad]
                other_params = [p for n, p in self.model.named_parameters() if "roberta" not in n and p.requires_grad]
                optimizer_grouped_parameters = [{'params': roberta_params, 'lr': self.learning_rate * 0.1}, {'params': other_params, 'lr': self.learning_rate}]
                self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=self.weight_decay)
                logger.info("Optimizer re-created with differential learning rates (based on loaded state).")
            else:
                self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
                logger.info("Optimizer re-created with single learning rate (based on loaded state).")

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                 total_steps_placeholder = 1000
                 self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.scheduler_warmup_steps, num_training_steps=total_steps_placeholder)
                 self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 logger.info("Scheduler state loaded.")

            logger.info(f"ATAE-RoBERTa model, optimizer, and scheduler (if applicable) loaded successfully from {path}")

        except KeyError as e:
             logger.error(f"Missing key in checkpoint {path}: {e}. Checkpoint might be incompatible or incomplete.", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"Failed to load ATAE-RoBERTa model from {path}: {e}", exc_info=True)
            raise
