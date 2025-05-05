# src/utils/lstm_roberta_classifier.py
"""
Hybrid LSTM-RoBERTa classifier for sentiment analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
# Ensure AdamW is imported correctly
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn.functional as F
import os # Added for num_workers

# Configure logging if not already done elsewhere
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize the dataset.

        Args:
            texts: List of input texts
            labels: List of sentiment labels (0 for negative, 1 for neutral, 2 for positive) # Updated comment
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

        # --- Add a check for label validity ---
        if not (0 <= label <= 2): # Check if label is 0, 1, or 2
             logger.warning(f"Invalid label found at index {idx}: {label}. Replacing with 1 (Neutral). Text: {text[:100]}...")
             label = 1 # Replace invalid label, e.g., with neutral

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
        use_pooler_output: bool = False,
        num_classes: int = 3 # Added parameter for number of classes
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
            num_classes: Number of output classes (should be 3 for neg/neu/pos)
        """
        super().__init__()
        self.num_classes = num_classes # Store num_classes

        # Load RoBERTa model
        self.roberta = AutoModel.from_pretrained(roberta_model)

        # Freeze RoBERTa parameters if specified
        if freeze_roberta:
            logger.info("Freezing RoBERTa parameters.")
            for param in self.roberta.parameters():
                param.requires_grad = False
        else:
            logger.info("RoBERTa parameters will be fine-tuned (not frozen).")


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
        classifier_input_dim = lstm_output_dim + additional_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.LayerNorm(classifier_input_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, classifier_input_dim // 4),
            nn.LayerNorm(classifier_input_dim // 4) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            # --- Ensure final layer outputs num_classes ---
            nn.Linear(classifier_input_dim // 4, self.num_classes)
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
        if self.use_layer_norm and hasattr(self, 'roberta_layer_norm'):
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
        if self.use_layer_norm and hasattr(self, 'lstm_layer_norm'):
            attended = self.lstm_layer_norm(attended)

        # Global max pooling for sequence representation
        # Mask out padding tokens before max pooling
        # Ensure attention_mask is broadcastable: [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(attended)
        # Set padding tokens to a very small number before max pooling
        attended_masked = attended.masked_fill(mask_expanded == 0, -1e9)
        pooled, _ = torch.max(attended_masked, dim=1)


        # Concatenate with RoBERTa pooler output if enabled
        if self.use_pooler_output and pooler_output is not None:
            pooled = torch.cat([pooled, pooler_output], dim=1)

        # Classify
        logits = self.classifier(pooled)  # [batch_size, num_classes]

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
        max_grad_norm: float = 1.0,
        num_classes: int = 3 # Added num_classes
    ):
        """
        Initialize the classifier.
        Args are documented in the original code.
        """
        self.roberta_model = roberta_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.freeze_roberta = freeze_roberta # Store the freeze setting
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
        self.num_classes = num_classes # Store num_classes

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)

        # Initialize model - pass freeze_roberta setting and num_classes
        self.model = LSTMRoBERTaClassifier(
            roberta_model=roberta_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            freeze_roberta=self.freeze_roberta, # Pass the flag here
            num_attention_heads=num_attention_heads,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            use_pooler_output=use_pooler_output,
            num_classes=self.num_classes # Pass num_classes here
        ).to(self.device)

        # --- MODIFIED OPTIMIZER INITIALIZATION ---
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
                optimizer_grouped_parameters = [
                    {'params': roberta_params, 'lr': self.learning_rate * 0.1},
                    {'params': other_params, 'lr': self.learning_rate}
                ]
                self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=self.weight_decay)
                logger.info(f"Optimizer configured with differential learning rates: RoBERTa LR={self.learning_rate * 0.1}, Head LR={self.learning_rate}")
        else:
            self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
            logger.info(f"Optimizer configured with single learning rate: {self.learning_rate} (RoBERTa frozen).")
        # --- END MODIFIED OPTIMIZER INITIALIZATION ---

        self.scheduler = None
        if use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=scheduler_warmup_steps, num_training_steps=1000000)

        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        logger.info(f"Hybrid Classifier initialized on device: {self.device}")


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
            logger.info(f"Scheduler updated: {total_steps} total steps, {self.scheduler_warmup_steps} warmup steps.")

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        logger.info(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
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
                     # Optionally skip this batch or raise an error
                     # continue
                     raise ValueError(f"Invalid label found in batch {batch_idx} of epoch {epoch+1}")
                # --- End Debugging ---

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None: self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            log_message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}"

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                log_message += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

            logger.info(log_message)

        logger.info("Training finished.")
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

                # --- Debugging: Check labels before eval loss ---
                if labels.min() < 0 or labels.max() >= self.num_classes:
                     logger.error(f"Evaluation: Invalid label detected!")
                     logger.error(f"Labels in batch: {labels.tolist()}")
                     logger.error(f"Min label: {labels.min()}, Max label: {labels.max()}, Num classes: {self.num_classes}")
                     # Decide how to handle: skip batch, error out, etc.
                     continue # Skip batch if invalid label found during eval

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of dictionaries containing predictions and confidence scores
        """
        self.model.eval()
        dataset = SentimentDataset(texts, [-1] * len(texts), self.tokenizer, self.max_length) # Use placeholder label -1
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)

        predictions = []
        # Define the inverse map for prediction output
        label_map_inv = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # Ignore labels from the loader during prediction

                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                confs = probabilities[torch.arange(len(preds)), preds]

                for i in range(len(preds)):
                    pred_class_idx = preds[i].item()
                    confidence = confs[i].item()
                    # Map index back to string label
                    pred_label_str = label_map_inv.get(pred_class_idx, 'Unknown') # Handle potential unexpected index
                    predictions.append({
                        'label': pred_label_str,
                        'confidence': confidence
                    })
        return predictions

    def save_model(self, path: str):
        """Save the model, optimizer state, and architecture config."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'roberta_model': self.roberta_model,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'freeze_roberta': self.freeze_roberta,
                'num_attention_heads': self.num_attention_heads,
                'use_layer_norm': self.use_layer_norm,
                'use_residual': self.use_residual,
                'use_pooler_output': self.use_pooler_output,
                'num_classes': self.num_classes # Save num_classes
            }
        }
        if self.scheduler:
             save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        try:
            torch.save(save_dict, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)


    def load_model(self, path: str):
        """Load the model, optimizer state, and potentially scheduler state."""
        if not os.path.exists(path):
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model checkpoint found at {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            saved_config = checkpoint.get('model_config')
            if not saved_config:
                 logger.warning(f"Checkpoint {path} missing 'model_config'. Attempting load with current defaults.")
                 saved_freeze_roberta = self.freeze_roberta
                 # --- Default num_classes if not found in config ---
                 self.num_classes = getattr(self, 'num_classes', 3)
            else:
                 self.roberta_model = saved_config.get('roberta_model', self.roberta_model)
                 self.hidden_dim = saved_config.get('hidden_dim', self.hidden_dim)
                 self.num_layers = saved_config.get('num_layers', self.num_layers)
                 self.dropout = saved_config.get('dropout', self.dropout)
                 self.freeze_roberta = saved_config.get('freeze_roberta', self.freeze_roberta)
                 self.num_attention_heads = saved_config.get('num_attention_heads', self.num_attention_heads)
                 self.use_layer_norm = saved_config.get('use_layer_norm', self.use_layer_norm)
                 self.use_residual = saved_config.get('use_residual', self.use_residual)
                 self.use_pooler_output = saved_config.get('use_pooler_output', self.use_pooler_output)
                 # --- Load num_classes from saved config ---
                 self.num_classes = saved_config.get('num_classes', 3) # Default to 3 if missing
                 saved_freeze_roberta = self.freeze_roberta
                 logger.info("Model parameters updated from saved config.")

                 # Re-create the model architecture with loaded config (including num_classes)
                 self.model = LSTMRoBERTaClassifier(
                     roberta_model=self.roberta_model, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                     dropout=self.dropout, freeze_roberta=self.freeze_roberta, num_attention_heads=self.num_attention_heads,
                     use_layer_norm=self.use_layer_norm, use_residual=self.use_residual, use_pooler_output=self.use_pooler_output,
                     num_classes=self.num_classes # Pass loaded num_classes
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

            logger.info(f"Model, optimizer, and scheduler (if applicable) loaded successfully from {path}")

        except KeyError as e:
             logger.error(f"Missing key in checkpoint {path}: {e}. Checkpoint might be incompatible or incomplete.", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}", exc_info=True)
            raise
