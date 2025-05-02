# src/utils/gnn_roberta_classifier.py
"""
Graph Neural Network (GNN) inspired classifier combined with RoBERTa embeddings
for sentiment classification. Uses a GAT-like attention mechanism over token sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import numpy as np

# Re-use the SentimentDataset from the LSTM version as input format is compatible
try:
    from .lstm_roberta_classifier import SentimentDataset
except ImportError:
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

class SimpleGATLayer(nn.Module):
    """
    A simplified Graph Attention Network (GAT)-like layer operating on sequences.
    Each token attends to all other tokens based on learnable weights.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, dropout: float, alpha: float = 0.2):
        """
        Args:
            in_features: Dimension of input features (e.g., RoBERTa hidden size)
            out_features: Dimension of output features per head
            n_heads: Number of attention heads
            dropout: Dropout rate for attention weights and output
            alpha: Negative slope for LeakyReLU activation
        """
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.in_features = in_features

        # Linear transformation for each head (applied to all nodes)
        self.W = nn.Parameter(torch.zeros(size=(in_features, n_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters for each head
        self.a = nn.Parameter(torch.zeros(size=(n_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: Input node features [batch_size, seq_len, in_features]
            attention_mask: Mask for padding tokens [batch_size, seq_len] (1 for real, 0 for padding)

        Returns:
            Output node features after attention [batch_size, seq_len, n_heads * out_features]
        """
        batch_size, seq_len, _ = h.size()

        # 1. Apply linear transformation W
        # Wh shape: [batch_size, seq_len, n_heads * out_features]
        Wh = torch.matmul(h, self.W)
        # Reshape for multi-head processing: [batch_size, seq_len, n_heads, out_features]
        Wh = Wh.view(batch_size, seq_len, self.n_heads, self.out_features)

        # 2. Compute attention scores (simplified GAT mechanism)
        # Prepare input for attention score calculation: [batch_size, seq_len, seq_len, n_heads, 2 * out_features]
        Wh_repeated_rows = Wh.unsqueeze(2).repeat(1, 1, seq_len, 1, 1)
        Wh_repeated_cols = Wh.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        # Concatenate pairs: [batch_size, seq_len, seq_len, n_heads, 2 * out_features]
        a_input = torch.cat([Wh_repeated_rows, Wh_repeated_cols], dim=-1)

        # Apply attention parameters 'a': [batch_size, seq_len, seq_len, n_heads]
        # self.a shape: [n_heads, 2 * out_features] -> [1, 1, 1, n_heads, 2 * out_features] for matmul
        e = self.leakyrelu(torch.matmul(a_input, self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0).transpose(-1, -2)).squeeze(-1))

        # 3. Apply mask to attention scores
        if attention_mask is not None:
            # Create a mask for the attention matrix: [batch_size, 1, seq_len]
            mask = attention_mask.unsqueeze(1)
            # Apply mask: Set scores for padding tokens to -inf
            e = e.masked_fill(mask.unsqueeze(-1).unsqueeze(-1) == 0, -float('inf')) # Mask rows
            e = e.masked_fill(mask.unsqueeze(1).unsqueeze(-1) == 0, -float('inf')) # Mask columns

        # 4. Normalize attention scores using softmax
        # attention shape: [batch_size, seq_len, seq_len, n_heads]
        attention = F.softmax(e, dim=2) # Softmax over columns (j dimension in GAT paper)
        attention = self.dropout(attention)

        # 5. Apply attention to transformed features
        # Wh shape: [batch_size, seq_len, n_heads, out_features] -> [batch_size, n_heads, seq_len, out_features]
        Wh_permuted = Wh.permute(0, 3, 1, 2)
        # attention shape: [batch_size, seq_len, seq_len, n_heads] -> [batch_size, n_heads, seq_len, seq_len]
        attention_permuted = attention.permute(0, 3, 1, 2)

        # h_prime shape: [batch_size, n_heads, seq_len, out_features]
        h_prime = torch.matmul(attention_permuted, Wh_permuted)
        # Permute back: [batch_size, seq_len, n_heads, out_features]
        h_prime = h_prime.permute(0, 2, 1, 3)
        # Concatenate heads: [batch_size, seq_len, n_heads * out_features]
        h_prime = h_prime.contiguous().view(batch_size, seq_len, self.n_heads * self.out_features)

        return h_prime


class GNNRoBERTa(nn.Module):
    """
    GNN-inspired model using RoBERTa embeddings and a SimpleGATLayer.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        gnn_out_features: int = 128, # Output features per GAT head
        gnn_heads: int = 4,
        gnn_layers: int = 1, # Number of GAT layers
        dropout: float = 0.3,
        freeze_roberta: bool = True,
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

        # GAT Layers
        self.gat_layers = nn.ModuleList()
        current_dim = roberta_hidden_size
        for i in range(gnn_layers):
            self.gat_layers.append(
                SimpleGATLayer(
                    in_features=current_dim,
                    out_features=gnn_out_features,
                    n_heads=gnn_heads,
                    dropout=dropout
                )
            )
            current_dim = gnn_out_features * gnn_heads # Update dimension for next layer
            if use_layer_norm:
                 # Add LayerNorm after each GAT layer (optional)
                 self.gat_layers.append(nn.LayerNorm(current_dim))


        # Final classifier
        # The input dimension depends on the output of the last GAT layer
        final_gat_output_dim = gnn_out_features * gnn_heads
        self.classifier = nn.Sequential(
            nn.Linear(final_gat_output_dim, final_gat_output_dim // 2),
            nn.LayerNorm(final_gat_output_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_gat_output_dim // 2, 2) # Binary classification
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
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = roberta_output.last_hidden_state # [batch_size, seq_len, roberta_hidden_size]

        if self.use_layer_norm and hasattr(self, 'roberta_layer_norm'):
            sequence_output = self.roberta_layer_norm(sequence_output)

        # Pass through GAT layers
        gat_output = sequence_output
        for layer in self.gat_layers:
             if isinstance(layer, SimpleGATLayer):
                 gat_output = layer(gat_output, attention_mask=attention_mask)
             elif isinstance(layer, nn.LayerNorm): # Apply LayerNorm if present
                 gat_output = layer(gat_output)
             else: # Apply activation like ReLU between layers if desired
                 gat_output = F.elu(gat_output) # Example using ELU

        # Aggregate node features (e.g., mean pooling over sequence dimension, ignoring padding)
        # Mask padding tokens before pooling
        masked_gat_output = gat_output * attention_mask.unsqueeze(-1)
        # Sum non-padding tokens and divide by the number of non-padding tokens
        summed_output = masked_gat_output.sum(dim=1)
        num_non_padding = attention_mask.sum(dim=1, keepdim=True)
        # Avoid division by zero for sequences with only padding (shouldn't happen with proper input)
        num_non_padding = torch.clamp(num_non_padding, min=1)
        pooled_output = summed_output / num_non_padding # [batch_size, final_gat_output_dim]

        # Classify the pooled representation
        logits = self.classifier(pooled_output)

        return logits


class GNNRoBERTaSentimentClassifier:
    """
    Wrapper class for GNN-RoBERTa sentiment classification.
    """
    def __init__(
        self,
        roberta_model: str = 'roberta-base',
        gnn_out_features: int = 128, # GAT layer output features per head
        gnn_heads: int = 4,          # Number of GAT heads
        gnn_layers: int = 1,         # Number of GAT layers
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 128,
        num_epochs: int = 3,
        freeze_roberta: bool = True,
        device: str = None,
        use_layer_norm: bool = True,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
        scheduler_warmup_steps: int = 100,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0
    ):
        self.roberta_model = roberta_model
        self.gnn_out_features = gnn_out_features
        self.gnn_heads = gnn_heads
        self.gnn_layers = gnn_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.freeze_roberta = freeze_roberta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_layer_norm = use_layer_norm
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.model = GNNRoBERTa(
            roberta_model=roberta_model,
            gnn_out_features=gnn_out_features,
            gnn_heads=gnn_heads,
            gnn_layers=gnn_layers,
            dropout=dropout,
            freeze_roberta=freeze_roberta,
            use_layer_norm=use_layer_norm
        ).to(self.device)

        # Optimizer setup
        if not freeze_roberta:
            roberta_params = [p for n, p in self.model.named_parameters() if "roberta" in n and p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() if "roberta" not in n and p.requires_grad]
            self.optimizer = torch.optim.AdamW([
                {'params': roberta_params, 'lr': learning_rate * 0.1},
                {'params': other_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate, weight_decay=weight_decay
            )

        self.scheduler = None # Initialized in train()
        self.criterion = nn.CrossEntropyLoss()

        logging.info(f"GNN-RoBERTa Classifier initialized on device: {self.device}")
        logging.info(f"Freeze RoBERTa: {self.freeze_roberta}")
        logging.info(f"GNN Layers: {self.gnn_layers}, Heads: {self.gnn_heads}, Out Features/Head: {self.gnn_out_features}")


    def train(self, train_texts: List[str], train_labels: List[int],
             val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, List[float]]:
        """Train the model."""
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        # Consider using more workers if CPU allows and data loading is a bottleneck
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=min(4, os.cpu_count()))

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=min(4, os.cpu_count()))

        # Initialize scheduler
        if self.use_scheduler:
            total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.scheduler_warmup_steps,
                num_training_steps=total_steps
            )
            logging.info(f"Scheduler initialized: {total_steps} total steps, {self.scheduler_warmup_steps} warmup.")

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        logger = logging.getLogger(__name__)

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
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_samples
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)

            log_message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.scheduler:
                 save_dict['scheduler_state_dict'] = self.scheduler.state_dict()

            # Save architecture config needed for reloading
            save_dict['model_config'] = {
                'roberta_model': self.roberta_model,
                'gnn_out_features': self.gnn_out_features,
                'gnn_heads': self.gnn_heads,
                'gnn_layers': self.gnn_layers,
                'dropout': self.dropout,
                'freeze_roberta': self.freeze_roberta,
                'use_layer_norm': self.use_layer_norm,
            }
            torch.save(save_dict, path)
            logging.info(f"GNN-RoBERTa model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to save GNN-RoBERTa model to {path}: {e}", exc_info=True)


    def load_model(self, path: str):
        """Load the model and optimizer state."""
        if not os.path.exists(path):
            logging.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No GNN-RoBERTa model checkpoint found at {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # --- Critical: Re-instantiate model with saved architecture ---
            # This assumes the __init__ args match the keys in model_config
            saved_config = checkpoint.get('model_config', {})
            if not saved_config:
                 logging.warning(f"Checkpoint {path} missing 'model_config'. Attempting load with current defaults.")
                 # If config is missing, the current instance's parameters MUST match the saved model's architecture
            else:
                 # Update current instance parameters from saved config BEFORE loading state_dict
                 self.roberta_model = saved_config.get('roberta_model', self.roberta_model)
                 self.gnn_out_features = saved_config.get('gnn_out_features', self.gnn_out_features)
                 self.gnn_heads = saved_config.get('gnn_heads', self.gnn_heads)
                 self.gnn_layers = saved_config.get('gnn_layers', self.gnn_layers)
                 self.dropout = saved_config.get('dropout', self.dropout)
                 self.freeze_roberta = saved_config.get('freeze_roberta', self.freeze_roberta)
                 self.use_layer_norm = saved_config.get('use_layer_norm', self.use_layer_norm)

                 # Re-create the model architecture based on loaded config
                 self.model = GNNRoBERTa(
                     roberta_model=self.roberta_model,
                     gnn_out_features=self.gnn_out_features,
                     gnn_heads=self.gnn_heads,
                     gnn_layers=self.gnn_layers,
                     dropout=self.dropout,
                     freeze_roberta=self.freeze_roberta, # Use loaded freeze state
                     use_layer_norm=self.use_layer_norm
                 ).to(self.device)
                 logging.info("Model architecture re-created from saved config.")


            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # --- Re-create optimizer AFTER model is potentially recreated ---
            # This ensures optimizer parameters match the potentially updated model structure/freeze state
            if not self.freeze_roberta:
                roberta_params = [p for n, p in self.model.named_parameters() if "roberta" in n and p.requires_grad]
                other_params = [p for n, p in self.model.named_parameters() if "roberta" not in n and p.requires_grad]
                self.optimizer = torch.optim.AdamW([
                    {'params': roberta_params, 'lr': self.learning_rate * 0.1}, # Use current LR settings
                    {'params': other_params, 'lr': self.learning_rate}
                ], weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.learning_rate, weight_decay=self.weight_decay
                )
            logging.info("Optimizer re-created based on loaded model state.")


            # Load optimizer state dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state if available and needed
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                # Re-initialize scheduler before loading state if needed (depends on scheduler type)
                 total_steps_placeholder = 1000 # Placeholder, actual steps depend on context
                 self.scheduler = get_linear_schedule_with_warmup(
                     self.optimizer,
                     num_warmup_steps=self.scheduler_warmup_steps, # Use current settings
                     num_training_steps=total_steps_placeholder
                 )
                 self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 logging.info("Scheduler state loaded.")


            logging.info(f"GNN-RoBERTa model, optimizer, and scheduler (if applicable) loaded successfully from {path}")

        except KeyError as e:
             logging.error(f"Missing key in checkpoint {path}: {e}. Checkpoint might be incompatible or incomplete.", exc_info=True)
             raise
        except Exception as e:
            logging.error(f"Failed to load GNN-RoBERTa model from {path}: {e}", exc_info=True)
            raise
