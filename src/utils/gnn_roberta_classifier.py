# src/utils/gnn_roberta_classifier.py
# Updated to handle None validation data during final training and 3 output classes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import datetime
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GATLayer Class (Unchanged) ---
class GATLayer(nn.Module):
    """
    Simplified Graph Attention Layer based on https://arxiv.org/abs/1710.10903
    Operates on node features (e.g., token embeddings) rather than a predefined graph structure.
    Assumes full connectivity initially, attention mechanism learns edge weights.
    """
    def __init__(self, in_features, out_features, heads, dropout, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node_features, attention_mask=None):
        batch_size, seq_len, _ = node_features.size()
        Wh = self.W(node_features)
        Wh = Wh.view(batch_size, seq_len, self.heads, self.out_features)
        Wh_i = Wh.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)
        Wh_j = Wh.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)
        concat_features = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.a(concat_features)
        e = e.squeeze(-1)
        e = self.leaky_relu(e)
        e = e.permute(0, 3, 1, 2)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            e = e.masked_fill(mask == 0, -float('inf'))
        attention = self.softmax(e)
        attention = F.dropout(attention, self.dropout, training=self.training)
        Wh_permuted = Wh.permute(0, 2, 1, 3)
        h_prime = torch.matmul(attention, Wh_permuted)
        h_prime = h_prime.permute(0, 2, 1, 3)
        if self.concat:
            out_features = h_prime.reshape(batch_size, seq_len, self.heads * self.out_features)
        else:
            out_features = h_prime.mean(dim=2)
        return out_features


# --- GNN-RoBERTa Model (Updated for num_labels) ---
class GNNRoBERTaModel(nn.Module):
    """
    GNN-RoBERTa model for text classification.
    Combines RoBERTa embeddings with multiple GAT layers.
    """
    def __init__(self, roberta_model_name='roberta-base', gnn_layers=2, gnn_heads=4, gnn_out_features=128, dropout=0.1, num_labels=3, freeze_roberta=False): # Default num_labels=3
        """
        Initializes the GNNRoBERTaModel.

        Args:
            roberta_model_name (str, optional): Name of the pre-trained RoBERTa model. Defaults to 'roberta-base'.
            gnn_layers (int, optional): Number of GAT layers. Defaults to 2.
            gnn_heads (int, optional): Number of attention heads in each GAT layer. Defaults to 4.
            gnn_out_features (int, optional): Output feature dimension per head in GAT layers. Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            num_labels (int, optional): Number of output classes. Defaults to 3.
            freeze_roberta (bool, optional): Whether to freeze RoBERTa layers during training. Defaults to False.
        """
        super(GNNRoBERTaModel, self).__init__()
        self.num_labels = num_labels # Store num_labels
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # GAT Layers
        self.gat_layers = nn.ModuleList()
        gat_input_dim = self.roberta_hidden_size
        for i in range(gnn_layers):
            concat_layer = True # Assuming concatenation for simplicity
            self.gat_layers.append(
                GATLayer(gat_input_dim, gnn_out_features, gnn_heads, dropout, concat=concat_layer)
            )
            if concat_layer: gat_input_dim = gnn_heads * gnn_out_features
            else: gat_input_dim = gnn_out_features

        self.final_gat_dim = gat_input_dim

        # Classifier layer
        self.dropout = nn.Dropout(dropout)
        # --- Ensure final layer outputs num_labels ---
        self.classifier = nn.Linear(self.final_gat_dim, self.num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Performs the forward pass of the GNN-RoBERTa model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Logits for each class (batch_size, num_labels).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        gat_output = sequence_output
        for layer in self.gat_layers:
            gat_output = layer(gat_output, attention_mask=attention_mask)
            gat_output = F.relu(gat_output)
        cls_token_output = gat_output[:, 0, :]
        pooled_output = self.dropout(cls_token_output)
        logits = self.classifier(pooled_output)
        return logits


# --- Classifier Wrapper (Updated for num_labels) ---
class GNNRoBERTaClassifier:
    """
    A wrapper class for training, evaluating, and using the GNNRoBERTaModel.
    Handles tokenization, data loading, training loop, evaluation, and saving/loading.
    """
    def __init__(self, config):
        """
        Initializes the GNNRoBERTaClassifier.

        Args:
            config (dict): Configuration dictionary containing model and training parameters.
                           Expected keys: roberta_model_name, gnn_layers, gnn_heads,
                           gnn_out_features, dropout, num_labels, freeze_roberta,
                           learning_rate, weight_decay, epochs, batch_size,
                           scheduler_warmup_steps, device, max_seq_length.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model_name'])
        self.max_seq_length = config.get('max_seq_length', 128)
        self.epochs = config.get('epochs', 3)
        self.num_classes = config.get('num_labels', 3) # Get num_labels from config, default 3

        self.model = GNNRoBERTaModel(
            roberta_model_name=config['roberta_model_name'],
            gnn_layers=config['gnn_layers'],
            gnn_heads=config['gnn_heads'],
            gnn_out_features=config['gnn_out_features'],
            dropout=config['dropout'],
            num_labels=self.num_classes, # Pass num_classes here
            freeze_roberta=config.get('freeze_roberta', False)
        ).to(self.device)

        logger.info(f"GNN-RoBERTa Classifier initialized on device: {self.device}")
        logger.info(f"Freeze RoBERTa: {config.get('freeze_roberta', False)}")
        logger.info(f"GNN Layers: {config['gnn_layers']}, Heads: {config['gnn_heads']}, Out Features/Head: {config['gnn_out_features']}")
        logger.info(f"Number of output classes: {self.num_classes}")
        logger.info(f"Training Epochs set to: {self.epochs}")

        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}


    def _create_dataloader(self, texts, labels, batch_size, sampler_type='random'):
        """Creates a DataLoader for the given data."""
        if texts is None:
            logger.warning(f"Attempted to create dataloader with None texts. Returning None.")
            return None

        input_ids = []
        attention_masks = []
        valid_labels = []
        valid_indices = []

        for i, (text, label) in enumerate(zip(texts, labels)):
            # --- Add label check during encoding ---
            if not (0 <= label < self.num_classes):
                 logger.warning(f"Invalid label found at index {i}: {label}. Skipping sample. Text: {text[:100]}...")
                 continue # Skip this sample

            encoded_dict = self.tokenizer.encode_plus(
                str(text), # Ensure text is string
                add_special_tokens=True, max_length=self.max_seq_length, padding='max_length',
                truncation=True, return_attention_mask=True, return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            valid_labels.append(label) # Add the valid label
            valid_indices.append(i) # Keep track of original index if needed

        if not input_ids:
             logger.error("No valid samples found after label checking. Cannot create DataLoader.")
             return None

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels_tensor = torch.tensor(valid_labels) # Use only valid labels

        dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

        if sampler_type == 'random': sampler = RandomSampler(dataset)
        elif sampler_type == 'sequential': sampler = SequentialSampler(dataset)
        else: raise ValueError(f"Invalid sampler_type: {sampler_type}")

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        logger.info(f"Created DataLoader with {len(dataset)} valid samples.")
        return dataloader

    def _format_time(self, elapsed):
        """Takes a time in seconds and returns a string hh:mm:ss"""
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self, train_texts, train_labels, val_texts, val_labels):
        """
        Trains the GNN-RoBERTa model. Handles None for val_texts/val_labels.
        """
        batch_size = self.config['batch_size']
        epochs = self.epochs
        learning_rate = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01)
        warmup_steps = self.config.get('scheduler_warmup_steps', 100)

        train_dataloader = self._create_dataloader(train_texts, train_labels, batch_size, sampler_type='random')
        if train_dataloader is None:
             logger.error("Failed to create training dataloader. Aborting training.")
             return self.history # Return empty history

        val_dataloader = None
        if val_texts is not None and val_labels is not None:
            logger.info("Validation data provided, creating validation dataloader...")
            val_dataloader = self._create_dataloader(val_texts, val_labels, batch_size, sampler_type='sequential')
            if val_dataloader is None: logger.warning("Validation dataloader creation returned None.")
        else:
            logger.info("No validation data provided, skipping validation dataloader creation.")

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        logger.info(f"Scheduler initialized: {total_steps} total steps, {warmup_steps} warmup.")
        loss_fn = nn.CrossEntropyLoss()
        self.history = {k: [] for k in self.history}

        for epoch_i in range(epochs):
            logger.info(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
            t0 = time.time()
            total_train_loss = 0
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    elapsed = self._format_time(time.time() - t0)
                    logger.info(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}.')

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # --- Debugging: Check labels before loss ---
                if b_labels.min() < 0 or b_labels.max() >= self.num_classes:
                     logger.error(f"Epoch {epoch_i+1}, Batch {step}: Invalid label detected!")
                     logger.error(f"Labels in batch: {b_labels.tolist()}")
                     logger.error(f"Min label: {b_labels.min()}, Max label: {b_labels.max()}, Num classes: {self.num_classes}")
                     raise ValueError(f"Invalid label found in batch {step} of epoch {epoch_i+1}")
                # --- End Debugging ---

                self.model.zero_grad()
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
                loss = loss_fn(logits, b_labels)
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self._format_time(time.time() - t0)
            logger.info(f"  Average training loss: {avg_train_loss:.4f}")
            logger.info(f"  Training epoch took: {training_time}")
            self.history['train_loss'].append(avg_train_loss)

            if val_dataloader:
                logger.info("Running Validation...")
                t0_val = time.time()
                # Use the evaluate method, passing the actual validation texts/labels
                avg_val_loss, accuracy, precision, recall, f1 = self.evaluate(
                    val_texts, val_labels, plot_cm=False # Use original val texts/labels
                )
                validation_time = self._format_time(time.time() - t0_val)

                logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  Validation took: {validation_time}")

                self.history['val_loss'].append(avg_val_loss)
                self.history['val_accuracy'].append(accuracy)
                self.history['val_precision'].append(precision)
                self.history['val_recall'].append(recall)
                self.history['val_f1'].append(f1)
            else:
                self.history['val_loss'].append(float('nan'))
                self.history['val_accuracy'].append(float('nan'))
                self.history['val_precision'].append(float('nan'))
                self.history['val_recall'].append(float('nan'))
                self.history['val_f1'].append(float('nan'))
                logger.info("  Skipping validation step.")

        logger.info("Training complete!")
        return self.history

    def evaluate(self, test_texts, test_labels, plot_cm=True, results_dir='src/results', model_name='GNNRoBERTa'):
        """
        Evaluates the model on the test set. Handles label filtering.

        Args:
            test_texts (list): List of test texts.
            test_labels (list): List of test labels.
            plot_cm (bool, optional): Whether to plot the confusion matrix. Defaults to True.
            results_dir (str, optional): Directory to save the confusion matrix plot. Defaults to 'src/results'.
            model_name (str, optional): Name prefix for the confusion matrix file. Defaults to 'GNNRoBERTa'.

        Returns:
            tuple: (loss, accuracy, precision, recall, f1)
        """
        batch_size = self.config['batch_size']
        # Create dataloader, filtering invalid labels internally
        test_dataloader = self._create_dataloader(test_texts, test_labels, batch_size, sampler_type='sequential')
        if test_dataloader is None:
             logger.error("Failed to create evaluation dataloader. Aborting evaluation.")
             return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        loss_fn = nn.CrossEntropyLoss()
        logger.info("Running Evaluation...")
        t0 = time.time()
        self.model.eval()

        total_eval_loss = 0
        all_preds = []
        all_true_labels_from_loader = [] # Store labels actually used by the loader

        for batch in test_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device) # These are already filtered labels

            # --- Debugging: Check labels before eval loss ---
            if b_labels.min() < 0 or b_labels.max() >= self.num_classes:
                 logger.error(f"Evaluation: Invalid label detected in batch!")
                 logger.error(f"Labels in batch: {b_labels.tolist()}")
                 continue # Skip batch

            with torch.no_grad():
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
                loss = loss_fn(logits, b_labels)

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            all_preds.extend(preds)
            all_true_labels_from_loader.extend(b_labels.to('cpu').numpy()) # Store the labels used

        if not all_true_labels_from_loader:
             logger.error("No valid samples were processed during evaluation.")
             return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        # Calculate metrics using the filtered labels from the loader
        avg_test_loss = total_eval_loss / len(test_dataloader)
        accuracy = accuracy_score(all_true_labels_from_loader, all_preds)
        # Use weighted average for multi-class
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels_from_loader, all_preds, average='weighted', zero_division=0)
        # Ensure CM uses the correct set of labels
        cm = confusion_matrix(all_true_labels_from_loader, all_preds, labels=list(range(self.num_classes)))

        evaluation_time = self._format_time(time.time() - t0)

        logger.info(f"  Test Loss: {avg_test_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision (Weighted): {precision:.4f}")
        logger.info(f"  Recall (Weighted): {recall:.4f}")
        logger.info(f"  F1-Score (Weighted): {f1:.4f}")
        logger.info(f"  Evaluation took: {evaluation_time}")
        logger.info(f"  Confusion Matrix:\n{cm}")

        if plot_cm:
            os.makedirs(results_dir, exist_ok=True)
            # Pass explicit class names for 3 classes
            class_names = ['Negative', 'Neutral', 'Positive']
            cm_path = os.path.join(results_dir, f"{model_name.lower()}_test_confusion_matrix.png")
            plot_confusion_matrix(cm, classes=class_names, output_path=cm_path)

        return avg_test_loss, accuracy, precision, recall, f1

    def predict(self, texts):
        """Predicts labels for a list of texts."""
        self.model.eval()
        batch_size = self.config['batch_size']
        # Create dataloader with placeholder labels
        dataloader = self._create_dataloader(texts, [-1]*len(texts), batch_size, sampler_type='sequential')
        if dataloader is None:
             logger.error("Failed to create prediction dataloader.")
             return []

        all_preds = []
        logger.info("Starting prediction...")
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            # Ignore batch[2] (placeholder labels)

            with torch.no_grad():
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            all_preds.extend(preds)

        logger.info("Prediction finished.")
        # Map indices back to labels if needed by caller
        # label_map_inv = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        # pred_labels = [label_map_inv.get(p, 'Unknown') for p in all_preds]
        # return pred_labels
        return all_preds # Return indices for now

    def save_model(self, save_path):
        """Saves the model state dictionary."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # --- Save num_labels in the state dict for easier loading ---
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config # Save the config used to initialize
        }
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Loads the model state dictionary."""
        if not os.path.exists(load_path):
             logger.error(f"Model path not found: {load_path}")
             raise FileNotFoundError(f"Model file not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # --- Load config from checkpoint if available ---
        if 'config' in checkpoint:
            loaded_config = checkpoint['config']
            # Update self.config, but keep essential runtime things like device
            runtime_device = self.device
            self.config.update(loaded_config)
            self.config['device'] = runtime_device # Restore runtime device
            self.num_classes = self.config.get('num_labels', 3) # Update num_classes
            self.max_seq_length = self.config.get('max_seq_length', 128) # Update max_length
            logger.info("Loaded config from checkpoint.")
            # Re-instantiate model based on loaded config
            self.model = GNNRoBERTaModel(
                roberta_model_name=self.config['roberta_model_name'],
                gnn_layers=self.config['gnn_layers'],
                gnn_heads=self.config['gnn_heads'],
                gnn_out_features=self.config['gnn_out_features'],
                dropout=self.config['dropout'],
                num_labels=self.num_classes,
                freeze_roberta=self.config.get('freeze_roberta', False)
            ).to(self.device)
        else:
             logger.warning("Config not found in checkpoint. Model architecture might not match.")
             # Attempt load anyway, assuming current config matches saved state

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        logger.info(f"Model loaded from {load_path}")

    # save_results method remains the same as in the original GNN classifier


# Example usage (typically called from a main script)
if __name__ == '__main__':
    # This is placeholder example code.
    pass # Main execution logic is in gnn_roberta_main.py

