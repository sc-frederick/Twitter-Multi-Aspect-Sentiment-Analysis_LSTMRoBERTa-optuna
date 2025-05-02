import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
# Corrected imports: AdamW is now typically imported from torch.optim
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup # Removed AdamW from here
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import datetime
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GATLayer Class ---
class GATLayer(nn.Module):
    """
    Simplified Graph Attention Layer based on https://arxiv.org/abs/1710.10903
    Operates on node features (e.g., token embeddings) rather than a predefined graph structure.
    Assumes full connectivity initially, attention mechanism learns edge weights.
    """
    def __init__(self, in_features, out_features, heads, dropout, alpha=0.2, concat=True):
        """
        Initializes the GATLayer.

        Args:
            in_features (int): Dimension of input features (e.g., RoBERTa hidden size).
            out_features (int): Dimension of output features per attention head.
            heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            alpha (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.
            concat (bool, optional): If True, concatenate outputs of heads, else average. Defaults to True.
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features      # Input feature dimension (RoBERTa hidden size)
        self.out_features = out_features    # Output feature dimension per head
        self.heads = heads                  # Number of attention heads
        self.concat = concat                # If true, concatenate head outputs; else average
        self.dropout = dropout              # Dropout rate

        # Linear transformation applied to input features for each head
        # Output dim: heads * out_features
        self.W = nn.Linear(in_features, heads * out_features, bias=False)

        # Attention mechanism: a linear layer applied to concatenated features [Wh_i || Wh_j]
        # Input dim: 2 * out_features (concatenated features from two nodes, per head)
        # Output dim: 1 (raw attention score)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1) # Softmax over the last dimension (keys/columns)

    def forward(self, node_features, attention_mask=None):
        """
        Performs the forward pass of the GAT layer.

        Args:
            node_features (torch.Tensor): Input node features (batch_size, seq_len, in_features).
                                          Typically RoBERTa last hidden state.
            attention_mask (torch.Tensor, optional): Mask from tokenizer (batch_size, seq_len).
                                                     1 for real tokens, 0 for padding. Defaults to None.

        Returns:
            torch.Tensor: Output node features after attention (batch_size, seq_len, heads * out_features or out_features).
        """
        batch_size, seq_len, _ = node_features.size() # B, N, D_in

        # 1. Apply linear transformation W
        # node_features: (B, N, D_in) -> Wh: (B, N, H * F_out)
        Wh = self.W(node_features)
        # Reshape for multi-head: (B, N, H, F_out)
        Wh = Wh.view(batch_size, seq_len, self.heads, self.out_features)

        # 2. Compute attention coefficients e_ij
        # Prepare for broadcasting attention mechanism `a` across all pairs (i, j)
        # Wh_i: (B, N, 1, H, F_out) -> expand -> (B, N, N, H, F_out)
        # Wh_j: (B, 1, N, H, F_out) -> expand -> (B, N, N, H, F_out)
        Wh_i = Wh.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)
        Wh_j = Wh.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)

        # Concatenate features for attention input: (B, N, N, H, 2 * F_out)
        concat_features = torch.cat([Wh_i, Wh_j], dim=-1)

        # Apply attention mechanism `a`: (B, N, N, H, 1)
        e = self.a(concat_features)
        # Squeeze last dim: (B, N, N, H)
        e = e.squeeze(-1)
        # Apply leaky ReLU: (B, N, N, H)
        e = self.leaky_relu(e)

        # Permute to (B, H, N, N) for easier masking and softmax
        e = e.permute(0, 3, 1, 2) # Now shape is (batch_size, heads, seq_len_query, seq_len_key)

        # 3. Apply mask to attention scores BEFORE softmax
        if attention_mask is not None:
            # attention_mask shape: (B, N)
            # We want to mask attention where the KEY (j dimension, last dim in `e`) is padding.
            # Reshape mask to (B, 1, 1, N) for broadcasting.
            # This will match `e`'s shape (B, H, N, N) by expanding dims 1 and 2.
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1, N)
            # Masked fill needs a boolean mask. True where mask == 0 (padding)
            # Set attention scores to -inf for padded keys (columns)
            e = e.masked_fill(mask == 0, -float('inf'))

        # 4. Normalize attention coefficients using softmax
        # Softmax is applied across the last dimension (keys/columns)
        # attention shape: (B, H, N, N)
        attention = self.softmax(e)

        # Apply dropout to attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 5. Compute output features (weighted sum)
        # Wh shape: (B, N, H, F_out) -> permute -> (B, H, N, F_out)
        Wh_permuted = Wh.permute(0, 2, 1, 3) # Shape: (B, H, N, F_out)

        # Weighted sum: attention * Wh
        # (B, H, N, N) @ (B, H, N, F_out) -> (B, H, N, F_out)
        h_prime = torch.matmul(attention, Wh_permuted)

        # Permute back to (B, N, H, F_out)
        h_prime = h_prime.permute(0, 2, 1, 3) # Shape: (B, N, H, F_out)

        # 6. Concatenate or average head outputs
        if self.concat:
            # Concatenate along the last dimension
            # (B, N, H * F_out)
            out_features = h_prime.reshape(batch_size, seq_len, self.heads * self.out_features)
        else:
            # Average along the heads dimension (dim 2)
            # (B, N, F_out)
            out_features = h_prime.mean(dim=2)

        return out_features


# --- GNN-RoBERTa Model ---
class GNNRoBERTaModel(nn.Module):
    """
    GNN-RoBERTa model for text classification.
    Combines RoBERTa embeddings with multiple GAT layers.
    """
    def __init__(self, roberta_model_name='roberta-base', gnn_layers=2, gnn_heads=4, gnn_out_features=128, dropout=0.1, num_labels=2, freeze_roberta=False):
        """
        Initializes the GNNRoBERTaModel.

        Args:
            roberta_model_name (str, optional): Name of the pre-trained RoBERTa model. Defaults to 'roberta-base'.
            gnn_layers (int, optional): Number of GAT layers. Defaults to 2.
            gnn_heads (int, optional): Number of attention heads in each GAT layer. Defaults to 4.
            gnn_out_features (int, optional): Output feature dimension per head in GAT layers. Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            num_labels (int, optional): Number of output classes. Defaults to 2.
            freeze_roberta (bool, optional): Whether to freeze RoBERTa layers during training. Defaults to False.
        """
        super(GNNRoBERTaModel, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size

        # Freeze RoBERTa layers if specified
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # GAT Layers
        self.gat_layers = nn.ModuleList()
        gat_input_dim = self.roberta_hidden_size
        for i in range(gnn_layers):
            # If it's the last GAT layer, ensure concat=False if we want averaging,
            # or adjust the classifier input dimension if concat=True.
            # Here, we assume concatenation for all but the last, or adjust classifier later.
            # For simplicity, let's assume concat=True for all layers for now.
            # The final output dimension will be gnn_heads * gnn_out_features.
            concat_layer = True # You might want to set this to False for the last layer
            self.gat_layers.append(
                GATLayer(gat_input_dim, gnn_out_features, gnn_heads, dropout, concat=concat_layer)
            )
            # Input dim for the next layer depends on whether the current layer concatenates
            if concat_layer:
                gat_input_dim = gnn_heads * gnn_out_features
            else:
                gat_input_dim = gnn_out_features # If averaging

        # The final dimension after all GAT layers
        self.final_gat_dim = gat_input_dim

        # Classifier layer
        self.dropout = nn.Dropout(dropout)
        # The input dimension to the classifier depends on the output of the last GAT layer
        self.classifier = nn.Linear(self.final_gat_dim, num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Performs the forward pass of the GNN-RoBERTa model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Logits for each class (batch_size, num_labels).
        """
        # Get RoBERTa embeddings
        # outputs[0] is the last hidden state: (batch_size, seq_len, hidden_size)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Pass through GAT layers
        gat_output = sequence_output
        for layer in self.gat_layers:
            # Pass the attention mask to the GAT layer
            gat_output = layer(gat_output, attention_mask=attention_mask)
            gat_output = F.relu(gat_output) # Apply activation after each GAT layer

        # Use the representation of the [CLS] token (first token) for classification
        # Assumes the [CLS] token representation aggregates sequence information after GAT layers
        cls_token_output = gat_output[:, 0, :] # (batch_size, final_gat_dim)

        # Apply dropout and classify
        pooled_output = self.dropout(cls_token_output)
        logits = self.classifier(pooled_output) # (batch_size, num_labels)

        return logits


# --- Classifier Wrapper ---
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
        self.max_seq_length = config.get('max_seq_length', 128) # Default max sequence length
        self.epochs = config.get('epochs', 3) # <<< ADDED: Store epochs from config, default to 3

        self.model = GNNRoBERTaModel(
            roberta_model_name=config['roberta_model_name'],
            gnn_layers=config['gnn_layers'],
            gnn_heads=config['gnn_heads'],
            gnn_out_features=config['gnn_out_features'],
            dropout=config['dropout'],
            num_labels=config.get('num_labels', 2),
            freeze_roberta=config.get('freeze_roberta', False)
        ).to(self.device)

        logger.info(f"GNN-RoBERTa Classifier initialized on device: {self.device}")
        logger.info(f"Freeze RoBERTa: {config.get('freeze_roberta', False)}")
        logger.info(f"GNN Layers: {config['gnn_layers']}, Heads: {config['gnn_heads']}, Out Features/Head: {config['gnn_out_features']}")
        logger.info(f"Training Epochs set to: {self.epochs}") # <<< ADDED: Log epochs

        # Store training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}


    def _create_dataloader(self, texts, labels, batch_size, sampler_type='random'):
        """Creates a DataLoader for the given data."""
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
                max_length=self.max_seq_length, # Pad & truncate all sentences.
                padding='max_length',         # Pad to max_length
                truncation=True,              # Truncate to max_length
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        if sampler_type == 'random':
            sampler = RandomSampler(dataset)
        elif sampler_type == 'sequential':
            sampler = SequentialSampler(dataset)
        else:
            raise ValueError(f"Invalid sampler_type: {sampler_type}")

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def _format_time(self, elapsed):
        """Takes a time in seconds and returns a string hh:mm:ss"""
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self, train_texts, train_labels, val_texts, val_labels):
        """
        Trains the GNN-RoBERTa model.

        Args:
            train_texts (list): List of training texts.
            train_labels (list): List of training labels.
            val_texts (list): List of validation texts.
            val_labels (list): List of validation labels.

        Returns:
            dict: Training history containing loss and metrics per epoch.
        """
        batch_size = self.config['batch_size']
        epochs = self.epochs # <<< CHANGED: Use stored self.epochs
        learning_rate = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01)
        warmup_steps = self.config.get('scheduler_warmup_steps', 100)

        train_dataloader = self._create_dataloader(train_texts, train_labels, batch_size, sampler_type='random')
        val_dataloader = self._create_dataloader(val_texts, val_labels, batch_size, sampler_type='sequential')

        # Optimizer and Scheduler
        # Use AdamW imported from torch.optim
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        logger.info(f"Scheduler initialized: {total_steps} total steps, {warmup_steps} warmup.")

        loss_fn = nn.CrossEntropyLoss()

        # Reset history before training
        self.history = {k: [] for k in self.history}

        for epoch_i in range(epochs): # Use the epochs variable derived from self.epochs
            logger.info(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
            t0 = time.time()
            total_train_loss = 0
            self.model.train() # Put model in training mode

            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    elapsed = self._format_time(time.time() - t0)
                    logger.info(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}.')

                # Unpack batch, move to device
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad() # Clear previous gradients

                # Forward pass
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)

                # Calculate loss
                loss = loss_fn(logits, b_labels)
                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters
                optimizer.step()
                scheduler.step() # Update learning rate schedule

            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self._format_time(time.time() - t0)
            logger.info(f"  Average training loss: {avg_train_loss:.4f}")
            logger.info(f"  Training epoch took: {training_time}")

            # --- Validation ---
            logger.info("Running Validation...")
            t0 = time.time()
            self.model.eval() # Put model in evaluation mode

            total_eval_accuracy = 0
            total_eval_loss = 0
            all_preds = []
            all_labels = []

            for batch in val_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad(): # No gradient calculation during validation
                    logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
                    loss = loss_fn(logits, b_labels)

                total_eval_loss += loss.item()

                # Move logits and labels to CPU for sklearn metrics
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                preds = np.argmax(logits, axis=1).flatten()
                all_preds.extend(preds)
                all_labels.extend(label_ids)

            # Calculate metrics
            avg_val_loss = total_eval_loss / len(val_dataloader)
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

            validation_time = self._format_time(time.time() - t0)

            # Log and store history
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  Validation took: {validation_time}")

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(accuracy)
            self.history['val_precision'].append(precision)
            self.history['val_recall'].append(recall)
            self.history['val_f1'].append(f1)

        logger.info("Training complete!")
        return self.history

    def evaluate(self, test_texts, test_labels, plot_cm=True, results_dir='src/results', model_name='GNNRoBERTa'):
        """
        Evaluates the model on the test set.

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
        test_dataloader = self._create_dataloader(test_texts, test_labels, batch_size, sampler_type='sequential')
        loss_fn = nn.CrossEntropyLoss()

        logger.info("Running Evaluation on Test Set...")
        t0 = time.time()
        self.model.eval() # Evaluation mode

        total_eval_loss = 0
        all_preds = []
        all_labels = []

        for batch in test_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
                loss = loss_fn(logits, b_labels)

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1).flatten()
            all_preds.extend(preds)
            all_labels.extend(label_ids)

        # Calculate metrics
        avg_test_loss = total_eval_loss / len(test_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        evaluation_time = self._format_time(time.time() - t0)

        logger.info(f"  Test Loss: {avg_test_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  Evaluation took: {evaluation_time}")
        logger.info(f"  Confusion Matrix:\n{cm}")

        # Plot Confusion Matrix
        if plot_cm:
            os.makedirs(results_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(self.config.get('num_labels', 2)), yticklabels=range(self.config.get('num_labels', 2)))
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{model_name} - Confusion Matrix")
            plot_path = os.path.join(results_dir, f"{model_name.lower()}_test_confusion_matrix.png")
            try:
                plt.savefig(plot_path)
                logger.info(f"Confusion matrix saved to {plot_path}")
            except Exception as e:
                logger.error(f"Failed to save confusion matrix plot: {e}")
            plt.close()

        return avg_test_loss, accuracy, precision, recall, f1

    def predict(self, texts):
        """Predicts labels for a list of texts."""
        self.model.eval()
        batch_size = self.config['batch_size']
        # Create dataloader without labels
        input_ids = []
        attention_masks = []
        for text in texts:
             encoded_dict = self.tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=self.max_seq_length,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
            )
             input_ids.append(encoded_dict['input_ids'])
             attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        all_preds = []
        logger.info("Starting prediction...")
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            all_preds.extend(preds)

        logger.info("Prediction finished.")
        return all_preds

    def save_model(self, save_path):
        """Saves the model state dictionary."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Loads the model state dictionary."""
        if not os.path.exists(load_path):
             logger.error(f"Model path not found: {load_path}")
             raise FileNotFoundError(f"Model file not found at {load_path}")
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device) # Ensure model is on the correct device
        logger.info(f"Model loaded from {load_path}")

    def save_results(self, results_path, results_data):
        """Saves evaluation results to a JSON file."""
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        # Load existing results if file exists
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {} # Start fresh if file is corrupted
        else:
            existing_data = {}

        # Update with new results (using a unique key, e.g., model name + timestamp or mode)
        result_key = f"{results_data.get('model_name', 'GNNRoBERTa')}_{results_data.get('mode', 'Test')}_{int(time.time())}"
        existing_data[result_key] = results_data

        # Save updated results
        try:
            with open(results_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {results_path}: {e}")

# Example usage (typically called from a main script)
if __name__ == '__main__':
    # This is placeholder example code.
    # The actual usage would be driven by a main script like gnn_roberta_main.py
    # which would load config, data, and call the classifier methods.

    # Example Config (replace with actual loading from yaml)
    config_example = {
        'roberta_model_name': 'roberta-base',
        'gnn_layers': 2,
        'gnn_heads': 4,
        'gnn_out_features': 64, # Smaller for example
        'dropout': 0.1,
        'num_labels': 2,
        'freeze_roberta': False,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'epochs': 1, # Small for example
        'batch_size': 16, # Small for example
        'scheduler_warmup_steps': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_seq_length': 64
    }

    # Dummy Data
    train_texts_ex = ["This is great!", "This is terrible."] * 10
    train_labels_ex = [1, 0] * 10
    val_texts_ex = ["Looks good.", "Not good."] * 5
    val_labels_ex = [1, 0] * 5
    test_texts_ex = ["Amazing product.", "Very disappointing."] * 5
    test_labels_ex = [1, 0] * 5

    logger.info("--- Example GNN-RoBERTa Classifier Run ---")

    # Initialize Classifier
    classifier = GNNRoBERTaClassifier(config_example)

    # Train
    logger.info("Starting example training...")
    history = classifier.train(train_texts_ex, train_labels_ex, val_texts_ex, val_labels_ex)
    logger.info(f"Training History: {history}")

    # Evaluate
    logger.info("Starting example evaluation...")
    results = classifier.evaluate(test_texts_ex, test_labels_ex, model_name='GNNRoBERTa_Example')
    logger.info(f"Evaluation Results (Loss, Acc, Prec, Rec, F1): {results}")

    # Predict
    logger.info("Starting example prediction...")
    predictions = classifier.predict(["This might work.", "I hate this."])
    logger.info(f"Predictions: {predictions}") # Example: [1, 0]

    # Save/Load Model
    model_save_path = 'src/models/gnn_roberta_example.pt'
    classifier.save_model(model_save_path)
    # classifier.load_model(model_save_path) # Example of loading
    # logger.info("Model reloaded.")

    logger.info("--- Example Run Finished ---")
