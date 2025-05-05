# src/utils/gnn_roberta_classifier.py
# Updated to handle None validation data during final training and 3 output classes
# Removed label validation from Dataset constructor.

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
from typing import List, Dict # Added for type hints

# Re-use the SentimentDataset from the LSTM version as input format is compatible
try:
    from .lstm_roberta_classifier import SentimentDataset # Use the updated one
except ImportError:
    logger.error("Could not import SentimentDataset. Ensure lstm_roberta_classifier.py is accessible.")
    # Define a basic compatible Dataset if import fails (example structure)
    class SentimentDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
            """
            Initialize the dataset. Assumes labels are already validated.
            """
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            # --- REMOVED label validation check from __init__ ---

        def __len__(self) -> int: return len(self.texts)
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            text = str(self.texts[idx]); label = self.labels[idx] # Assume label is valid
            encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label, dtype=torch.long)}


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GATLayer Class (Unchanged) ---
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features; self.out_features = out_features; self.heads = heads
        self.concat = concat; self.dropout = dropout
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha); self.softmax = nn.Softmax(dim=-1)
    def forward(self, node_features, attention_mask=None):
        batch_size, seq_len, _ = node_features.size()
        Wh = self.W(node_features).view(batch_size, seq_len, self.heads, self.out_features)
        Wh_i = Wh.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)
        Wh_j = Wh.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.heads, self.out_features)
        concat_features = torch.cat([Wh_i, Wh_j], dim=-1); e = self.a(concat_features).squeeze(-1)
        e = self.leaky_relu(e).permute(0, 3, 1, 2)
        if attention_mask is not None: e = e.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -float('inf'))
        attention = F.dropout(self.softmax(e), self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        if self.concat: out_features = h_prime.reshape(batch_size, seq_len, self.heads * self.out_features)
        else: out_features = h_prime.mean(dim=2)
        return out_features

# --- GNN-RoBERTa Model (Unchanged from previous fix) ---
class GNNRoBERTaModel(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', gnn_layers=2, gnn_heads=4, gnn_out_features=128, dropout=0.1, num_labels=3, freeze_roberta=False):
        super(GNNRoBERTaModel, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        if freeze_roberta:
            for param in self.roberta.parameters(): param.requires_grad = False
        self.gat_layers = nn.ModuleList()
        gat_input_dim = self.roberta_hidden_size
        for i in range(gnn_layers):
            concat_layer = True
            self.gat_layers.append(GATLayer(gat_input_dim, gnn_out_features, gnn_heads, dropout, concat=concat_layer))
            if concat_layer: gat_input_dim = gnn_heads * gnn_out_features
            else: gat_input_dim = gnn_out_features
        self.final_gat_dim = gat_input_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.final_gat_dim, self.num_labels)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]; gat_output = sequence_output
        for layer in self.gat_layers:
            gat_output = layer(gat_output, attention_mask=attention_mask); gat_output = F.relu(gat_output)
        cls_token_output = gat_output[:, 0, :]; pooled_output = self.dropout(cls_token_output)
        logits = self.classifier(pooled_output); return logits

# --- Classifier Wrapper (Unchanged from previous fix) ---
class GNNRoBERTaClassifier:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model_name'])
        self.max_seq_length = config.get('max_seq_length', 128)
        self.epochs = config.get('epochs', 3)
        self.num_classes = config.get('num_labels', 3)
        self.model = GNNRoBERTaModel(
            roberta_model_name=config['roberta_model_name'], gnn_layers=config['gnn_layers'], gnn_heads=config['gnn_heads'],
            gnn_out_features=config['gnn_out_features'], dropout=config['dropout'], num_labels=self.num_classes,
            freeze_roberta=config.get('freeze_roberta', False)).to(self.device)
        logger.info(f"GNN-RoBERTa Classifier initialized on device: {self.device}")
        logger.info(f"Freeze RoBERTa: {config.get('freeze_roberta', False)}")
        logger.info(f"GNN Layers: {config['gnn_layers']}, Heads: {config['gnn_heads']}, Out Features/Head: {config['gnn_out_features']}")
        logger.info(f"Number of output classes: {self.num_classes}")
        logger.info(f"Training Epochs set to: {self.epochs}")
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}

    # _create_dataloader, _format_time, train, evaluate, predict, save_model, load_model methods remain the same as the previous fix
    # (Including the label checks within train/evaluate loops and _create_dataloader)
    def _create_dataloader(self, texts, labels, batch_size, sampler_type='random'):
        if texts is None: logger.warning(f"Attempted to create dataloader with None texts. Returning None."); return None
        input_ids = []; attention_masks = []; valid_labels = []; valid_indices = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if not (0 <= label < self.num_classes):
                 logger.warning(f"Invalid label found at index {i}: {label}. Skipping sample. Text: {text[:100]}...")
                 continue
            encoded_dict = self.tokenizer.encode_plus(str(text), add_special_tokens=True, max_length=self.max_seq_length, padding='max_length',
                                                      truncation=True, return_attention_mask=True, return_tensors='pt')
            input_ids.append(encoded_dict['input_ids']); attention_masks.append(encoded_dict['attention_mask'])
            valid_labels.append(label); valid_indices.append(i)
        if not input_ids: logger.error("No valid samples found after label checking. Cannot create DataLoader."); return None
        input_ids = torch.cat(input_ids, dim=0); attention_masks = torch.cat(attention_masks, dim=0); labels_tensor = torch.tensor(valid_labels)
        dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
        if sampler_type == 'random': sampler = RandomSampler(dataset)
        elif sampler_type == 'sequential': sampler = SequentialSampler(dataset)
        else: raise ValueError(f"Invalid sampler_type: {sampler_type}")
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        logger.info(f"Created DataLoader with {len(dataset)} valid samples."); return dataloader

    def _format_time(self, elapsed): elapsed_rounded = int(round((elapsed))); return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self, train_texts, train_labels, val_texts, val_labels):
        batch_size = self.config['batch_size']; epochs = self.epochs; learning_rate = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01); warmup_steps = self.config.get('scheduler_warmup_steps', 100)
        train_dataloader = self._create_dataloader(train_texts, train_labels, batch_size, sampler_type='random')
        if train_dataloader is None: logger.error("Failed to create training dataloader. Aborting training."); return self.history
        val_dataloader = None
        if val_texts is not None and val_labels is not None:
            logger.info("Validation data provided, creating validation dataloader...")
            val_dataloader = self._create_dataloader(val_texts, val_labels, batch_size, sampler_type='sequential')
            if val_dataloader is None: logger.warning("Validation dataloader creation returned None.")
        else: logger.info("No validation data provided, skipping validation dataloader creation.")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        logger.info(f"Scheduler initialized: {total_steps} total steps, {warmup_steps} warmup.")
        loss_fn = nn.CrossEntropyLoss(); self.history = {k: [] for k in self.history}
        for epoch_i in range(epochs):
            logger.info(f"\n======== Epoch {epoch_i + 1} / {epochs} ========"); t0 = time.time(); total_train_loss = 0; self.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0: elapsed = self._format_time(time.time() - t0); logger.info(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}.')
                b_input_ids = batch[0].to(self.device); b_input_mask = batch[1].to(self.device); b_labels = batch[2].to(self.device)
                if b_labels.min() < 0 or b_labels.max() >= self.num_classes:
                     logger.error(f"Epoch {epoch_i+1}, Batch {step}: Invalid label detected! Labels: {b_labels.tolist()}, Min: {b_labels.min()}, Max: {b_labels.max()}, Num classes: {self.num_classes}")
                     raise ValueError(f"Invalid label found in batch {step} of epoch {epoch_i+1}")
                self.model.zero_grad(); logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask); loss = loss_fn(logits, b_labels)
                total_train_loss += loss.item(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); optimizer.step(); scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader); training_time = self._format_time(time.time() - t0)
            logger.info(f"  Average training loss: {avg_train_loss:.4f}"); logger.info(f"  Training epoch took: {training_time}"); self.history['train_loss'].append(avg_train_loss)
            if val_dataloader:
                logger.info("Running Validation..."); t0_val = time.time()
                avg_val_loss, accuracy, precision, recall, f1 = self.evaluate(val_texts, val_labels, plot_cm=False)
                validation_time = self._format_time(time.time() - t0_val)
                logger.info(f"  Validation Loss: {avg_val_loss:.4f}"); logger.info(f"  Accuracy: {accuracy:.4f}"); logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}"); logger.info(f"  F1-Score: {f1:.4f}"); logger.info(f"  Validation took: {validation_time}")
                self.history['val_loss'].append(avg_val_loss); self.history['val_accuracy'].append(accuracy); self.history['val_precision'].append(precision); self.history['val_recall'].append(recall); self.history['val_f1'].append(f1)
            else:
                self.history['val_loss'].append(float('nan')); self.history['val_accuracy'].append(float('nan')); self.history['val_precision'].append(float('nan')); self.history['val_recall'].append(float('nan')); self.history['val_f1'].append(float('nan'))
                logger.info("  Skipping validation step.")
        logger.info("Training complete!"); return self.history

    def evaluate(self, test_texts, test_labels, plot_cm=True, results_dir='src/results', model_name='GNNRoBERTa'):
        batch_size = self.config['batch_size']
        test_dataloader = self._create_dataloader(test_texts, test_labels, batch_size, sampler_type='sequential')
        if test_dataloader is None: logger.error("Failed to create evaluation dataloader."); return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        loss_fn = nn.CrossEntropyLoss(); logger.info("Running Evaluation..."); t0 = time.time(); self.model.eval()
        total_eval_loss = 0; all_preds = []; all_true_labels_from_loader = []
        for batch in test_dataloader:
            b_input_ids = batch[0].to(self.device); b_input_mask = batch[1].to(self.device); b_labels = batch[2].to(self.device)
            if b_labels.min() < 0 or b_labels.max() >= self.num_classes: logger.error(f"Evaluation: Invalid label detected in batch! Labels: {b_labels.tolist()}"); continue
            with torch.no_grad(): logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask); loss = loss_fn(logits, b_labels)
            total_eval_loss += loss.item(); logits = logits.detach().cpu().numpy(); preds = np.argmax(logits, axis=1).flatten()
            all_preds.extend(preds); all_true_labels_from_loader.extend(b_labels.to('cpu').numpy())
        if not all_true_labels_from_loader: logger.error("No valid samples processed during evaluation."); return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        avg_test_loss = total_eval_loss / len(test_dataloader); accuracy = accuracy_score(all_true_labels_from_loader, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels_from_loader, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_true_labels_from_loader, all_preds, labels=list(range(self.num_classes)))
        evaluation_time = self._format_time(time.time() - t0)
        logger.info(f"  Test Loss: {avg_test_loss:.4f}"); logger.info(f"  Accuracy: {accuracy:.4f}"); logger.info(f"  Precision (Weighted): {precision:.4f}")
        logger.info(f"  Recall (Weighted): {recall:.4f}"); logger.info(f"  F1-Score (Weighted): {f1:.4f}"); logger.info(f"  Evaluation took: {evaluation_time}"); logger.info(f"  Confusion Matrix:\n{cm}")
        if plot_cm:
            os.makedirs(results_dir, exist_ok=True); class_names = ['Negative', 'Neutral', 'Positive']
            cm_path = os.path.join(results_dir, f"{model_name.lower()}_test_confusion_matrix.png")
            plot_confusion_matrix(cm, classes=class_names, output_path=cm_path)
        return avg_test_loss, accuracy, precision, recall, f1

    def predict(self, texts):
        self.model.eval(); batch_size = self.config['batch_size']
        dataloader = self._create_dataloader(texts, [-1]*len(texts), batch_size, sampler_type='sequential')
        if dataloader is None: logger.error("Failed to create prediction dataloader."); return []
        all_preds = []; logger.info("Starting prediction...")
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device); b_input_mask = batch[1].to(self.device)
            with torch.no_grad(): logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy(); preds = np.argmax(logits, axis=1).flatten(); all_preds.extend(preds)
        logger.info("Prediction finished."); return all_preds

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dict = {'model_state_dict': self.model.state_dict(), 'config': self.config}
        torch.save(save_dict, save_path); logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        if not os.path.exists(load_path): raise FileNotFoundError(f"Model file not found at {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        if 'config' in checkpoint:
            loaded_config = checkpoint['config']; runtime_device = self.device; self.config.update(loaded_config)
            self.config['device'] = runtime_device; self.num_classes = self.config.get('num_labels', 3); self.max_seq_length = self.config.get('max_seq_length', 128)
            logger.info("Loaded config from checkpoint.")
            self.model = GNNRoBERTaModel(
                roberta_model_name=self.config['roberta_model_name'], gnn_layers=self.config['gnn_layers'], gnn_heads=self.config['gnn_heads'],
                gnn_out_features=self.config['gnn_out_features'], dropout=self.config['dropout'], num_labels=self.num_classes,
                freeze_roberta=self.config.get('freeze_roberta', False)).to(self.device)
        else: logger.warning("Config not found in checkpoint. Model architecture might not match.")
        self.model.load_state_dict(checkpoint['model_state_dict']); self.model.to(self.device); logger.info(f"Model loaded from {load_path}")

# Example usage (placeholder)
if __name__ == '__main__': pass

