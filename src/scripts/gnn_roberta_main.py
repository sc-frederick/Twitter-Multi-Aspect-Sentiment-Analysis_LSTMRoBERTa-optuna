# src/scripts/gnn_roberta_main.py (Adapted for new data loading and splitting)

import logging
import os
import sys
import argparse
import time
import torch
import optuna
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Import pandas

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Make sure utils can be imported
try:
    from utils.data_processor import DataProcessor
    from utils.gnn_roberta_classifier import GNNRoBERTaClassifier
    from utils.lstm_roberta_classifier import SentimentDataset # Re-use compatible dataset
    from utils.results_tracker import save_model_results
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Please ensure 'utils' directory contains necessary files and is importable.")
    sys.exit(1)

from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Configuration & Basic Type Conversion (Re-used) ---
def load_config():
    """Load configuration and perform basic type casting for known numeric fields."""
    config_path = os.path.join(src_dir, 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # --- Type Casting (Copied from previous version) ---
        if 'gnn_roberta' in config:
            gnn_cfg = config['gnn_roberta']
            float_keys_model = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
            int_keys_model = ['gnn_out_features', 'gnn_heads', 'gnn_layers', 'batch_size',
                             'max_length', 'scheduler_warmup_steps',
                             'gradient_accumulation_steps', 'final_epochs']
            bool_keys_model = ['freeze_roberta', 'use_layer_norm', 'use_scheduler']
            for key in float_keys_model:
                if key in gnn_cfg:
                    try: gnn_cfg[key] = float(gnn_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'gnn_roberta.{key}' to float. Value: {gnn_cfg[key]}")
            for key in int_keys_model:
                 if key in gnn_cfg:
                    try: gnn_cfg[key] = int(gnn_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'gnn_roberta.{key}' to int. Value: {gnn_cfg[key]}")
            for key in bool_keys_model:
                 if key in gnn_cfg:
                    try:
                        if isinstance(gnn_cfg[key], str):
                            val_lower = gnn_cfg[key].lower()
                            if val_lower in ['true', 'yes', 'on', '1']: gnn_cfg[key] = True
                            elif val_lower in ['false', 'no', 'off', '0']: gnn_cfg[key] = False
                            else: raise ValueError("Invalid boolean string")
                        else: gnn_cfg[key] = bool(gnn_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'gnn_roberta.{key}' to bool. Value: {gnn_cfg[key]}")
            if 'hpo' in gnn_cfg:
                hpo_cfg = gnn_cfg['hpo']
                int_keys_hpo = ['n_trials', 'timeout_per_trial', 'hpo_sample_size', 'hpo_epochs']
                for key in int_keys_hpo:
                    if key in hpo_cfg:
                        try: hpo_cfg[key] = int(hpo_cfg[key])
                        except (ValueError, TypeError): logger.warning(f"Could not convert config key 'gnn_roberta.hpo.{key}' to int. Value: {hpo_cfg[key]}")
                if 'search_space' in hpo_cfg:
                    for param, spec in hpo_cfg['search_space'].items():
                        spec_type = spec.get('type')
                        if spec_type == 'float' or spec_type == 'int':
                             for bound in ['low', 'high', 'step']:
                                 if bound in spec:
                                     try: spec[bound] = float(spec[bound])
                                     except (ValueError, TypeError): logger.warning(f"Could not convert search space bound '{param}.{bound}' to float. Value: {spec[bound]}")
        # --- End Type Casting ---
        os.makedirs(os.path.join(src_dir, config.get('results_dir', 'results')), exist_ok=True)
        os.makedirs(os.path.join(src_dir, config.get('models_dir', 'models')), exist_ok=True)
        logger.info(f"Configuration loaded and basic types cast from {config_path}")
        return config
    except FileNotFoundError: logger.error(f"Configuration file not found at {config_path}"); sys.exit(1)
    except yaml.YAMLError as e: logger.error(f"Error parsing configuration file: {e}"); sys.exit(1)
    except Exception as e: logger.error(f"Failed to load or process configuration: {e}", exc_info=True); sys.exit(1)

CONFIG = load_config()

# --- GPU Check (Re-used) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    try: logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e: logger.warning(f"Could not get GPU name: {e}")

# --- Plotting Function (Re-used) ---
def plot_confusion_matrix(cm, classes=None, output_path=None):
    """Plot confusion matrix and optionally save."""
    if cm is None: logger.warning("Confusion matrix is None, cannot plot."); return
    if classes is None: classes = ['Negative', 'Neutral', 'Positive'] # Updated for 3 classes
    try:
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix'); plt.tight_layout()
        if output_path:
            try: os.makedirs(os.path.dirname(output_path), exist_ok=True); plt.savefig(output_path); logger.info(f"Confusion matrix saved to {output_path}")
            except Exception as e: logger.error(f"Failed to save confusion matrix plot: {e}")
        plt.close()
    except Exception as plot_err: logger.error(f"Error during confusion matrix plotting: {plot_err}"); plt.close()

# --- Objective Function for Optuna (Re-used, ensure it uses correct data splits and returns F1) ---
def objective(trial, train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo, model_config, hpo_config):
    """Optuna objective function to train and evaluate one GNN-RoBERTa trial."""
    params_to_tune = {}
    search_space = hpo_config.get('search_space', {})
    for param, spec in search_space.items():
        spec_type = spec.get('type')
        try:
            if spec_type == 'float':
                low, high = float(spec['low']), float(spec['high'])
                step = float(spec['step']) if 'step' in spec else None
                params_to_tune[param] = trial.suggest_float(param, low, high, step=step, log=spec.get('log', False))
            elif spec_type == 'int':
                 low, high = int(spec['low']), int(spec['high'])
                 step = int(spec.get('step', 1))
                 params_to_tune[param] = trial.suggest_int(param, low, high, step=step)
            elif spec_type == 'categorical':
                params_to_tune[param] = trial.suggest_categorical(param, spec['choices'])
        except (KeyError, ValueError, TypeError) as e:
             logger.error(f"Trial {trial.number}: Error processing search space for '{param}': {e}. Skipping.")
             continue

    logger.info(f"\n--- Optuna Trial {trial.number} ---")
    logger.info(f"Suggested Parameters: {params_to_tune}")
    current_params = {**model_config, **params_to_tune}

    # --- Ensure 'epochs' is set for the HPO trial ---
    try:
        hpo_epochs = int(hpo_config['hpo_epochs'])
        current_params['epochs'] = hpo_epochs
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Trial {trial.number}: Error getting hpo_epochs: {e}. Using default.")
        current_params['epochs'] = int(model_config.get('final_epochs', 3))

    # --- Instantiate GNN-RoBERTa Classifier ---
    try:
        # Pass the entire dictionary, GNNRoBERTaClassifier constructor handles it
        model = GNNRoBERTaClassifier(config=current_params)

        # --- Use the correct data splits passed to the objective function ---
        history = model.train(train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo)

        # --- Evaluate using the classifier's evaluate method ---
        # Need to filter labels for metrics calculation if there are invalid ones
        valid_indices_val = [i for i, lbl in enumerate(val_labels_hpo) if lbl in [0, 1, 2]]
        val_texts_hpo_filtered = [val_texts_hpo[i] for i in valid_indices_val]
        val_labels_hpo_filtered = [val_labels_hpo[i] for i in valid_indices_val]

        if not val_labels_hpo_filtered:
             logger.error(f"Trial {trial.number}: No valid validation labels found after filtering. Cannot evaluate.")
             return 0.0 # Return failure

        val_loss, val_accuracy, _, _, val_f1 = model.evaluate(
            val_texts_hpo_filtered,
            val_labels_hpo_filtered,
            plot_cm=False # No need to plot CM for every trial
        )

        logger.info(f"Trial {trial.number} - Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}")

        # Report F1-score to Optuna
        metric_to_optimize = val_f1
        trial.report(metric_to_optimize, step=current_params['epochs'])
        if trial.should_prune(): raise optuna.TrialPruned()
        return metric_to_optimize

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        return trial.user_attrs.get("last_reported_value", 0.0)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 0.0

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Optimize/Train/Test GNN-RoBERTa model")
    parser.add_argument("--mode", type=str, choices=['optimize', 'train', 'test'], default='train', help="Operation mode")
    parser.add_argument("--sample_size_train", type=int, default=None, help="Override train samples loaded")
    parser.add_argument("--sample_size_test", type=int, default=None, help="Override test samples loaded")
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials")
    parser.add_argument("--final_epochs", type=int, default=None, help="Override final training epochs")
    args = parser.parse_args()

    global_config = CONFIG
    model_config = CONFIG.get('gnn_roberta', {}) # <--- Changed section name
    hpo_config = model_config.get('hpo', {})

    # Apply CLI overrides
    sample_train = args.sample_size_train if args.sample_size_train is not None else model_config.get('base_sample_size', None)
    sample_test = args.sample_size_test if args.sample_size_test is not None else None
    n_trials = int(args.n_trials if args.n_trials is not None else hpo_config.get('n_trials', 20))
    final_epochs = int(args.final_epochs if args.final_epochs is not None else model_config.get('final_epochs', 3))
    run_hpo = bool(hpo_config.get('enabled', True)) if args.mode == 'optimize' else False

    # Define paths
    results_dir = os.path.join(src_dir, global_config.get('results_dir', 'results'))
    models_dir = os.path.join(src_dir, global_config.get('models_dir', 'models'))
    model_save_path = os.path.join(models_dir, 'gnn_roberta_model_final.pt') # <--- Changed name

    # --- Load Data using the updated DataProcessor ---
    logger.info(f"Loading train/test data...")
    data_processor = DataProcessor()
    try:
        train_df, test_df = data_processor.load_data(sample_size_train=sample_train, sample_size_test=sample_test)
        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    except Exception as data_err:
        logger.error(f"Failed to load data: {data_err}", exc_info=True)
        sys.exit(1)

    # --- Prepare Data Splits ---
    train_texts_all = train_df['text_clean'].tolist()
    train_labels_all = train_df['label'].tolist()
    test_texts = test_df['text_clean'].tolist()
    test_labels = test_df['label'].tolist()

    try:
        val_size = float(global_config.get('hpo_validation_size', 0.25))
        random_state = int(global_config.get('random_state', 42))
        stratify_train = train_labels_all if len(set(train_labels_all)) > 1 else None

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts_all, train_labels_all, test_size=val_size,
            random_state=random_state, stratify=stratify_train)
        logger.info(f"Split training data: Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
        logger.info(f"Test set size remains: {len(test_texts)}")
    except Exception as split_err:
        logger.error(f"Failed to split training data into train/validation: {split_err}", exc_info=True)
        sys.exit(1)

    # --- HPO Setup ---
    best_params_from_hpo = {}
    study = None
    if args.mode == 'optimize' and run_hpo:
        logger.info(f"Starting GNN-RoBERTa hyperparameter optimization with Optuna for {n_trials} trials...")
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna_gnn_roberta.db')}" # <--- Changed DB name
        study = optuna.create_study(
            study_name=f"gnn-roberta-hpo-{time.strftime('%Y%m%d-%H%M')}", direction="maximize", # Maximize F1
            storage=storage_name, load_if_exists=True)
        logger.info(f"Using Optuna storage: {storage_name}")
        try:
            study.optimize(
                lambda trial: objective(trial, train_texts, train_labels, val_texts, val_labels, model_config, hpo_config),
                n_trials=n_trials, timeout=None, gc_after_trial=True)
            logger.info("Optimization finished.")
            logger.info(f"Number of finished trials: {len(study.trials)}")
            logger.info(f"Best trial overall (F1-Score): {study.best_value:.4f}") # Log F1
            best_params_from_hpo = study.best_params
            logger.info(f"Best hyperparameters: {best_params_from_hpo}")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            best_params_from_hpo = {}

    # Determine final parameters
    final_params = {**model_config}
    final_params.update(best_params_from_hpo)

    # Ensure final parameters have correct types and include 'epochs'
    float_keys_final = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
    int_keys_final = ['gnn_out_features', 'gnn_heads', 'gnn_layers', 'batch_size', 'max_length', 'scheduler_warmup_steps', 'gradient_accumulation_steps']
    bool_keys_final = ['freeze_roberta', 'use_layer_norm', 'use_scheduler']
    try:
        for key in float_keys_final: final_params[key] = float(final_params.get(key, model_config.get(key)))
        for key in int_keys_final: final_params[key] = int(final_params.get(key, model_config.get(key)))
        for key in bool_keys_final: final_params[key] = bool(final_params.get(key, model_config.get(key)))
        final_params['epochs'] = final_epochs # Set final epochs
        final_params['device'] = DEVICE # Ensure device is in config
        final_params['num_labels'] = 3 # Explicitly set num_labels for this dataset
        final_params['max_seq_length'] = int(final_params.get('max_length', 128)) # Ensure max_seq_length

        # Ensure roberta_model_name exists (might be named differently in config)
        if 'roberta_model' in final_params and 'roberta_model_name' not in final_params:
            final_params['roberta_model_name'] = final_params['roberta_model']
        elif 'roberta_model_name' not in final_params:
             final_params['roberta_model_name'] = 'roberta-base' # Default if missing

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error casting/setting final parameters: {e}", exc_info=True)
        sys.exit(1)


    # --- Train Final Model or Test ---
    model_to_evaluate = None
    final_training_time = None
    if args.mode == 'optimize' or args.mode == 'train':
        logger.info("Training final GNN-RoBERTa model...")
        log_params = {k: final_params.get(k) for k in ['learning_rate', 'gnn_out_features', 'gnn_heads', 'gnn_layers', 'dropout', 'weight_decay', 'freeze_roberta', 'epochs']}
        logger.info(f"Using parameters: {log_params}")
        try:
            # Instantiate Final GNN-RoBERTa Classifier with final_params
            final_model_classifier = GNNRoBERTaClassifier(config=final_params)

            start_time = time.time()
            logger.info(f"Starting final training on {len(train_texts)} samples, validating on {len(val_texts)} samples, for {final_params['epochs']} epochs.")
            # Train on the training set, validate on the validation set
            final_model_classifier.train(train_texts, train_labels, val_texts, val_labels)
            final_training_time = time.time() - start_time
            logger.info(f"Final model training time: {final_training_time:.2f}s")

            final_model_classifier.save_model(model_save_path)
            logger.info(f"Final GNN-RoBERTa model saved to {model_save_path}")
            model_to_evaluate = final_model_classifier
        except Exception as train_err:
             logger.error(f"Failed during final GNN-RoBERTa training: {train_err}", exc_info=True)
             model_to_evaluate = None

    elif args.mode == 'test':
        logger.info(f"Loading GNN-RoBERTa model for testing from {model_save_path}...")
        if os.path.exists(model_save_path):
             model_config_for_load = CONFIG.get('gnn_roberta', {}) # Load relevant config section
             try:
                 # Instantiate with ARCHITECTURE params before loading weights
                 test_config = model_config_for_load.copy()
                 test_config['device'] = DEVICE
                 test_config['num_labels'] = 3 # Set explicitly
                 test_config['batch_size'] = int(model_config_for_load.get('batch_size', 16))
                 test_config['max_seq_length'] = int(model_config_for_load.get('max_length', 128))
                 # Ensure roberta_model_name exists
                 if 'roberta_model' in test_config and 'roberta_model_name' not in test_config:
                     test_config['roberta_model_name'] = test_config['roberta_model']
                 elif 'roberta_model_name' not in test_config:
                     test_config['roberta_model_name'] = 'roberta-base'

                 model_to_evaluate = GNNRoBERTaClassifier(config=test_config)
                 model_to_evaluate.load_model(model_save_path) # Load weights and potentially optimizer state
                 logger.info(f"GNN-RoBERTa model loaded successfully from {model_save_path}")
             except Exception as e:
                 logger.error(f"Failed to load GNN-RoBERTa model state: {e}", exc_info=True)
                 model_to_evaluate = None
        else:
            logger.error(f"No GNN-RoBERTa model found at {model_save_path}. Cannot test.")
            return

    # --- Evaluate on the Reserved Test Set ---
    if model_to_evaluate:
        logger.info("Evaluating final GNN-RoBERTa model on the reserved test set...")
        try:
            # Filter labels before evaluation
            valid_indices_test = [i for i, lbl in enumerate(test_labels) if lbl in [0, 1, 2]]
            if len(valid_indices_test) < len(test_labels):
                 logger.warning(f"Filtered out {len(test_labels) - len(valid_indices_test)} samples from test set due to invalid labels.")
            test_texts_filtered = [test_texts[i] for i in valid_indices_test]
            test_labels_filtered = [test_labels[i] for i in valid_indices_test]

            if not test_labels_filtered:
                 logger.error("No valid test labels remaining after filtering. Cannot evaluate.")
            else:
                test_loss, accuracy, precision, recall, f1 = model_to_evaluate.evaluate(
                    test_texts_filtered,
                    test_labels_filtered,
                    plot_cm=True,
                    results_dir=results_dir,
                    model_name='GNNRoBERTa_Final'
                )
                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'loss': test_loss}

                logger.info("\nFinal Test Set Evaluation Results (GNN-RoBERTa):")
                logger.info(f"Loss: {test_loss:.4f}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision (Weighted): {precision:.4f}")
                logger.info(f"Recall (Weighted): {recall:.4f}")
                logger.info(f"F1 Score (Weighted): {f1:.4f}")

                # Save results
                params_to_save = {
                    'mode': args.mode,
                    'train_sample_size_used': sample_train if sample_train is not None else 'all',
                    'test_sample_size_used': sample_test if sample_test is not None else 'all',
                    'final_epochs_run': final_epochs if args.mode != 'test' else 'N/A',
                    'evaluated_model_path': model_save_path,
                    **{k: v for k, v in final_params.items() if k != 'hpo'}
                }
                if final_training_time is not None: params_to_save['final_training_time_seconds'] = round(final_training_time, 2)
                if args.mode == 'optimize' and study:
                     params_to_save['hpo_n_trials_run'] = len(study.trials)
                     params_to_save['hpo_best_val_f1'] = study.best_value # Metric was F1
                     params_to_save['hpo_best_params'] = study.best_params

                save_model_results(
                    model_name=f"GNNRoBERTa_{args.mode.capitalize()}", # Changed name
                    metrics=metrics, parameters=params_to_save, example_predictions={})
                logger.info("Evaluation results saved.")
        except Exception as eval_err:
             logger.error(f"Failed during final evaluation: {eval_err}", exc_info=True)
    else:
        logger.warning("No GNN-RoBERTa model was available for final evaluation.")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
