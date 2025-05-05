# src/scripts/han_roberta_main.py (Adapted for new data loading and splitting)

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
    from utils.han_roberta_classifier import HANRoBERTaSentimentClassifier, SentimentDataset
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
        if 'han_roberta' in config:
            hr_cfg = config['han_roberta']
            float_keys_model = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
            int_keys_model = ['hidden_dim', 'num_layers', 'batch_size', 'max_length',
                             'scheduler_warmup_steps', 'gradient_accumulation_steps',
                             'final_epochs']
            bool_keys_model = ['freeze_roberta', 'use_layer_norm', 'use_gru',
                              'use_scheduler']
            for key in float_keys_model:
                if key in hr_cfg:
                    try: hr_cfg[key] = float(hr_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'han_roberta.{key}' to float. Value: {hr_cfg[key]}")
            for key in int_keys_model:
                 if key in hr_cfg:
                    try: hr_cfg[key] = int(hr_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'han_roberta.{key}' to int. Value: {hr_cfg[key]}")
            for key in bool_keys_model:
                 if key in hr_cfg:
                    try:
                        if isinstance(hr_cfg[key], str):
                            val_lower = hr_cfg[key].lower()
                            if val_lower in ['true', 'yes', 'on', '1']: hr_cfg[key] = True
                            elif val_lower in ['false', 'no', 'off', '0']: hr_cfg[key] = False
                            else: raise ValueError("Invalid boolean string")
                        else: hr_cfg[key] = bool(hr_cfg[key])
                    except (ValueError, TypeError): logger.warning(f"Could not convert config key 'han_roberta.{key}' to bool. Value: {hr_cfg[key]}")
            if 'hpo' in hr_cfg:
                hpo_cfg = hr_cfg['hpo']
                int_keys_hpo = ['n_trials', 'timeout_per_trial', 'hpo_sample_size', 'hpo_epochs']
                for key in int_keys_hpo:
                    if key in hpo_cfg:
                        try: hpo_cfg[key] = int(hpo_cfg[key])
                        except (ValueError, TypeError): logger.warning(f"Could not convert config key 'han_roberta.hpo.{key}' to int. Value: {hpo_cfg[key]}")
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

# --- Objective Function for Optuna (Re-used, ensure it uses correct data splits) ---
def objective(trial, train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo, model_config, hpo_config):
    """Optuna objective function to train and evaluate one HAN-RoBERTa trial."""
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

    try:
        # Ensure types before passing to classifier
        lr = float(current_params.get('learning_rate', model_config.get('learning_rate', 1e-5)))
        hidden_dim = int(current_params.get('hidden_dim', model_config.get('hidden_dim', 256)))
        num_layers = int(current_params.get('num_layers', model_config.get('num_layers', 1)))
        dropout = float(current_params.get('dropout', model_config.get('dropout', 0.3)))
        weight_decay = float(current_params.get('weight_decay', model_config.get('weight_decay', 0.01)))
        freeze_roberta = bool(current_params.get('freeze_roberta', model_config.get('freeze_roberta', True)))
        use_layer_norm = bool(current_params.get('use_layer_norm', model_config.get('use_layer_norm', True)))
        use_gru = bool(current_params.get('use_gru', model_config.get('use_gru', True)))
        use_scheduler = bool(current_params.get('use_scheduler', model_config.get('use_scheduler', True)))
        batch_size = int(current_params.get('batch_size', model_config.get('batch_size', 16)))
        max_length = int(current_params.get('max_length', model_config.get('max_length', 128)))
        scheduler_warmup_steps = int(current_params.get('scheduler_warmup_steps', model_config.get('scheduler_warmup_steps', 100)))
        gradient_accumulation_steps = int(current_params.get('gradient_accumulation_steps', model_config.get('gradient_accumulation_steps', 2)))
        max_grad_norm = float(current_params.get('max_grad_norm', model_config.get('max_grad_norm', 1.0)))

        model = HANRoBERTaSentimentClassifier(
            roberta_model=str(current_params['roberta_model']),
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, learning_rate=lr,
            batch_size=batch_size, max_length=max_length, num_epochs=int(hpo_config['hpo_epochs']),
            freeze_roberta=freeze_roberta, device=DEVICE, use_gru=use_gru, use_layer_norm=use_layer_norm,
            weight_decay=weight_decay, use_scheduler=use_scheduler, scheduler_warmup_steps=scheduler_warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm
        )

        # --- Use the correct data splits passed to the objective function ---
        history = model.train(train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo)
        # --- Evaluation needs a DataLoader for the validation set ---
        val_dataset = SentimentDataset(val_texts_hpo, val_labels_hpo, model.tokenizer, model.max_length)
        val_loader = DataLoader(val_dataset, batch_size=model.batch_size)
        val_loss, val_accuracy = model.evaluate(val_loader)

        logger.info(f"Trial {trial.number} - Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
        trial.report(val_accuracy, step=hpo_config['hpo_epochs'])
        if trial.should_prune(): raise optuna.TrialPruned()
        return val_accuracy

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        return trial.user_attrs.get("last_reported_value", 0.0)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 0.0

# --- Main Execution Logic ---
def main():
    # Remove finetune mode for simplicity in this update
    parser = argparse.ArgumentParser(description="Optimize/Train/Test HAN-RoBERTa model")
    parser.add_argument("--mode", type=str, choices=['optimize', 'train', 'test'], default='train', help="Operation mode")
    parser.add_argument("--sample_size_train", type=int, default=None, help="Override train samples loaded")
    parser.add_argument("--sample_size_test", type=int, default=None, help="Override test samples loaded")
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials")
    parser.add_argument("--final_epochs", type=int, default=None, help="Override final training epochs")
    args = parser.parse_args()

    global_config = CONFIG
    model_config = CONFIG.get('han_roberta', {}) # <--- Changed section name
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
    model_save_path = os.path.join(models_dir, 'han_roberta_model_final.pt') # <--- Changed name

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
        logger.info(f"Starting HAN-RoBERTa hyperparameter optimization with Optuna for {n_trials} trials...")
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna_han_roberta.db')}" # <--- Changed DB name
        study = optuna.create_study(
            study_name=f"han-roberta-hpo-{time.strftime('%Y%m%d-%H%M')}", direction="maximize",
            storage=storage_name, load_if_exists=True)
        logger.info(f"Using Optuna storage: {storage_name}")
        try:
            study.optimize(
                lambda trial: objective(trial, train_texts, train_labels, val_texts, val_labels, model_config, hpo_config),
                n_trials=n_trials, timeout=None, gc_after_trial=True)
            logger.info("Optimization finished.")
            logger.info(f"Number of finished trials: {len(study.trials)}")
            logger.info(f"Best trial overall (Accuracy): {study.best_value:.4f}")
            best_params_from_hpo = study.best_params
            logger.info(f"Best hyperparameters: {best_params_from_hpo}")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            best_params_from_hpo = {}

    # Determine final parameters
    final_params = {**model_config}
    final_params.update(best_params_from_hpo)

    # Ensure final parameters have correct types
    float_keys_final = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
    int_keys_final = ['hidden_dim', 'num_layers', 'batch_size', 'max_length', 'scheduler_warmup_steps', 'gradient_accumulation_steps']
    bool_keys_final = ['freeze_roberta', 'use_layer_norm', 'use_gru', 'use_scheduler']
    for key in float_keys_final: final_params[key] = float(final_params.get(key, model_config.get(key)))
    for key in int_keys_final: final_params[key] = int(final_params.get(key, model_config.get(key)))
    for key in bool_keys_final: final_params[key] = bool(final_params.get(key, model_config.get(key)))

    # --- Train Final Model or Test ---
    model_to_evaluate = None
    final_training_time = None
    if args.mode == 'optimize' or args.mode == 'train':
        logger.info("Training final HAN-RoBERTa model...")
        log_params = {k: final_params.get(k) for k in ['learning_rate', 'hidden_dim', 'num_layers', 'dropout', 'weight_decay', 'freeze_roberta', 'use_gru']}
        logger.info(f"Using parameters: {log_params}")
        try:
            final_model = HANRoBERTaSentimentClassifier(
                roberta_model=str(final_params['roberta_model']),
                hidden_dim=final_params['hidden_dim'], num_layers=final_params['num_layers'], dropout=final_params['dropout'],
                learning_rate=final_params['learning_rate'], batch_size=final_params['batch_size'], max_length=final_params['max_length'],
                num_epochs=final_epochs, freeze_roberta=final_params['freeze_roberta'], device=DEVICE,
                use_gru=final_params.get('use_gru', True), use_layer_norm=final_params.get('use_layer_norm', True),
                weight_decay=final_params.get('weight_decay', 0.01), use_scheduler=final_params.get('use_scheduler', True),
                scheduler_warmup_steps=final_params.get('scheduler_warmup_steps', 100),
                gradient_accumulation_steps=final_params.get('gradient_accumulation_steps', 2),
                max_grad_norm=final_params.get('max_grad_norm', 1.0)
            )
            start_time = time.time()
            logger.info(f"Starting final training on {len(train_texts)} samples, validating on {len(val_texts)} samples, for {final_epochs} epochs.")
            final_model.train(train_texts, train_labels, val_texts, val_labels)
            final_training_time = time.time() - start_time
            logger.info(f"Final model training time: {final_training_time:.2f}s")
            final_model.save_model(model_save_path)
            logger.info(f"Final HAN-RoBERTa model saved to {model_save_path}")
            model_to_evaluate = final_model
        except Exception as train_err:
             logger.error(f"Failed during final HAN-RoBERTa training: {train_err}", exc_info=True)
             model_to_evaluate = None

    elif args.mode == 'test':
        logger.info(f"Loading HAN-RoBERTa model for testing from {model_save_path}...")
        if os.path.exists(model_save_path):
             model_config_for_load = CONFIG.get('han_roberta', {}) # Load relevant config section
             try:
                 # Instantiate with ARCHITECTURE params before loading weights
                 model_to_evaluate = HANRoBERTaSentimentClassifier(
                     roberta_model=model_config_for_load.get('roberta_model', 'roberta-base'),
                     hidden_dim=int(model_config_for_load.get('hidden_dim', 256)),
                     num_layers=int(model_config_for_load.get('num_layers', 1)),
                     dropout=float(model_config_for_load.get('dropout', 0.3)),
                     use_gru=bool(model_config_for_load.get('use_gru', True)),
                     use_layer_norm=bool(model_config_for_load.get('use_layer_norm', True)),
                     device=DEVICE
                 )
                 model_to_evaluate.load_model(model_save_path) # Load weights and potentially optimizer state
                 logger.info(f"HAN-RoBERTa model loaded successfully from {model_save_path}")
             except Exception as e:
                 logger.error(f"Failed to load HAN-RoBERTa model state: {e}", exc_info=True)
                 model_to_evaluate = None
        else:
            logger.error(f"No HAN-RoBERTa model found at {model_save_path}. Cannot test.")
            return

    # --- Evaluate on the Reserved Test Set ---
    if model_to_evaluate:
        logger.info("Evaluating final HAN-RoBERTa model on the reserved test set...")
        try:
            test_predictions_raw = model_to_evaluate.predict(test_texts)
            pred_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            test_pred_labels = [pred_map.get(pred['label'], -1) for pred in test_predictions_raw]

            valid_indices = [i for i, lbl in enumerate(test_labels) if lbl in [0, 1, 2]]
            if len(valid_indices) < len(test_labels):
                logger.warning(f"Filtered out {len(test_labels) - len(valid_indices)} samples from test set due to invalid labels.")
            test_labels_filtered = [test_labels[i] for i in valid_indices]
            test_pred_labels_filtered = [test_pred_labels[i] for i in valid_indices]

            if not test_labels_filtered:
                 logger.error("No valid test labels remaining after filtering. Cannot calculate metrics.")
            else:
                accuracy = accuracy_score(test_labels_filtered, test_pred_labels_filtered)
                precision = precision_score(test_labels_filtered, test_pred_labels_filtered, average='weighted', zero_division=0)
                recall = recall_score(test_labels_filtered, test_pred_labels_filtered, average='weighted', zero_division=0)
                f1 = f1_score(test_labels_filtered, test_pred_labels_filtered, average='weighted', zero_division=0)
                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

                logger.info("\nFinal Test Set Evaluation Results (HAN-RoBERTa):")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision (Weighted): {precision:.4f}")
                logger.info(f"Recall (Weighted): {recall:.4f}")
                logger.info(f"F1 Score (Weighted): {f1:.4f}")

                try:
                    cm = sk_confusion_matrix(test_labels_filtered, test_pred_labels_filtered, labels=[0, 1, 2])
                    cm_path = os.path.join(results_dir, f'han_roberta_{args.mode}_confusion_matrix.png') # Changed name
                    plot_confusion_matrix(cm, classes=['Negative', 'Neutral', 'Positive'], output_path=cm_path)
                except ImportError: logger.warning("Scikit-learn not fully available. Cannot generate confusion matrix.")
                except Exception as e: logger.error(f"Could not generate/save confusion matrix: {e}")

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
                     params_to_save['hpo_best_val_accuracy'] = study.best_value
                     params_to_save['hpo_best_params'] = study.best_params

                save_model_results(
                    model_name=f"HANRoBERTa_{args.mode.capitalize()}", # Changed name
                    metrics=metrics, parameters=params_to_save, example_predictions={})
                logger.info("Evaluation results saved.")
        except Exception as eval_err:
             logger.error(f"Failed during final evaluation: {eval_err}", exc_info=True)
    else:
        logger.warning("No HAN-RoBERTa model was available for final evaluation.")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
