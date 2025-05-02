# src/scripts/gnn_roberta_main.py (Adapted for GNN-RoBERTa)

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

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Make sure utils can be imported
try:
    from utils.data_processor import DataProcessor
    # Import the new GNN-RoBERTa classifier and compatible dataset
    from utils.gnn_roberta_classifier import GNNRoBERTaClassifier
    # Use the same SentimentDataset as it's compatible with RoBERTa inputs
    from utils.lstm_roberta_classifier import SentimentDataset
    from utils.results_tracker import save_model_results
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Please ensure 'utils' directory contains 'gnn_roberta_classifier.py' and is importable.")
    sys.exit(1)

from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Configuration & Basic Type Conversion ---
def load_config():
    """Load configuration and perform basic type casting for known numeric fields."""
    config_path = os.path.join(src_dir, 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- Type Casting for GNN-RoBERTa Section ---
        if 'gnn_roberta' in config:
            gnn_cfg = config['gnn_roberta']
            float_keys_model = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
            int_keys_model = ['gnn_out_features', 'gnn_heads', 'gnn_layers', 'batch_size',
                             'max_length', 'scheduler_warmup_steps',
                             'gradient_accumulation_steps', 'final_epochs'] # Keep final_epochs here
            bool_keys_model = ['freeze_roberta', 'use_layer_norm', 'use_scheduler'] # use_layer_norm might not be directly used by GNN model but could be in wrapper

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

            # Cast HPO settings
            if 'hpo' in gnn_cfg:
                hpo_cfg = gnn_cfg['hpo']
                int_keys_hpo = ['n_trials', 'timeout_per_trial', 'hpo_sample_size', 'hpo_epochs'] # hpo_epochs added
                for key in int_keys_hpo:
                    if key in hpo_cfg:
                        try: hpo_cfg[key] = int(hpo_cfg[key])
                        except (ValueError, TypeError): logger.warning(f"Could not convert config key 'gnn_roberta.hpo.{key}' to int. Value: {hpo_cfg[key]}")
                # Cast search space bounds
                if 'search_space' in hpo_cfg:
                    for param, spec in hpo_cfg['search_space'].items():
                        spec_type = spec.get('type')
                        if spec_type == 'float' or spec_type == 'int':
                             for bound in ['low', 'high', 'step']:
                                 if bound in spec:
                                     try: spec[bound] = float(spec[bound]) # Use float for int bounds too
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

# --- GPU Check ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    try: logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e: logger.warning(f"Could not get GPU name: {e}")
# --- End GPU Check ---

# --- Plotting Function (Re-used) ---
def plot_confusion_matrix(cm, classes=None, output_path=None):
    """Plot confusion matrix and optionally save."""
    if cm is None: logger.warning("Confusion matrix is None, cannot plot."); return
    if classes is None: classes = ['Negative', 'Positive']
    try:
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix'); plt.tight_layout()
        if output_path:
            try: os.makedirs(os.path.dirname(output_path), exist_ok=True); plt.savefig(output_path); logger.info(f"Confusion matrix saved to {output_path}")
            except Exception as e: logger.error(f"Failed to save confusion matrix plot: {e}")
        plt.close()
    except Exception as plot_err: logger.error(f"Error during confusion matrix plotting: {plot_err}"); plt.close()

# --- Objective Function for Optuna (Adapted for GNN-RoBERTa) ---
def objective(trial, train_texts, train_labels, val_texts, val_labels, model_config, hpo_config):
    """Optuna objective function to train and evaluate one GNN-RoBERTa trial."""
    params_to_tune = {}
    search_space = hpo_config.get('search_space', {})

    # Suggest parameters based on search_space defined in config.yaml for gnn_roberta
    for param, spec in search_space.items():
        spec_type = spec.get('type')
        try:
            if spec_type == 'float':
                low, high = float(spec['low']), float(spec['high'])
                step = float(spec['step']) if 'step' in spec else None
                log_scale = spec.get('log', False)
                params_to_tune[param] = trial.suggest_float(param, low, high, step=step, log=log_scale)
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

    # Combine suggested params with fixed params from model_config
    current_params = {**model_config, **params_to_tune}

    # --- Ensure 'epochs' is set for the HPO trial ---
    try:
        hpo_epochs = int(hpo_config['hpo_epochs'])
        current_params['epochs'] = hpo_epochs # Add/overwrite epochs for this trial
        logger.info(f"Trial {trial.number}: Setting epochs to {hpo_epochs} from hpo_config")
    except KeyError:
        logger.error(f"Trial {trial.number}: 'hpo_epochs' not found in hpo_config. Using default from model_config.")
        # Fallback to final_epochs if hpo_epochs isn't defined, though it should be
        current_params['epochs'] = int(model_config.get('final_epochs', 3))
    except (ValueError, TypeError) as e:
        logger.error(f"Trial {trial.number}: Error converting hpo_epochs ('{hpo_config.get('hpo_epochs')}') to int: {e}. Using default.")
        current_params['epochs'] = int(model_config.get('final_epochs', 3))


    # --- Instantiate GNN-RoBERTa Classifier with the combined config ---
    try:
        # Pass the entire dictionary
        model = GNNRoBERTaClassifier(config=current_params)

        # Train and evaluate
        # The train method now uses self.epochs internally
        history = model.train(train_texts, train_labels, val_texts, val_labels)

        # Evaluate using the classifier's evaluate method
        # Create a DataLoader for evaluation
        val_dataset = SentimentDataset(val_texts, val_labels, model.tokenizer, model.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=model.config['batch_size']) # Use batch size from config
        val_loss, val_accuracy, _, _, val_f1 = model.evaluate(val_texts, val_labels, plot_cm=False) # Use evaluate method

        logger.info(f"Trial {trial.number} - Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}")

        # Report intermediate value to Optuna for pruning
        # Use F1-score as the metric to maximize, as it's often better for imbalanced datasets
        metric_to_optimize = val_f1
        trial.report(metric_to_optimize, step=current_params['epochs']) # Report based on the actual epochs run
        if trial.should_prune():
            raise optuna.TrialPruned()

        return metric_to_optimize # Return metric to maximize

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        # Return last reported value or a default bad value (e.g., 0.0 for F1)
        return trial.user_attrs.get("last_reported_value", 0.0)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 0.0 # Indicate failure

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Optimize/Train/Test GNN-RoBERTa model")
    parser.add_argument( "--mode", type=str, choices=['optimize', 'train', 'test'], default='train', help="Operation mode")
    parser.add_argument("--sample_size", type=int, default=None, help="Override samples loaded")
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials")
    parser.add_argument("--final_epochs", type=int, default=None, help="Override final training epochs")
    args = parser.parse_args()

    global_config = CONFIG
    # --- Load GNN-RoBERTa specific config ---
    model_config = CONFIG.get('gnn_roberta', {}) # <--- Changed section name
    if not model_config:
        logger.error("Configuration section 'gnn_roberta' not found in config.yaml!")
        sys.exit(1)
    hpo_config = model_config.get('hpo', {})

    # Apply CLI overrides
    base_sample_size = int(args.sample_size if args.sample_size is not None else model_config.get('base_sample_size', 10000))
    n_trials = int(args.n_trials if args.n_trials is not None else hpo_config.get('n_trials', 20))
    # Get final_epochs from args or config, ensuring it's an int
    try:
        final_epochs = int(args.final_epochs if args.final_epochs is not None else model_config.get('final_epochs', 3))
    except (ValueError, TypeError):
        logger.warning(f"Invalid final_epochs value found. Using default: 3")
        final_epochs = 3
    run_hpo = bool(hpo_config.get('enabled', True)) if args.mode == 'optimize' else False

    # Define paths
    results_dir = os.path.join(src_dir, global_config.get('results_dir', 'results'))
    models_dir = os.path.join(src_dir, global_config.get('models_dir', 'models'))
    # --- Specific model path for GNN-RoBERTa ---
    model_save_path = os.path.join(models_dir, 'gnn_roberta_model_final.pt') # <--- Changed name

    # Determine DB path
    db_path_config = global_config.get('database_path', 'data/tweets_dataset.db')
    db_path = os.path.abspath(os.path.join(src_dir, db_path_config)) if not os.path.isabs(db_path_config) else db_path_config

    # --- Load and Process Data ---
    logger.info(f"Loading data (up to {base_sample_size} samples) from {db_path}...")
    data_processor = DataProcessor(database_path=db_path)
    try: df = data_processor.load_data(sample_size=base_sample_size)
    except Exception as data_err: logger.error(f"Failed to load data: {data_err}", exc_info=True); sys.exit(1)
    try: labels = (df['target'].astype(float) == 4.0).astype(int).tolist()
    except Exception as label_err: logger.error(f"Failed to process 'target' column: {label_err}. Check data.", exc_info=True); sys.exit(1)
    texts = df['text_clean'].tolist()

    # Split data: Train+Val / Test
    try:
        stratify_split = labels if len(set(labels)) > 1 else None
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=float(global_config.get('test_size', 0.2)),
            random_state=int(global_config.get('random_state', 42)), stratify=stratify_split)
        logger.info(f"Data split: Train/Val size: {len(train_val_texts)}, Test size: {len(test_texts)}")
    except Exception as split_err: logger.error(f"Failed to split data: {split_err}", exc_info=True); sys.exit(1)

    # Prepare HPO data subset
    train_texts_hpo, val_texts_hpo, train_labels_hpo, val_labels_hpo = None, None, None, None
    if args.mode == 'optimize':
        hpo_sample_size_cfg = int(hpo_config.get('hpo_sample_size', 5000))
        hpo_total_size = min(len(train_val_texts), hpo_sample_size_cfg)
        if hpo_total_size <= 0: logger.error(f"Cannot perform HPO with {hpo_total_size} samples."); sys.exit(1)
        if hpo_total_size < len(train_val_texts):
             stratify_hpo_sub = train_val_labels if len(set(train_val_labels)) > 1 else None
             hpo_texts, _, hpo_labels, _ = train_test_split(train_val_texts, train_val_labels, train_size=hpo_total_size,
                 random_state=int(global_config.get('random_state', 42)), stratify=stratify_hpo_sub)
             logger.info(f"Using {hpo_total_size} samples for HPO train/validation.")
        else: hpo_texts, hpo_labels = train_val_texts, train_val_labels
        try:
            stratify_hpo_final = hpo_labels if len(set(hpo_labels)) > 1 else None
            if stratify_hpo_final is None: logger.warning("Cannot stratify HPO train/val split.")
            train_texts_hpo, val_texts_hpo, train_labels_hpo, val_labels_hpo = train_test_split(
                hpo_texts, hpo_labels, test_size=float(global_config.get('hpo_validation_size', 0.25)),
                random_state=int(global_config.get('random_state', 42)), stratify=stratify_hpo_final)
        except Exception as hpo_split_err: logger.error(f"Failed to split HPO data: {hpo_split_err}", exc_info=True); sys.exit(1)
        if not train_texts_hpo or not val_texts_hpo: logger.error("HPO train or validation set empty."); sys.exit(1)
        logger.info(f"HPO Train size: {len(train_texts_hpo)}, HPO Validation size: {len(val_texts_hpo)}")

    # --- Optuna Optimization (GNN-RoBERTa) ---
    best_params_from_hpo = {}
    study = None
    if args.mode == 'optimize' and run_hpo:
        logger.info(f"Starting GNN-RoBERTa hyperparameter optimization with Optuna for {n_trials} trials...")
        # --- Use a different DB name for GNN-RoBERTa study ---
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna_gnn_roberta.db')}" # <--- Changed DB name
        study = optuna.create_study(
            study_name=f"gnn-roberta-hpo-{time.strftime('%Y%m%d-%H%M')}", direction="maximize", # Maximize F1-score
            storage=storage_name, load_if_exists=True)
        logger.info(f"Using Optuna storage: {storage_name}")
        try:
            study.optimize(
                lambda trial: objective(trial, train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo, model_config, hpo_config),
                n_trials=n_trials, timeout=None, gc_after_trial=True)
            logger.info("Optimization finished.")
            logger.info(f"Number of finished trials: {len(study.trials)}")
            logger.info(f"Best trial overall (F1-Score): {study.best_value:.4f}") # Log F1
            logger.info("Best hyperparameters overall:")
            best_params_from_hpo = study.best_params
            for key, value in best_params_from_hpo.items(): logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            best_params_from_hpo = {}

    # Determine final parameters for 'train' mode
    final_params = {**model_config, **best_params_from_hpo}

    # --- Ensure final parameters have correct types and include 'epochs' ---
    if args.mode == 'train' or args.mode == 'optimize':
        # Adapt keys based on GNN-RoBERTa config
        float_keys_final = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
        int_keys_final = ['gnn_out_features', 'gnn_heads', 'gnn_layers', 'batch_size',
                         'max_length', 'scheduler_warmup_steps', 'gradient_accumulation_steps']
        bool_keys_final = ['freeze_roberta', 'use_layer_norm', 'use_scheduler']
        try:
            for key in float_keys_final: final_params[key] = float(final_params.get(key, model_config.get(key)))
            for key in int_keys_final: final_params[key] = int(final_params.get(key, model_config.get(key)))
            for key in bool_keys_final:
                 val = final_params.get(key, model_config.get(key))
                 if isinstance(val, str):
                     val_lower = val.lower(); final_params[key] = val_lower in ['true', 'yes', 'on', '1']
                 else: final_params[key] = bool(val)
            # --- Set epochs for the final training run ---
            final_params['epochs'] = final_epochs
            logger.info(f"Setting final training epochs to: {final_epochs}")

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error casting final parameters: {e}", exc_info=True)
            sys.exit(1)


    # --- Train Final Model (Mode: optimize or train) ---
    model_to_evaluate = None
    final_training_time = None
    if args.mode == 'optimize' or args.mode == 'train':
        logger.info("Training final GNN-RoBERTa model...")
        # Log relevant params
        log_params = {k: final_params.get(k) for k in ['learning_rate', 'gnn_out_features', 'gnn_heads', 'gnn_layers', 'dropout', 'weight_decay', 'freeze_roberta', 'epochs']}
        logger.info(f"Using parameters: {log_params}")
        try:
            # --- Instantiate Final GNN-RoBERTa Classifier with final_params ---
            # final_params now includes the 'epochs' key set to final_epochs
            final_model_classifier = GNNRoBERTaClassifier(config=final_params)

            start_time = time.time()
            logger.info(f"Starting final training on {len(train_val_texts)} samples for {final_params['epochs']} epochs.")
            # Train on full train_val set, no validation needed here for the final run
            final_model_classifier.train(train_val_texts, train_val_labels, None, None)
            final_training_time = time.time() - start_time
            logger.info(f"Final model training time: {final_training_time:.2f}s")

            # Save the final GNN-RoBERTa model
            final_model_classifier.save_model(model_save_path)
            logger.info(f"Final GNN-RoBERTa model saved to {model_save_path}")
            model_to_evaluate = final_model_classifier # Use the wrapper for evaluation
        except Exception as train_err:
             logger.error(f"Failed during final GNN-RoBERTa training: {train_err}", exc_info=True)
             model_to_evaluate = None

    # --- Load Model for Testing (Mode: test) ---
    elif args.mode == 'test':
        logger.info(f"Loading GNN-RoBERTa model for testing from {model_save_path}...")
        if os.path.exists(model_save_path):
             try:
                 # Instantiate a classifier wrapper first, using base config.
                 # load_model will then load the saved state dict.
                 # We need to provide the essential architectural params from config
                 # for the initial instantiation before loading the state.
                 test_config = model_config.copy() # Start with base config
                 test_config['device'] = DEVICE # Ensure device is set
                 # Add other necessary keys if GNNRoBERTaClassifier expects them at init
                 # (e.g., batch_size, max_length might be needed for helper methods,
                 # but the core model structure is defined by GNNRoBERTaModel params)
                 test_config['batch_size'] = int(model_config.get('batch_size', 16))
                 test_config['max_seq_length'] = int(model_config.get('max_length', 128))
                 test_config['num_labels'] = int(model_config.get('num_labels', 2))

                 model_to_evaluate = GNNRoBERTaClassifier(config=test_config)
                 model_to_evaluate.load_model(model_save_path) # Load the saved weights
                 logger.info(f"GNN-RoBERTa model loaded successfully from {model_save_path}")
             except Exception as e:
                 logger.error(f"Failed to load GNN-RoBERTa model state: {e}", exc_info=True)
                 model_to_evaluate = None
        else:
            logger.error(f"No GNN-RoBERTa model found at {model_save_path}. Cannot test.")
            model_to_evaluate = None

    # --- Evaluate on the Test Set ---
    if model_to_evaluate:
        if 'test_texts' not in locals() or 'test_labels' not in locals():
             logger.error("Test data not loaded. Cannot evaluate.")
        else:
            logger.info("Evaluating GNN-RoBERTa model on the test set...")
            try:
                # Use the evaluate method of the classifier wrapper
                test_loss, accuracy, precision, recall, f1 = model_to_evaluate.evaluate(
                    test_texts,
                    test_labels,
                    plot_cm=True, # Enable plotting for the final test
                    results_dir=results_dir,
                    model_name='GNNRoBERTa_Final'
                )
                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'loss': test_loss}

                logger.info("\nFinal Test Set Evaluation Results (GNN-RoBERTa):")
                logger.info(f"Loss: {test_loss:.4f}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1 Score: {f1:.4f}")

                # Save results
                params_to_save = {
                    'mode': args.mode,
                    'base_sample_size': base_sample_size,
                    'evaluated_model_path': model_save_path,
                    # Include final params used for training/eval or loaded config for test
                    # Use the config stored within the loaded classifier instance if testing
                    **(model_to_evaluate.config if args.mode == 'test' else final_params)
                }
                # Add final epochs if training was done
                if args.mode in ['train', 'optimize']:
                     params_to_save['final_epochs_run'] = final_params['epochs']

                params_to_save.pop('hpo', None) # Remove HPO sub-dict if present
                if final_training_time is not None and args.mode in ['train', 'optimize']:
                     params_to_save['final_training_time_seconds'] = round(final_training_time, 2)
                if args.mode == 'optimize' and study:
                     params_to_save['hpo_n_trials_run'] = len(study.trials)
                     params_to_save['hpo_best_val_f1'] = study.best_value # Metric was F1
                     params_to_save['hpo_best_params'] = study.best_params

                # Use the results tracker utility
                save_model_results(
                    model_name=f"GNNRoBERTa_{args.mode.capitalize()}", # Changed name
                    metrics=metrics,
                    parameters=params_to_save,
                    example_predictions={} # Add example predictions if desired
                )
                logger.info("Evaluation results saved via results_tracker.")

            except Exception as eval_err:
                logger.error(f"Failed during final evaluation: {eval_err}", exc_info=True)

    elif not model_to_evaluate:
         logger.warning("No GNN-RoBERTa model was available or trained/loaded for evaluation.")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
