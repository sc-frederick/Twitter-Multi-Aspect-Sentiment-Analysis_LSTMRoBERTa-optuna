# src/scripts/lstm_roberta_main.py (Modified for Optuna and config.yaml)

import logging
import os
import sys
import argparse
import time
import torch
import optuna
import yaml # Import YAML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    from utils.lstm_roberta_classifier import HybridSentimentClassifier, SentimentDataset
    from utils.results_tracker import save_model_results
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Please ensure the 'utils' directory is in the correct location relative to 'scripts' and is importable.")
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

        # --- Add Type Casting Here ---
        if 'lstm_roberta' in config:
            lr_cfg = config['lstm_roberta']
            # Cast default model parameters
            float_keys_model = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
            int_keys_model = ['hidden_dim', 'num_layers', 'batch_size', 'max_length',
                             'num_attention_heads', 'scheduler_warmup_steps',
                             'gradient_accumulation_steps', 'final_epochs']
            bool_keys_model = ['freeze_roberta', 'use_layer_norm', 'use_residual',
                              'use_pooler_output', 'use_scheduler']

            for key in float_keys_model:
                if key in lr_cfg:
                    try:
                        lr_cfg[key] = float(lr_cfg[key])
                    except (ValueError, TypeError):
                         logger.warning(f"Could not convert config key 'lstm_roberta.{key}' to float. Value: {lr_cfg[key]}")
            for key in int_keys_model:
                 if key in lr_cfg:
                    try:
                        lr_cfg[key] = int(lr_cfg[key])
                    except (ValueError, TypeError):
                         logger.warning(f"Could not convert config key 'lstm_roberta.{key}' to int. Value: {lr_cfg[key]}")
            for key in bool_keys_model:
                 if key in lr_cfg:
                    try:
                        # Handle YAML boolean variations
                        if isinstance(lr_cfg[key], str):
                            val_lower = lr_cfg[key].lower()
                            if val_lower in ['true', 'yes', 'on', '1']:
                                lr_cfg[key] = True
                            elif val_lower in ['false', 'no', 'off', '0']:
                                 lr_cfg[key] = False
                            else:
                                raise ValueError("Invalid boolean string")
                        else:
                             lr_cfg[key] = bool(lr_cfg[key])
                    except (ValueError, TypeError):
                         logger.warning(f"Could not convert config key 'lstm_roberta.{key}' to bool. Value: {lr_cfg[key]}")


            # Cast HPO settings and search space bounds
            if 'hpo' in lr_cfg:
                hpo_cfg = lr_cfg['hpo']
                int_keys_hpo = ['n_trials', 'timeout_per_trial', 'hpo_sample_size', 'hpo_epochs']
                for key in int_keys_hpo:
                    if key in hpo_cfg:
                        try:
                            hpo_cfg[key] = int(hpo_cfg[key])
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert config key 'lstm_roberta.hpo.{key}' to int. Value: {hpo_cfg[key]}")

                if 'search_space' in hpo_cfg:
                    for param, spec in hpo_cfg['search_space'].items():
                        spec_type = spec.get('type')
                        if spec_type == 'float' or spec_type == 'int':
                             for bound in ['low', 'high', 'step']:
                                 if bound in spec:
                                     try:
                                         spec[bound] = float(spec[bound]) # Use float for int bounds too, optuna handles it
                                     except (ValueError, TypeError):
                                         logger.warning(f"Could not convert search space bound '{param}.{bound}' to float. Value: {spec[bound]}")
        # --- End Type Casting ---

        # Create necessary directories using potentially updated config values
        os.makedirs(os.path.join(src_dir, config.get('results_dir', 'results')), exist_ok=True)
        os.makedirs(os.path.join(src_dir, config.get('models_dir', 'models')), exist_ok=True)
        logger.info(f"Configuration loaded and basic types cast from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e: # Catch broader errors during loading/casting
        logger.error(f"Failed to load or process configuration: {e}", exc_info=True)
        sys.exit(1)

CONFIG = load_config()

# --- GPU Check ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    try:
        logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.warning(f"Could not get GPU name: {e}")
# --- End GPU Check ---

# --- Plotting Function ---
def plot_confusion_matrix(cm, classes=None, output_path=None):
    """Plot confusion matrix and optionally save."""
    if cm is None:
        logger.warning("Confusion matrix is None, cannot plot.")
        return
    if classes is None:
        classes = ['Negative', 'Positive']

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                    xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger.info(f"Confusion matrix saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save confusion matrix plot: {e}")
        # plt.show() # Avoid showing in automated runs
        plt.close() # Close the figure explicitly
    except Exception as plot_err:
         logger.error(f"Error during confusion matrix plotting: {plot_err}")
         plt.close() # Ensure figure is closed even if error occurs

# --- Objective Function for Optuna ---
def objective(trial, train_texts, train_labels, val_texts, val_labels, model_config, hpo_config):
    """Optuna objective function to train and evaluate one trial."""

    params_to_tune = {}
    search_space = hpo_config.get('search_space', {})

    for param, spec in search_space.items():
        spec_type = spec.get('type')
        # Use try-except for robustness during suggestions
        try:
            if spec_type == 'float':
                # Ensure bounds are float before suggesting
                low = float(spec['low'])
                high = float(spec['high'])
                step = spec.get('step') # Can be None
                if step is not None: step = float(step)
                params_to_tune[param] = trial.suggest_float(param, low, high, step=step, log=spec.get('log', False))
            elif spec_type == 'int':
                 low = int(spec['low'])
                 high = int(spec['high'])
                 step = spec.get('step', 1) # Default step 1 for int
                 params_to_tune[param] = trial.suggest_int(param, low, high, step=int(step))
            elif spec_type == 'categorical':
                params_to_tune[param] = trial.suggest_categorical(param, spec['choices'])
        except KeyError as e:
             logger.error(f"Trial {trial.number}: Missing key '{e}' in search space spec for '{param}'. Skipping.")
             continue
        except (ValueError, TypeError) as e:
             logger.error(f"Trial {trial.number}: Invalid value in search space spec for '{param}': {e}. Skipping.")
             continue

    logger.info(f"\n--- Optuna Trial {trial.number} ---")
    logger.info(f"Suggested Parameters: {params_to_tune}")

    # Combine suggested params with fixed params from model_config
    # Make sure defaults exist for all expected keys in HybridSentimentClassifier
    current_params = {
        **model_config, # Start with potentially type-cast defaults
        **params_to_tune  # Optuna's suggested values override
    }

    try:
        # Ensure key numeric params are correct type before passing to classifier
        # Use .get with defaults from model_config in case Optuna didn't tune them
        lr = float(current_params.get('learning_rate', model_config.get('learning_rate', 1e-5)))
        hidden_dim = int(current_params.get('hidden_dim', model_config.get('hidden_dim', 256)))
        num_layers = int(current_params.get('num_layers', model_config.get('num_layers', 2)))
        dropout = float(current_params.get('dropout', model_config.get('dropout', 0.3)))
        weight_decay = float(current_params.get('weight_decay', model_config.get('weight_decay', 0.01)))
        num_attention_heads = int(current_params.get('num_attention_heads', model_config.get('num_attention_heads', 4)))
        # Booleans
        freeze_roberta = bool(current_params.get('freeze_roberta', model_config.get('freeze_roberta', True)))
        use_layer_norm = bool(current_params.get('use_layer_norm', model_config.get('use_layer_norm', True)))
        use_residual = bool(current_params.get('use_residual', model_config.get('use_residual', True)))
        use_pooler_output = bool(current_params.get('use_pooler_output', model_config.get('use_pooler_output', False)))
        use_scheduler = bool(current_params.get('use_scheduler', model_config.get('use_scheduler', True)))
        # Ints
        batch_size = int(current_params.get('batch_size', model_config.get('batch_size', 16)))
        max_length = int(current_params.get('max_length', model_config.get('max_length', 128)))
        scheduler_warmup_steps = int(current_params.get('scheduler_warmup_steps', model_config.get('scheduler_warmup_steps', 100)))
        gradient_accumulation_steps = int(current_params.get('gradient_accumulation_steps', model_config.get('gradient_accumulation_steps', 2)))
        max_grad_norm = float(current_params.get('max_grad_norm', model_config.get('max_grad_norm', 1.0)))

        model = HybridSentimentClassifier(
            roberta_model=str(current_params['roberta_model']),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            batch_size=batch_size,
            max_length=max_length,
            num_epochs=int(hpo_config['hpo_epochs']), # Use HPO epochs
            freeze_roberta=freeze_roberta,
            device=DEVICE,
            num_attention_heads=num_attention_heads,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            use_pooler_output=use_pooler_output,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            scheduler_warmup_steps=scheduler_warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm
        )

        # Train and evaluate
        history = model.train(train_texts, train_labels, val_texts, val_labels)
        val_dataset = SentimentDataset(val_texts, val_labels, model.tokenizer, model.max_length)
        val_loader = DataLoader(val_dataset, batch_size=model.batch_size)
        val_loss, val_accuracy = model.evaluate(val_loader)

        logger.info(f"Trial {trial.number} - Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        # Report intermediate value to Optuna for pruning (optional)
        trial.report(val_accuracy, step=hpo_config['hpo_epochs'])
        # Check if trial should be pruned (optional)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_accuracy # Return metric to maximize

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        # Return a value indicating pruning, Optuna handles it
        # Often return the last reported value or a default bad value
        return trial.user_attrs.get("last_reported_value", 0.0)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        # Return a value indicating failure (e.g., low accuracy)
        return 0.0


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Optimize/Train/Test hybrid LSTM-RoBERTa model")
    parser.add_argument( "--mode", type=str, choices=['optimize', 'train', 'test'], default='train', help="Operation mode")
    parser.add_argument("--sample_size", type=int, default=None, help="Override total samples loaded (from config: base_sample_size)")
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials (from config: hpo.n_trials)")
    parser.add_argument("--final_epochs", type=int, default=None, help="Override final training epochs (from config: lstm_roberta.final_epochs)")
    args = parser.parse_args()

    global_config = CONFIG
    model_config = CONFIG.get('lstm_roberta', {})
    hpo_config = model_config.get('hpo', {})

    # Apply CLI overrides (ensure casting if needed)
    base_sample_size = int(args.sample_size if args.sample_size is not None else model_config.get('base_sample_size', 10000))
    n_trials = int(args.n_trials if args.n_trials is not None else hpo_config.get('n_trials', 20))
    final_epochs = int(args.final_epochs if args.final_epochs is not None else model_config.get('final_epochs', 3))
    run_hpo = bool(hpo_config.get('enabled', True)) if args.mode == 'optimize' else False

    # Define paths using config
    results_dir = os.path.join(src_dir, global_config.get('results_dir', 'results'))
    models_dir = os.path.join(src_dir, global_config.get('models_dir', 'models'))
    model_save_path = os.path.join(models_dir, 'lstm_roberta_model_final.pt')
    # Construct absolute path for DB if relative path is given in config
    db_path_config = global_config.get('database_path', 'data/tweets_dataset.db')
    if not os.path.isabs(db_path_config):
         db_path = os.path.abspath(os.path.join(src_dir, db_path_config))
    else:
         db_path = db_path_config

    # Load and process data
    logger.info(f"Loading data (up to {base_sample_size} samples) from {db_path}...")
    data_processor = DataProcessor(database_path=db_path)
    try:
        df = data_processor.load_data(sample_size=base_sample_size)
    except Exception as data_err:
        logger.error(f"Failed to load data: {data_err}", exc_info=True)
        sys.exit(1)

    # Ensure target column is numeric (handle potential errors)
    try:
        labels = (df['target'].astype(float) == 4.0).astype(int).tolist() # Assuming 4 is positive
    except Exception as label_err:
        logger.error(f"Failed to process 'target' column: {label_err}. Check data.", exc_info=True)
        logger.info(f"Target column info:\n{df['target'].describe()}")
        logger.info(f"Target value counts:\n{df['target'].value_counts()}")
        sys.exit(1)
    texts = df['text_clean'].tolist()


    # Split data: Train+Val / Test
    try:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=float(global_config.get('test_size', 0.2)),
            random_state=int(global_config.get('random_state', 42)),
            stratify=labels
        )
    except Exception as split_err:
        logger.error(f"Failed to split data: {split_err}", exc_info=True)
        sys.exit(1)

    # Prepare HPO data subset if needed
    train_texts_hpo, val_texts_hpo, train_labels_hpo, val_labels_hpo = None, None, None, None
    if run_hpo or args.mode == 'optimize':
        hpo_sample_size_cfg = int(hpo_config.get('hpo_sample_size', 5000))
        hpo_total_size = min(len(train_val_texts), hpo_sample_size_cfg)
        if hpo_total_size <= 0:
            logger.error(f"Cannot perform HPO with {hpo_total_size} samples.")
            sys.exit(1)

        if hpo_total_size < len(train_val_texts):
             # Stratify split for HPO subset
             hpo_texts, _, hpo_labels, _ = train_test_split(
                 train_val_texts, train_val_labels, train_size=hpo_total_size,
                 random_state=int(global_config.get('random_state', 42)), stratify=train_val_labels
             )
             logger.info(f"Using {hpo_total_size} samples for HPO train/validation.")
        else:
             hpo_texts, hpo_labels = train_val_texts, train_val_labels

        # Split HPO data into train/validation, ensuring stratification if possible
        try:
            if len(set(hpo_labels)) > 1: # Check if stratification is possible
                train_texts_hpo, val_texts_hpo, train_labels_hpo, val_labels_hpo = train_test_split(
                    hpo_texts, hpo_labels,
                    test_size=float(global_config.get('hpo_validation_size', 0.25)),
                    random_state=int(global_config.get('random_state', 42)),
                    stratify=hpo_labels
                )
            else: # Cannot stratify if only one class present
                 logger.warning("Only one class present in HPO data subset, cannot stratify HPO train/val split.")
                 train_texts_hpo, val_texts_hpo, train_labels_hpo, val_labels_hpo = train_test_split(
                    hpo_texts, hpo_labels,
                    test_size=float(global_config.get('hpo_validation_size', 0.25)),
                    random_state=int(global_config.get('random_state', 42))
                 )
        except Exception as hpo_split_err:
            logger.error(f"Failed to split HPO data: {hpo_split_err}", exc_info=True)
            sys.exit(1)

        if not train_texts_hpo or not val_texts_hpo:
            logger.error("HPO train or validation set is empty after splitting. Check sample sizes and splits.")
            sys.exit(1)
        logger.info(f"HPO Train size: {len(train_texts_hpo)}, HPO Validation size: {len(val_texts_hpo)}")
    logger.info(f"Final Test size: {len(test_texts)}")


    best_params_from_hpo = {}
    study = None
    # --- Optuna Optimization ---
    if args.mode == 'optimize' and run_hpo:
        logger.info(f"Starting hyperparameter optimization with Optuna for {n_trials} trials...")
        # Add SQLite storage for study persistence (optional but recommended)
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna_lstm_roberta.db')}"
        study = optuna.create_study(
            study_name=f"lstm-roberta-hpo-{time.strftime('%Y%m%d-%H%M')}",
            direction="maximize",
            storage=storage_name,
            load_if_exists=True # Resume study if DB exists
            # Add pruner here if desired, e.g., MedianPruner
            # pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
        )
        logger.info(f"Using Optuna storage: {storage_name}")
        try:
            study.optimize(
                lambda trial: objective(trial, train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo, model_config, hpo_config),
                n_trials=n_trials,
                timeout=None, # Add overall timeout if desired
                gc_after_trial=True # Help manage memory
            )
            logger.info("Optimization finished.")
            logger.info(f"Number of finished trials in this run: {len(study.trials)}") # May differ if resumed
            logger.info(f"Best trial overall (Accuracy): {study.best_value:.4f}")
            logger.info("Best hyperparameters overall:")
            best_params_from_hpo = study.best_params
            for key, value in best_params_from_hpo.items():
                logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            logger.warning("Proceeding with default/config parameters for final training.")
            best_params_from_hpo = {} # Reset to avoid using partial results

    # Determine final parameters (start with config defaults, update with HPO results)
    final_params = {**model_config} # Start with potentially type-cast defaults from config
    final_params.update(best_params_from_hpo) # Update with HPO results (Optuna returns correct types)

    # Ensure final parameters have correct types *before* instantiation
    float_keys_final = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
    int_keys_final = ['hidden_dim', 'num_layers', 'batch_size', 'max_length', 'num_attention_heads', 'scheduler_warmup_steps', 'gradient_accumulation_steps']
    bool_keys_final = ['freeze_roberta', 'use_layer_norm', 'use_residual', 'use_pooler_output', 'use_scheduler']

    # Use .get with default from model_config itself to avoid KeyErrors if HPO didn't run/tune everything
    for key in float_keys_final:
        final_params[key] = float(final_params.get(key, model_config.get(key)))
    for key in int_keys_final:
        final_params[key] = int(final_params.get(key, model_config.get(key)))
    for key in bool_keys_final:
        final_params[key] = bool(final_params.get(key, model_config.get(key)))


    # --- Train Final Model or Test ---
    model_to_evaluate = None
    final_training_time = None
    if args.mode == 'optimize' or args.mode == 'train':
        logger.info("Training final model with best/specified hyperparameters...")
        log_params = {k: final_params.get(k) for k in ['learning_rate', 'hidden_dim', 'num_layers', 'dropout', 'weight_decay', 'num_attention_heads']}
        logger.info(f"Using parameters: {log_params}")

        try:
            final_model = HybridSentimentClassifier(
                roberta_model=str(final_params['roberta_model']),
                hidden_dim=final_params['hidden_dim'],
                num_layers=final_params['num_layers'],
                dropout=final_params['dropout'],
                learning_rate=final_params['learning_rate'],
                batch_size=final_params['batch_size'],
                max_length=final_params['max_length'],
                num_epochs=final_epochs, # Use final_epochs value
                freeze_roberta=final_params['freeze_roberta'],
                device=DEVICE,
                num_attention_heads=final_params.get('num_attention_heads', 4),
                use_layer_norm=final_params.get('use_layer_norm', True),
                use_residual=final_params.get('use_residual', True),
                use_pooler_output=final_params.get('use_pooler_output', False),
                weight_decay=final_params.get('weight_decay', 0.01),
                use_scheduler=final_params.get('use_scheduler', True),
                scheduler_warmup_steps=final_params.get('scheduler_warmup_steps', 100),
                gradient_accumulation_steps=final_params.get('gradient_accumulation_steps', 2),
                max_grad_norm=final_params.get('max_grad_norm', 1.0)
            )

            start_time = time.time()
            # Use the full train_val dataset for final training
            final_model.train(train_val_texts, train_val_labels, None, None) # No validation needed here
            final_training_time = time.time() - start_time
            logger.info(f"Final model training time: {final_training_time:.2f}s")

            final_model.save_model(model_save_path)
            logger.info(f"Final model saved to {model_save_path}")
            model_to_evaluate = final_model
        except Exception as train_err:
             logger.error(f"Failed during final model training: {train_err}", exc_info=True)
             model_to_evaluate = None # Cannot evaluate if training failed

    elif args.mode == 'test':
        logger.info("Loading model for testing...")
        if os.path.exists(model_save_path):
             model_config_for_load = CONFIG.get('lstm_roberta', {})
             try:
                 # Instantiate with config defaults/architecture params before loading state dict
                 model_to_evaluate = HybridSentimentClassifier(
                     roberta_model=model_config_for_load.get('roberta_model', 'roberta-base'),
                     hidden_dim=int(model_config_for_load.get('hidden_dim', 256)),
                     num_layers=int(model_config_for_load.get('num_layers', 2)),
                     num_attention_heads=int(model_config_for_load.get('num_attention_heads', 4)),
                     use_layer_norm=bool(model_config_for_load.get('use_layer_norm', True)),
                     use_residual=bool(model_config_for_load.get('use_residual', True)),
                     use_pooler_output=bool(model_config_for_load.get('use_pooler_output', False)),
                     # Include other necessary ARCHITECTURE params from config
                     device=DEVICE
                 )
                 model_to_evaluate.load_model(model_save_path)
                 logger.info(f"Model loaded from {model_save_path}")
             except Exception as e:
                 logger.error(f"Failed to load model state from {model_save_path}: {e}", exc_info=True)
                 model_to_evaluate = None
        else:
            logger.error(f"No model found at {model_save_path}. Cannot test.")
            return

    # --- Evaluate on the final Test Set ---
    if model_to_evaluate:
        logger.info("Evaluating final model on the test set...")
        try:
            test_predictions_raw = model_to_evaluate.predict(test_texts)
            test_pred_labels = [1 if pred['label'] == 'Positive' else 0 for pred in test_predictions_raw]

            accuracy = accuracy_score(test_labels, test_pred_labels)
            precision = precision_score(test_labels, test_pred_labels, zero_division=0)
            recall = recall_score(test_labels, test_pred_labels, zero_division=0)
            f1 = f1_score(test_labels, test_pred_labels, zero_division=0)
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

            logger.info("\nFinal Test Set Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            try:
                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                cm = sk_confusion_matrix(test_labels, test_pred_labels)
                cm_path = os.path.join(results_dir, 'lstm_roberta_test_confusion_matrix.png')
                plot_confusion_matrix(cm, output_path=cm_path)
            except ImportError:
                 logger.warning("Scikit-learn not fully available. Cannot generate confusion matrix.")
            except Exception as e:
                logger.error(f"Could not generate/save confusion matrix: {e}")

            # Save final results
            params_to_save = {
                'mode': args.mode,
                'base_sample_size': base_sample_size,
                'final_epochs': final_epochs,
                 # Log the final parameters used for the evaluated model
                **{k: v for k, v in final_params.items() if k != 'hpo'} # Exclude HPO sub-dict
            }
            if final_training_time is not None:
                 params_to_save['final_training_time_seconds'] = round(final_training_time, 2)
            if args.mode == 'optimize' and study:
                 params_to_save['hpo_n_trials_run'] = len(study.trials)
                 params_to_save['hpo_best_val_accuracy'] = study.best_value
                 # Log best params found separately for clarity
                 params_to_save['hpo_best_params'] = study.best_params


            save_model_results(
                model_name="LSTMRoBERTa_Final",
                metrics=metrics,
                parameters=params_to_save,
                example_predictions={} # Add examples if needed
            )
            logger.info("Final results saved.")
        except Exception as eval_err:
             logger.error(f"Failed during final evaluation: {eval_err}", exc_info=True)

    else:
        logger.warning("No model was available for final evaluation.")


if __name__ == "__main__":
    # Suppress TensorFlow/XLA/oneDNN informational messages if TF is installed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 = INFO, 2 = WARNING, 3 = ERROR
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN opts if causing issues
    main()