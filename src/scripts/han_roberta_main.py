# src/scripts/han_roberta_main.py (Adapted for HAN-RoBERTa)

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
    # Import the new HAN-RoBERTa classifier and compatible dataset
    from utils.han_roberta_classifier import HANRoBERTaSentimentClassifier, SentimentDataset
    from utils.results_tracker import save_model_results
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Please ensure 'utils' directory contains necessary files (han_roberta_classifier.py, etc.) and is importable.")
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

        # --- Type Casting for HAN-RoBERTa Section ---
        # Adapt this based on the keys you add to config.yaml for 'han_roberta'
        if 'han_roberta' in config:
            hr_cfg = config['han_roberta']
            float_keys_model = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
            int_keys_model = ['hidden_dim', 'num_layers', 'batch_size', 'max_length',
                             'scheduler_warmup_steps', 'gradient_accumulation_steps',
                             'final_epochs'] # Removed num_attention_heads
            bool_keys_model = ['freeze_roberta', 'use_layer_norm', 'use_gru', # Added use_gru
                              'use_scheduler'] # Removed use_residual, use_pooler_output

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

            # Cast HPO settings (assuming similar structure)
            if 'hpo' in hr_cfg:
                hpo_cfg = hr_cfg['hpo']
                int_keys_hpo = ['n_trials', 'timeout_per_trial', 'hpo_sample_size', 'hpo_epochs']
                for key in int_keys_hpo:
                    if key in hpo_cfg:
                        try: hpo_cfg[key] = int(hpo_cfg[key])
                        except (ValueError, TypeError): logger.warning(f"Could not convert config key 'han_roberta.hpo.{key}' to int. Value: {hpo_cfg[key]}")
                # Cast search space bounds (assuming similar structure)
                if 'search_space' in hpo_cfg:
                    for param, spec in hpo_cfg['search_space'].items():
                        spec_type = spec.get('type')
                        if spec_type == 'float' or spec_type == 'int':
                             for bound in ['low', 'high', 'step']:
                                 if bound in spec:
                                     try: spec[bound] = float(spec[bound])
                                     except (ValueError, TypeError): logger.warning(f"Could not convert search space bound '{param}.{bound}' to float. Value: {spec[bound]}")
        # --- End Type Casting ---

        # Create necessary directories
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
    except Exception as e:
        logger.error(f"Failed to load or process configuration: {e}", exc_info=True)
        sys.exit(1)

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

# --- Objective Function for Optuna (Adapted for HAN-RoBERTa) ---
def objective(trial, train_texts, train_labels, val_texts, val_labels, model_config, hpo_config):
    """Optuna objective function to train and evaluate one HAN-RoBERTa trial."""
    params_to_tune = {}
    search_space = hpo_config.get('search_space', {})

    # Suggest parameters based on search_space defined in config.yaml for han_roberta
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

    try:
        # Ensure key numeric/bool params are correct type before passing to classifier
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

        # --- Instantiate HAN-RoBERTa Model for Trial ---
        model = HANRoBERTaSentimentClassifier(
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
            use_gru=use_gru,
            use_layer_norm=use_layer_norm,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            scheduler_warmup_steps=scheduler_warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm
        )

        # Train and evaluate
        history = model.train(train_texts, train_labels, val_texts, val_labels)
        # Evaluation uses the evaluate method of the classifier
        val_dataset = SentimentDataset(val_texts, val_labels, model.tokenizer, model.max_length)
        val_loader = DataLoader(val_dataset, batch_size=model.batch_size)
        val_loss, val_accuracy = model.evaluate(val_loader)

        logger.info(f"Trial {trial.number} - Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        # Report intermediate value to Optuna for pruning
        trial.report(val_accuracy, step=hpo_config['hpo_epochs'])
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_accuracy # Return metric to maximize

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        return trial.user_attrs.get("last_reported_value", 0.0) # Return last reported or default bad value
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 0.0 # Return a value indicating failure

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Optimize/Train/Test/Finetune HAN-RoBERTa model")
    # Modes and paths are similar to LSTM version
    parser.add_argument( "--mode", type=str, choices=['optimize', 'train', 'test', 'finetune'], default='train', help="Operation mode")
    parser.add_argument("--sample_size", type=int, default=None, help="Override samples loaded for initial train/test/optimize")
    parser.add_argument("--finetune_data_path", type=str, default=None, help="Path to dataset for fine-tuning")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to model checkpoint to load (defaults to final HAN model path)")
    parser.add_argument("--save_finetuned_model_path", type=str, default=None, help="Path to save the fine-tuned HAN model")
    # Training params
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials")
    parser.add_argument("--final_epochs", type=int, default=None, help="Override final training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=3, help="Number of epochs for fine-tuning")

    args = parser.parse_args()

    global_config = CONFIG
    # --- Load HAN-RoBERTa specific config ---
    model_config = CONFIG.get('han_roberta', {}) # <--- Changed section name
    if not model_config:
        logger.error("Configuration section 'han_roberta' not found in config.yaml!")
        sys.exit(1)
    hpo_config = model_config.get('hpo', {})

    # Apply CLI overrides
    base_sample_size = int(args.sample_size if args.sample_size is not None else model_config.get('base_sample_size', 10000))
    n_trials = int(args.n_trials if args.n_trials is not None else hpo_config.get('n_trials', 20))
    final_epochs = int(args.final_epochs if args.final_epochs is not None else model_config.get('final_epochs', 3))
    run_hpo = bool(hpo_config.get('enabled', True)) if args.mode == 'optimize' else False

    # Define paths
    results_dir = os.path.join(src_dir, global_config.get('results_dir', 'results'))
    models_dir = os.path.join(src_dir, global_config.get('models_dir', 'models'))
    # --- Specific model paths for HAN-RoBERTa ---
    default_final_model_path = os.path.join(models_dir, 'han_roberta_model_final.pt') # <--- Changed name
    default_finetuned_model_path = os.path.join(models_dir, 'han_roberta_model_finetuned.pt') # <--- Changed name

    model_load_path = args.load_model_path if args.load_model_path else default_final_model_path
    model_save_path_final = default_final_model_path
    model_save_path_finetuned = args.save_finetuned_model_path if args.save_finetuned_model_path else default_finetuned_model_path

    # Determine DB path (same logic as before)
    db_path_config = global_config.get('database_path', 'data/tweets_dataset.db')
    db_path_initial = os.path.abspath(os.path.join(src_dir, db_path_config)) if not os.path.isabs(db_path_config) else db_path_config
    db_path_to_use = db_path_initial
    if args.mode == 'finetune':
        if not args.finetune_data_path: logger.error("--finetune_data_path required for finetune mode."); sys.exit(1)
        db_path_to_use = os.path.abspath(os.path.join(src_dir, args.finetune_data_path)) if not os.path.isabs(args.finetune_data_path) else args.finetune_data_path
        logger.info(f"Using fine-tuning data path: {db_path_to_use}")
    else: logger.info(f"Using initial data path: {db_path_to_use}")

    # --- Load and Process Data (Common for optimize, train, test) ---
    # Uses the same logic and SentimentDataset as RoBERTa input is similar
    if args.mode in ['optimize', 'train', 'test']:
        logger.info(f"Loading initial data (up to {base_sample_size} samples) from {db_path_initial}...")
        data_processor = DataProcessor(database_path=db_path_initial)
        try: df = data_processor.load_data(sample_size=base_sample_size)
        except Exception as data_err: logger.error(f"Failed to load initial data: {data_err}", exc_info=True); sys.exit(1)
        try: labels = (df['target'].astype(float) == 4.0).astype(int).tolist() # Adapt if target format differs
        except Exception as label_err: logger.error(f"Failed to process 'target' column: {label_err}. Check data.", exc_info=True); sys.exit(1)
        texts = df['text_clean'].tolist()
        try:
            stratify_split = labels if len(set(labels)) > 1 else None
            train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                texts, labels, test_size=float(global_config.get('test_size', 0.2)),
                random_state=int(global_config.get('random_state', 42)), stratify=stratify_split)
            logger.info(f"Initial data split: Train/Val size: {len(train_val_texts)}, Test size: {len(test_texts)}")
        except Exception as split_err: logger.error(f"Failed to split initial data: {split_err}", exc_info=True); sys.exit(1)

        # Prepare HPO data subset (same logic as before)
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

    # --- Optuna Optimization (HAN-RoBERTa) ---
    best_params_from_hpo = {}
    study = None
    if args.mode == 'optimize' and run_hpo:
        logger.info(f"Starting HAN-RoBERTa hyperparameter optimization with Optuna for {n_trials} trials...")
        # --- Use a different DB name for HAN-RoBERTa study ---
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna_han_roberta.db')}" # <--- Changed DB name
        study = optuna.create_study(
            study_name=f"han-roberta-hpo-{time.strftime('%Y%m%d-%H%M')}", direction="maximize",
            storage=storage_name, load_if_exists=True)
        logger.info(f"Using Optuna storage: {storage_name}")
        try:
            study.optimize(
                lambda trial: objective(trial, train_texts_hpo, train_labels_hpo, val_texts_hpo, val_labels_hpo, model_config, hpo_config),
                n_trials=n_trials, timeout=None, gc_after_trial=True)
            logger.info("Optimization finished.")
            logger.info(f"Number of finished trials: {len(study.trials)}")
            logger.info(f"Best trial overall (Accuracy): {study.best_value:.4f}")
            logger.info("Best hyperparameters overall:")
            best_params_from_hpo = study.best_params
            for key, value in best_params_from_hpo.items(): logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            best_params_from_hpo = {}

    # Determine final parameters for 'train' mode
    final_params = {**model_config, **best_params_from_hpo}

    # Ensure final parameters have correct types for 'train' mode
    if args.mode == 'train' or args.mode == 'optimize':
        # Adapt keys based on HAN-RoBERTa config
        float_keys_final = ['learning_rate', 'dropout', 'weight_decay', 'max_grad_norm']
        int_keys_final = ['hidden_dim', 'num_layers', 'batch_size', 'max_length', 'scheduler_warmup_steps', 'gradient_accumulation_steps']
        bool_keys_final = ['freeze_roberta', 'use_layer_norm', 'use_gru', 'use_scheduler']
        try:
            for key in float_keys_final: final_params[key] = float(final_params.get(key, model_config.get(key)))
            for key in int_keys_final: final_params[key] = int(final_params.get(key, model_config.get(key)))
            for key in bool_keys_final:
                 val = final_params.get(key, model_config.get(key))
                 if isinstance(val, str):
                     val_lower = val.lower(); final_params[key] = val_lower in ['true', 'yes', 'on', '1']
                 else: final_params[key] = bool(val)
        except (ValueError, TypeError, KeyError) as e: logger.error(f"Error casting final parameters: {e}", exc_info=True); sys.exit(1)

    # --- Train Final Model (Mode: optimize or train) ---
    model_to_evaluate = None
    final_training_time = None
    if args.mode == 'optimize' or args.mode == 'train':
        logger.info("Training final HAN-RoBERTa model...")
        # Log relevant params
        log_params = {k: final_params.get(k) for k in ['learning_rate', 'hidden_dim', 'num_layers', 'dropout', 'weight_decay', 'freeze_roberta', 'use_gru']}
        logger.info(f"Using parameters: {log_params}")
        try:
            # --- Instantiate Final HAN-RoBERTa Model ---
            final_model = HANRoBERTaSentimentClassifier(
                roberta_model=str(final_params['roberta_model']),
                hidden_dim=final_params['hidden_dim'],
                num_layers=final_params['num_layers'],
                dropout=final_params['dropout'],
                learning_rate=final_params['learning_rate'],
                batch_size=final_params['batch_size'],
                max_length=final_params['max_length'],
                num_epochs=final_epochs,
                freeze_roberta=final_params['freeze_roberta'],
                device=DEVICE,
                use_gru=final_params.get('use_gru', True),
                use_layer_norm=final_params.get('use_layer_norm', True),
                weight_decay=final_params.get('weight_decay', 0.01),
                use_scheduler=final_params.get('use_scheduler', True),
                scheduler_warmup_steps=final_params.get('scheduler_warmup_steps', 100),
                gradient_accumulation_steps=final_params.get('gradient_accumulation_steps', 2),
                max_grad_norm=final_params.get('max_grad_norm', 1.0)
            )
            start_time = time.time()
            logger.info(f"Starting final training on {len(train_val_texts)} samples for {final_epochs} epochs.")
            final_model.train(train_val_texts, train_val_labels, None, None)
            final_training_time = time.time() - start_time
            logger.info(f"Final model training time: {final_training_time:.2f}s")
            # Save the final HAN-RoBERTa model
            final_model.save_model(model_save_path_final)
            logger.info(f"Final HAN-RoBERTa model saved to {model_save_path_final}")
            model_to_evaluate = final_model
        except Exception as train_err:
             logger.error(f"Failed during final HAN-RoBERTa training: {train_err}", exc_info=True)
             model_to_evaluate = None

    # --- Fine-tune Existing Model (Mode: finetune) ---
    elif args.mode == 'finetune':
        logger.info(f"Starting HAN-RoBERTa fine-tuning mode...")
        if not os.path.exists(model_load_path): logger.error(f"Model not found at {model_load_path}. Cannot fine-tune."); sys.exit(1)
        logger.info(f"Loading dataset from {db_path_to_use} for fine-tuning...")
        data_processor_ft = DataProcessor(database_path=db_path_to_use)
        try:
            df_large = data_processor_ft.load_data(sample_size=None)
            # Adapt label processing if needed for the new dataset
            labels_large = (df_large['target'].astype(float) == 4.0).astype(int).tolist()
            texts_large = df_large['text_clean'].tolist()
            logger.info(f"Loaded {len(texts_large)} samples for fine-tuning.")
        except Exception as data_err: logger.error(f"Failed to load fine-tuning data: {data_err}", exc_info=True); sys.exit(1)

        logger.info("Instantiating HAN-RoBERTa structure based on config...")
        # Load config section used for the *original* training of the model being loaded
        # Assuming it was also 'han_roberta', otherwise adjust CONFIG.get()
        model_config_for_load = CONFIG.get('han_roberta', {})
        try:
            # Instantiate with ARCHITECTURE params from config
            model_to_finetune = HANRoBERTaSentimentClassifier(
                 roberta_model=model_config_for_load.get('roberta_model', 'roberta-base'),
                 hidden_dim=int(model_config_for_load.get('hidden_dim', 256)),
                 num_layers=int(model_config_for_load.get('num_layers', 1)),
                 dropout=float(model_config_for_load.get('dropout', 0.3)),
                 use_gru=bool(model_config_for_load.get('use_gru', True)),
                 use_layer_norm=bool(model_config_for_load.get('use_layer_norm', True)),
                 # Pass other necessary params for init, they might be overwritten by loaded state
                 learning_rate=float(model_config_for_load.get('learning_rate', 2e-5)),
                 batch_size=int(model_config_for_load.get('batch_size', 16)),
                 max_length=int(model_config_for_load.get('max_length', 128)),
                 num_epochs=args.finetune_epochs, # Use finetune epochs
                 freeze_roberta=bool(model_config_for_load.get('freeze_roberta', True)), # Use saved model's freeze state
                 device=DEVICE,
                 weight_decay=float(model_config_for_load.get('weight_decay', 0.01)),
                 use_scheduler=bool(model_config_for_load.get('use_scheduler', True)),
                 scheduler_warmup_steps=int(model_config_for_load.get('scheduler_warmup_steps', 100)),
                 gradient_accumulation_steps=int(model_config_for_load.get('gradient_accumulation_steps', 2)),
                 max_grad_norm=float(model_config_for_load.get('max_grad_norm', 1.0))
            )
            logger.info(f"Loading model and optimizer state from {model_load_path}...")
            model_to_finetune.load_model(model_load_path) # Load weights and optimizer state

            logger.info(f"Starting fine-tuning on {len(texts_large)} samples for {args.finetune_epochs} epochs...")
            start_time = time.time()
            model_to_finetune.train(texts_large, labels_large, None, None) # Train on new data
            finetuning_time = time.time() - start_time
            logger.info(f"Fine-tuning completed in {finetuning_time:.2f}s")

            model_to_finetune.save_model(model_save_path_finetuned) # Save fine-tuned model
            logger.info(f"Fine-tuned HAN-RoBERTa model saved to {model_save_path_finetuned}")
            model_to_evaluate = model_to_finetune # Set for potential evaluation
            logger.warning("Evaluation after fine-tuning uses initial test set. Load appropriate test data for proper evaluation.")
        except Exception as ft_err:
             logger.error(f"Error during HAN-RoBERTa fine-tuning: {ft_err}", exc_info=True)
             model_to_evaluate = None

    # --- Load Model for Testing (Mode: test) ---
    elif args.mode == 'test':
        logger.info(f"Loading HAN-RoBERTa model for testing from {model_load_path}...")
        if os.path.exists(model_load_path):
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
                 # Load only model weights for testing
                 checkpoint = torch.load(model_load_path, map_location=DEVICE)
                 model_to_evaluate.model.load_state_dict(checkpoint['model_state_dict'])
                 logger.info(f"HAN-RoBERTa model state loaded successfully from {model_load_path}")
             except Exception as e:
                 logger.error(f"Failed to load HAN-RoBERTa model state: {e}", exc_info=True)
                 model_to_evaluate = None
        else:
            logger.error(f"No HAN-RoBERTa model found at {model_load_path}. Cannot test.")
            model_to_evaluate = None

    # --- Evaluate on the Test Set ---
    # Uses the initial test split (test_texts, test_labels)
    if model_to_evaluate and args.mode != 'finetune':
        if 'test_texts' not in locals() or 'test_labels' not in locals():
             logger.error("Test data not loaded. Cannot evaluate.")
        else:
            logger.info("Evaluating HAN-RoBERTa model on the test set...")
            try:
                test_predictions_raw = model_to_evaluate.predict(test_texts)
                # Assuming predict returns list of dicts {'label': 'Positive'/'Negative', 'confidence': float}
                test_pred_labels = [1 if pred.get('label') == 'Positive' else 0 for pred in test_predictions_raw]

                accuracy = accuracy_score(test_labels, test_pred_labels)
                precision = precision_score(test_labels, test_pred_labels, zero_division=0)
                recall = recall_score(test_labels, test_pred_labels, zero_division=0)
                f1 = f1_score(test_labels, test_pred_labels, zero_division=0)
                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

                logger.info("\nFinal Test Set Evaluation Results (HAN-RoBERTa):")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1 Score: {f1:.4f}")

                try:
                    cm = sk_confusion_matrix(test_labels, test_pred_labels)
                    cm_path = os.path.join(results_dir, f'han_roberta_{args.mode}_confusion_matrix.png') # Changed name
                    plot_confusion_matrix(cm, output_path=cm_path)
                except Exception as e: logger.error(f"Could not generate/save confusion matrix: {e}")

                # Save results, adapting names and params
                params_to_save = {
                    'mode': args.mode,
                    'base_sample_size': base_sample_size if args.mode != 'finetune' else 'N/A',
                    'final_epochs': final_epochs if args.mode == 'train' else 'N/A',
                    'evaluated_model_path': model_load_path if args.mode == 'test' else model_save_path_final,
                    **(final_params if args.mode in ['train', 'optimize'] else model_config_for_load)
                }
                params_to_save.pop('hpo', None) # Remove HPO sub-dict if present
                if final_training_time is not None and args.mode in ['train', 'optimize']:
                     params_to_save['final_training_time_seconds'] = round(final_training_time, 2)
                if args.mode == 'optimize' and study:
                     params_to_save['hpo_n_trials_run'] = len(study.trials)
                     params_to_save['hpo_best_val_accuracy'] = study.best_value
                     params_to_save['hpo_best_params'] = study.best_params

                save_model_results(
                    model_name=f"HANRoBERTa_{args.mode.capitalize()}", # Changed name
                    metrics=metrics, parameters=params_to_save, example_predictions={})
                logger.info("Evaluation results saved.")
            except Exception as eval_err: logger.error(f"Failed during final evaluation: {eval_err}", exc_info=True)

    elif args.mode == 'finetune' and model_to_evaluate:
         logger.info("HAN-RoBERTa Fine-tuning complete. Model saved.")
         logger.warning("Skipping automatic evaluation after fine-tuning.")
    elif not model_to_evaluate and args.mode != 'finetune':
         logger.warning("No HAN-RoBERTa model was available or trained/loaded for evaluation.")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
