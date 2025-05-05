# Hybrid RoBERTa Sentiment Analysis Models

This repository contains PyTorch implementations of four hybrid models for performing 3-class sentiment analysis (Negative, Neutral, Positive) on text data, leveraging RoBERTa embeddings combined with different architectures:

1. **LSTM-RoBERTa:** Combines RoBERTa embeddings with a Bidirectional Long Short-Term Memory (BiLSTM) layer and Multi-Head Attention.
2. **HAN-RoBERTa:** Implements a Hierarchical Attention Network (HAN) inspired mechanism, using RoBERTa embeddings followed by a GRU/LSTM layer and attention.
3. **GNN-RoBERTa:** Utilizes RoBERTa token embeddings as input to Graph Attention Network (GAT)-like layers for classification.
4. **ATAE-RoBERTa:** Adapts the Attention-based LSTM with Aspect Embedding (ATAE) concept, using RoBERTa embeddings followed by a BiLSTM and an Attention layer.

All models leverage the contextual embeddings from RoBERTa and sequence modeling/attention capabilities implemented in PyTorch and the Hugging Face Transformers library.

The project includes an automated data processing pipeline that downloads the `kaggle/tweet-sentiment-extraction` dataset from Kaggle using `kagglehub`, preprocesses the text, maps sentiments ('negative', 'neutral', 'positive') to numerical labels (0, 1, 2), and prepares the data for the models. It also features hyperparameter optimization using Optuna for all four model types and tracks results.

## Features

- **Four Hybrid Models:**
    - LSTM-RoBERTa: RoBERTa + BiLSTM + Multi-Head Attention.
    - HAN-RoBERTa: RoBERTa + GRU/LSTM + Attention.
    - GNN-RoBERTa: RoBERTa + GAT-like Attention Layers.
    - ATAE-RoBERTa: RoBERTa + BiLSTM + Attention.
- **PyTorch Implementation:** Built using PyTorch and the Hugging Face Transformers library.
- **Automated Data Handling:** Downloads the `kaggle/tweet-sentiment-extraction` dataset via `kagglehub`, preprocesses text, and maps sentiment labels.
- **Hyperparameter Optimization:** Integrated Optuna for efficient hyperparameter search (learning rate, hidden dimensions, dropout, etc.) for all four model types.
- **Configuration Driven:** Uses `src/config.yaml` for easy management of paths, global settings, and model-specific parameters (separate sections for `lstm_roberta`, `han_roberta`, `gnn_roberta`, `atae_roberta`), including HPO settings.
- **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if detected for faster training and inference.
- **Result Tracking:** Saves model checkpoints, evaluation metrics (accuracy, precision, recall, F1), and Optuna study results separately for each model type. Results are logged in `src/model_results.json`.
- **Sequential Runner:** Includes a script (`run_all_models.sh`) to execute all four model pipelines sequentially.

## Prerequisites

- **Python:** Python 3.8+ (3.9, 3.10, or 3.11 recommended)
- **Package Manager:** pip
- **CUDA (for GPU):** NVIDIA GPU drivers and a compatible CUDA Toolkit version installed and configured.
- **Kaggle Account & API Credentials:** Needed for automatic dataset download via `kagglehub`. Configure your credentials (typically `~/.kaggle/kaggle.json`).
- **Key Python Packages:** See `requirements.txt` (includes `torch`, `transformers`, `optuna`, `nltk`, `pandas`, `scikit-learn`, `pyyaml`, `kagglehub`, `seaborn`, `matplotlib`, etc.)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/sc-frederick/Twitter-Multi-Aspect-Sentiment-Analysis_LSTMRoBERTa-optuna.git # Replace with your repo URL
cd your-repo-name
```
2. **Create a virtual environment (Recommended):**

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (Command Prompt)
# venv\Scripts\activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt
```
4. **Download NLTK data:** Run Python interpreter:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
exit()
```

## Dataset Setup

The first time you run any main script (`src/scripts/*_main.py`), the `DataProcessor` will attempt to:

1. Download the `kaggle/tweet-sentiment-extraction` dataset from Kaggle using `kagglehub`. Ensure your Kaggle API token is correctly set up. The data will be stored in the `src/data/` directory by default.
2. Load `train.csv` and `test.csv`.
3. Preprocess the text data (cleaning, fixing contractions, etc.).
4. Map the 'sentiment' column ('negative', 'neutral', 'positive') to numerical labels (0, 1, 2) in a new 'label' column.

Subsequent runs will use the already downloaded and processed data if available locally.

## Configuration (`src/config.yaml`)

Most parameters are controlled via the `src/config.yaml` file. Key sections include:

- **Global Settings:** Project name, paths for results/models, random seed, data directory path.
- **Data Settings:** Train/test split sizes (though the primary split comes from `train.csv`/`test.csv`), HPO validation split size (applied to the data loaded from `train.csv`).
- **`lstm_roberta`:** Configuration specific to the LSTM-RoBERTa model.
- **`han_roberta`:** Configuration specific to the HAN-RoBERTa model.
- **`gnn_roberta`:** Configuration specific to the GNN-RoBERTa model.
- **`atae_roberta`:** Configuration specific to the ATAE-RoBERTa model.

Each model section contains:

- **Model Defaults:** Architecture parameters (base RoBERTa model, hidden dims, layers, dropout, heads, etc.), training defaults (learning rate, batch size, epochs, scheduler settings), freezing RoBERTa layers. These defaults may reflect the best parameters found during previous HPO runs.
- **Data Loading:** Default sample size to load from the dataset (`base_sample_size`).
- **`hpo`** (Hyperparameter Optimization):
    - `enabled`: Set to `true` to run Optuna when using `--mode optimize`.
    - `n_trials`: Number of optimization trials.
    - `hpo_sample_size`: How much data (subset of loaded training data) to use during HPO trials.
    - `hpo_epochs`: Epochs per HPO trial.
    - `search_space`: Defines the ranges/choices for hyperparameters Optuna will tune for that specific model.

Modify this file to change default behaviors, paths, or the HPO search space for each model.

## Usage

The main scripts are in `src/scripts/`:

- `lstm_roberta_main.py`
- `han_roberta_main.py`
- `gnn_roberta_main.py`
- `atae_roberta_main.py`

They operate in different modes specified by the `--mode` argument.

<br>

**Common Modes:**

**<br>
**

### **1\. Hyperparameter Optimization (`--mode optimize`)**

Runs Optuna to find the best hyperparameters for the respective model, using settings from its section in `config.yaml`. After optimization, it trains a final model using the best found parameters on the full training set (derived from `train.csv`) and evaluates it on the test set (derived from `test.csv`).

```
# Optimize LSTM-RoBERTa
python src/scripts/lstm_roberta_main.py --mode optimize

# Optimize HAN-RoBERTa
python src/scripts/han_roberta_main.py --mode optimize

# Optimize GNN-RoBERTa
python src/scripts/gnn_roberta_main.py --mode optimize

# Optimize ATAE-RoBERTa
python src/scripts/atae_roberta_main.py --mode optimize

# Optional Overrides (Example for GNN):
python src/scripts/gnn_roberta_main.py --mode optimize --n_trials 15 --sample_size_train 20000
```

<br>

<br>

### 2\. Training (--mode train)

Skips Optuna and trains a single model using the default parameters specified in its config.yaml section (or HPO best params if optimization was run previously and defaults weren't manually reset). Trains on the full training set (from train.csv, potentially sampled) and evaluates on the test set (from test.csv, potentially sampled).  

```
# Train LSTM-RoBERTa
python src/scripts/lstm_roberta_main.py --mode train

# Train HAN-RoBERTa
python src/scripts/han_roberta_main.py --mode train

# Train GNN-RoBERTa
python src/scripts/gnn_roberta_main.py --mode train

# Train ATAE-RoBERTa
python src/scripts/atae_roberta_main.py --mode train

# Optional Overrides (Example for ATAE):
python src/scripts/atae_roberta_main.py --mode train --final_epochs 5 --sample_size_train 50000
```

<br>

<br>
<br>
<br>

### 3\. Testing (--mode test)

Loads the last saved final model for the respective type (e.g., lstm\_roberta\_model\_final.pt, han\_roberta\_model\_final.pt, etc.) from the src/models/ directory and evaluates it on the test set (from test.csv).  

```
# Test LSTM-RoBERTa
python src/scripts/lstm_roberta_main.py --mode test

# Test HAN-RoBERTa
python src/scripts/han_roberta_main.py --mode test

# Test GNN-RoBERTa
python src/scripts/gnn_roberta_main.py --mode test

# Test ATAE-RoBERTa
python src/scripts/atae_roberta_main.py --mode test

# Optional Override (Example for LSTM): Test a specific model checkpoint
# python src/scripts/lstm_roberta_main.py --mode test --load_model_path src/models/some_other_lstm_model.pt
# (Note: Loading specific paths might require code adjustments if not fully implemented)
```

<br>

<br>
<br>
<br>

### Running All Models Sequentially

Use the provided wrapper script (run\_all\_models.sh for Linux/macOS) located in the project root. Pass any arguments intended for the underlying Python scripts directly to the wrapper script.  

```
# Example: Train all models for 5 epochs on Linux/macOS
chmod +x run_all_models.sh # Make executable first
./run_all_models.sh --mode train --final_epochs 5

# Example: Optimize all models with 25 trials
./run_all_models.sh --mode optimize --n_trials 25
```

<br>

<br>
<br>
<br>

### GPU Support

The scripts will automatically use an available NVIDIA GPU (CUDA) if detected by PyTorch. Ensure your drivers and CUDA toolkit are correctly installed. To force CPU usage:  

```
# Linux/macOS
export CUDA_VISIBLE_DEVICES=""
python src/scripts/lstm_roberta_main.py --mode train

# Windows (Command Prompt)
set CUDA_VISIBLE_DEVICES=""
python src\scripts\lstm_roberta_main.py --mode train

# Windows (PowerShell)
$env:CUDA_VISIBLE_DEVICES=""
python src\scripts\lstm_roberta_main.py --mode train
```

<br>

<br>
<br>
<br>

### Output

```
Models: Trained models are saved in the directory specified by models_dir in config.yaml (default: src/models/), with distinct names like lstm_roberta_model_final.pt, han_roberta_model_final.pt, gnn_roberta_model_final.pt, atae_roberta_model_final.pt.

Results: Evaluation metrics (accuracy, precision, recall, F1) and parameters are logged to src/model_results.json. Entries are identified by model name and mode (e.g., LSTMRoBERTa_Train, HANRoBERTa_Optimize, GNNRoBERTa_Test, ATAERoBERTa_Train). Confusion matrix plots are saved to the results_dir (default: src/results/) with model-specific names (e.g., lstm_roberta_test_confusion_matrix.png).

Optuna Databases: Optimization studies are stored in separate SQLite databases in the results_dir (e.g., optuna_lstm_roberta.db, optuna_han_roberta.db, optuna_gnn_roberta.db, optuna_atae_roberta.db).
```

<br>