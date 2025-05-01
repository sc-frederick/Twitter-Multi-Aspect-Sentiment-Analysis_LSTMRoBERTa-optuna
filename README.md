# LSTM-RoBERTa Sentiment Analysis

This repository contains a hybrid Long Short-Term Memory (LSTM) and RoBERTa model for performing sentiment analysis on text data, specifically designed for datasets like Twitter posts. It leverages the contextual embeddings from RoBERTa and the sequence modeling capabilities of LSTM, implemented in PyTorch.

The project includes a data processing pipeline that automatically downloads a dataset from Kaggle, preprocesses the text, and stores it locally in an SQLite database for efficient access. It also features hyperparameter optimization using Optuna to fine-tune the model for optimal performance.

## Features

- **Hybrid Model:** Combines RoBERTa embeddings with LSTM layers for sentiment classification.
- **PyTorch Implementation:** Built using PyTorch and the Hugging Face Transformers library.
- **Automated Data Handling:** Downloads dataset from Kaggle (`zphudzz/tweets-clean-posneg-v1`), preprocesses text, and manages data via SQLite.
- **Hyperparameter Optimization:** Integrated Optuna for efficient hyperparameter search (learning rate, hidden dimensions, dropout, etc.).
- **Configuration Driven:** Uses a `config.yaml` file for easy management of paths, model parameters, and optimization settings.
- **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if detected for faster training and inference.
- **Result Tracking:** Saves model checkpoints, evaluation metrics, and optimization results.

## Prerequisites

- **Python:** Python 3.8+ (3.9, 3.10, or 3.11 recommended)
- **Package Manager:** `pip`
- **CUDA (for GPU):** NVIDIA GPU drivers and a compatible CUDA Toolkit version installed and configured (including potentially setting the `CUDA_HOME` environment variable).
- **Kaggle Account & API Credentials:** Needed for automatic dataset download. Configure your credentials (typically `~/.kaggle/kaggle.json`).
- **Key Python Packages:** See `requirements.txt` (includes `torch`, `transformers`, `optuna`, `nltk`, `pandas`, `scikit-learn`, `pyyaml`, `kagglehub`, etc.)

## Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/sc-frederick/Twitter-Multi-Aspect-Sentiment-Analysis_LSTMRoBERTa-optuna.git](https://github.com/sc-frederick/Twitter-Multi-Aspect-Sentiment-Analysis_LSTMRoBERTa-optuna.git)
cd Twitter-Multi-Aspect-Sentiment-Analysis_LSTMRoBERTa-optuna
```


2. **Create a virtual environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows
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

The first time you run the script (`src/scripts/lstm_roberta_main.py`), the `DataProcessor` will attempt to:

1. Download the `zphudzz/tweets-clean-posneg-v1` dataset from Kaggle using `kagglehub`. Ensure your Kaggle API token is correctly set up (usually by placing `kaggle.json` in `~/.kaggle/`).
2. Preprocess the text data (cleaning, tokenizing, etc.).
3. Save the processed data into an SQLite database specified by `database_path` in `src/config.yaml`.

Subsequent runs will load data directly from the SQLite database, skipping the download and preprocessing steps.

## Configuration (`src/config.yaml`)

Most parameters are controlled via the `src/config.yaml` file. Key sections include:

- **Global Settings:** Project name, paths for results/models, random seed.
- **Data Settings:** Train/test split sizes.
- **`lstm_roberta`:**
    - **Model Defaults:** Architecture parameters (base RoBERTa model, hidden dims, layers, dropout, attention heads), training defaults (learning rate, batch size, etc.), freezing RoBERTa layers.
    - **Data Loading:** Default sample size to load from the dataset.
    - **`hpo` (Hyperparameter Optimization):**
        - `enabled`: Set to `true` to run Optuna when using `--mode optimize`.
        - `n_trials`: Number of optimization trials.
        - `hpo_sample_size`: How much data to use during HPO trials (subset of loaded data).
        - `hpo_epochs`: Epochs per HPO trial.
        - `search_space`: Defines the ranges/choices for hyperparameters Optuna will tune.

Modify this file to change default behaviors, paths, or the HPO search space.

## Usage

The main script is `src/scripts/lstm_roberta_main.py`. It operates in different modes specified by the `--mode` argument.

**1\. Hyperparameter Optimization (`--mode optimize`)**

This mode runs Optuna to find the best hyperparameters based on the validation set performance, using the settings defined in `config.yaml` under `lstm_roberta.hpo`. After optimization, it trains a final model using the best found parameters on the full training set and evaluates it on the test set.

```
bash
python src/scripts/lstm_roberta_main.py --mode optimize
Optional Overrides:# Run fewer trials, use a larger sample during HPO
python src/scripts/lstm_roberta_main.py --mode optimize --n_trials 15 --sample_size 20000
(Note: --sample_size here overrides base_sample_size from config, affecting the initial data load. hpo_sample_size from config still determines the subset used during actual HPO trials).

```
**2\. Training (`--mode train`)**
This mode skips Optuna and trains a single model using the default parameters specified in config.yaml (or overridden by Optuna's best results if optimization was run previously and saved state implicitly). It trains on the full training set and evaluates on the test set.
```
python src/scripts/lstm_roberta_main.py --mode train
Optional Overrides:# Train for more epochs using a specific sample size
python src/scripts/lstm_roberta_main.py --mode train --final_epochs 5 --sample_size 50000
```
**3\. Testing (`--mode test`)**
This mode loads the last saved final model (lstm_roberta_model_final.pt from the configured models_dir) and evaluates it on the test set.
```
python src/scripts/lstm_roberta_main.py --mode test
Optional Override:# Test using a different sample size from the dataset for evaluation
python src/scripts/lstm_roberta_main.py --mode test --sample_size 10000
```
**GPU Support**
The script will automatically use an available NVIDIA GPU (CUDA) if detected by PyTorch. Ensure your drivers and CUDA toolkit are correctly installed and configured in your environment. If you encounter CUDA-related errors during compilation steps (often from the transformers library), make sure the CUDA_HOME environment variable is set correctly before running the script. 
To force CPU usage (for debugging or if GPU setup is problematic):
```
:export CUDA_VISIBLE_DEVICES=""
python src/scripts/lstm_roberta_main.py --mode train
```
**OutputModels:**
Trained models are saved in the directory specified by models_dir in config.yaml (default: src/models/), typically as lstm_roberta_model_final.pt.


**Results:**
Evaluation metrics and parameters are logged to src/model_results.json and src/model_results.csv via results_tracker.py. Confusion matrix plots are saved to the results_dir (default: src/results/).


**Optuna Database:**
If optimization is run, the study results are stored in an SQLite database in the results_dir (e.g., src/results/optuna_lstm_roberta.db).


**Logs:**
 Console output provides detailed logging of the process. You can redirect this to a file when running on a server: 
 ```
 python ... > run.log 2>&1.
```