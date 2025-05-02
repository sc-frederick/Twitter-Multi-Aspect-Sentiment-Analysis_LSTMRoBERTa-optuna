# **LSTM-RoBERTa & HAN-RoBERTa Sentiment Analysis**

This repository contains implementations of two hybrid models for performing sentiment analysis on text data, specifically designed for datasets like Twitter posts:

1. **LSTM-RoBERTa:** Combines Long Short-Term Memory (LSTM) layers with RoBERTa embeddings.  
2. **HAN-RoBERTa:** Combines a Hierarchical Attention Network (HAN) inspired mechanism with RoBERTa embeddings.

Both models leverage the contextual embeddings from RoBERTa and sequence modeling capabilities (LSTM or GRU+Attention) implemented in PyTorch.

The project includes a data processing pipeline that automatically downloads a dataset from Kaggle, preprocesses the text, and stores it locally in an SQLite database for efficient access. It also features hyperparameter optimization using Optuna to fine-tune the models for optimal performance.

## **Features**

* **Hybrid Models:**  
  * LSTM-RoBERTa: Combines RoBERTa embeddings with LSTM layers.  
  * HAN-RoBERTa: Combines RoBERTa embeddings with a GRU/LSTM layer followed by an Attention mechanism.  
* **PyTorch Implementation:** Built using PyTorch and the Hugging Face Transformers library.  
* **Automated Data Handling:** Downloads dataset from Kaggle (zphudzz/tweets-clean-posneg-v1), preprocesses text, and manages data via SQLite.  
* **Hyperparameter Optimization:** Integrated Optuna for efficient hyperparameter search (learning rate, hidden dimensions, dropout, etc.) for both model types.  
* **Configuration Driven:** Uses a config.yaml file for easy management of paths, model parameters (separate sections for LSTM and HAN), and optimization settings.  
* **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if detected for faster training and inference.  
* **Result Tracking:** Saves model checkpoints, evaluation metrics, and optimization results separately for each model type where applicable.  
* **Fine-tuning Mode:** Allows loading a previously trained model and continuing training on a new (potentially larger) dataset.  
* **Sequential Runner:** Includes scripts (run\_all\_models.sh/.bat) to execute both model pipelines back-to-back.

## **Prerequisites**

* **Python:** Python 3.8+ (3.9, 3.10, or 3.11 recommended)  
* **Package Manager:** pip  
* **CUDA (for GPU):** NVIDIA GPU drivers and a compatible CUDA Toolkit version installed and configured (including potentially setting the CUDA\_HOME environment variable).  
* **Kaggle Account & API Credentials:** Needed for automatic dataset download. Configure your credentials (typically \~/.kaggle/kaggle.json).  
* **Key Python Packages:** See setup.py or requirements.txt (includes torch, transformers, optuna, nltk, pandas, scikit-learn, pyyaml, kagglehub, etc.)

## **Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/yourusername/your-repo-name.git\](https://github.com/yourusername/your-repo-name.git) \# Replace with your repo URL  
   cd your-repo-name

2. **Create a virtual environment (Recommended):**  
   python \-m venv venv  
   source venv/bin/activate \# Linux/macOS  
   \# venv\\Scripts\\activate \# Windows

3. **Install dependencies (using setup.py is preferred):**  
   pip install \-e .  
   \# OR if you have a requirements.txt:  
   \# pip install \-r requirements.txt

4. **Download NLTK data:** Run Python interpreter:  
   import nltk  
   nltk.download('punkt')  
   nltk.download('stopwords')  
   nltk.download('wordnet')  
   exit()

## **Dataset Setup**

The first time you run either main script (src/scripts/lstm\_roberta\_main.py or src/scripts/han\_roberta\_main.py), the DataProcessor will attempt to:

1. Download the zphudzz/tweets-clean-posneg-v1 dataset from Kaggle using kagglehub. Ensure your Kaggle API token is correctly set up (usually by placing kaggle.json in \~/.kaggle/).  
2. Preprocess the text data (cleaning, tokenizing, etc.).  
3. Save the processed data into an SQLite database specified by database\_path in src/config.yaml.

Subsequent runs will load data directly from the SQLite database, skipping the download and preprocessing steps.

## **Configuration (src/config.yaml)**

Most parameters are controlled via the src/config.yaml file. Key sections include:

* **Global Settings:** Project name, paths for results/models, random seed.  
* **Data Settings:** Train/test split sizes.  
* **lstm\_roberta:** Configuration specific to the LSTM-RoBERTa model (architecture, training defaults, HPO settings).  
* **han\_roberta:** Configuration specific to the HAN-RoBERTa model (architecture, training defaults, HPO settings).

Each model section contains:  
\- Model Defaults: Architecture parameters (base RoBERTa model, hidden dims, layers, dropout, etc.), training defaults (learning rate, batch size, etc.), freezing RoBERTa layers.  
\- Data Loading: Default sample size to load from the dataset.  
\- hpo (Hyperparameter Optimization):  
\- enabled: Set to true to run Optuna when using \--mode optimize.  
\- n\_trials: Number of optimization trials.  
\- hpo\_sample\_size: How much data to use during HPO trials (subset of loaded data).  
\- hpo\_epochs: Epochs per HPO trial.  
\- search\_space: Defines the ranges/choices for hyperparameters Optuna will tune for that specific model.  
Modify this file to change default behaviors, paths, or the HPO search space for each model.

## **Usage**

The main scripts are src/scripts/lstm\_roberta\_main.py and src/scripts/han\_roberta\_main.py. They operate in different modes specified by the \--mode argument.

**Common Modes:**

**1\. Hyperparameter Optimization (--mode optimize)**

Runs Optuna to find the best hyperparameters for the respective model, using settings from its section in config.yaml. After optimization, it trains a final model using the best found parameters on the full training set and evaluates it on the test set.

\# Optimize LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode optimize

\# Optimize HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode optimize

\# Optional Overrides (Example for LSTM):  
python src/scripts/lstm\_roberta\_main.py \--mode optimize \--n\_trials 15 \--sample\_size 20000

**2\. Training (--mode train)**

Skips Optuna and trains a single model using the default parameters specified in its config.yaml section. Trains on the full training set and evaluates on the test set.

\# Train LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode train

\# Train HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode train

\# Optional Overrides (Example for HAN):  
python src/scripts/han\_roberta\_main.py \--mode train \--final\_epochs 5 \--sample\_size 50000

**3\. Testing (--mode test)**

Loads the last saved final model for the respective type (e.g., lstm\_roberta\_model\_final.pt or han\_roberta\_model\_final.pt) and evaluates it on the test set.

\# Test LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode test

\# Test HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode test

\# Optional Override (Example for LSTM): Test a specific model checkpoint  
python src/scripts/lstm\_roberta\_main.py \--mode test \--load\_model\_path src/models/some\_other\_lstm\_model.pt

**4\. Fine-tuning (--mode finetune)**

Loads a previously saved model checkpoint and continues training on a new dataset.

\# Fine-tune a saved LSTM model on a new dataset  
python src/scripts/lstm\_roberta\_main.py \--mode finetune \\  
    \--load\_model\_path src/models/lstm\_roberta\_model\_final.pt \\  
    \--finetune\_data\_path /path/to/new\_larger\_dataset.db \\  
    \--finetune\_epochs 4 \\  
    \--save\_finetuned\_model\_path src/models/lstm\_roberta\_finetuned\_large.pt

\# Fine-tune a saved HAN model on a new dataset  
python src/scripts/han\_roberta\_main.py \--mode finetune \\  
    \--load\_model\_path src/models/han\_roberta\_model\_final.pt \\  
    \--finetune\_data\_path /path/to/new\_larger\_dataset.db \\  
    \--finetune\_epochs 4 \\  
    \--save\_finetuned\_model\_path src/models/han\_roberta\_finetuned\_large.pt

**Running Both Models Sequentially**

Use the provided wrapper scripts (run\_all\_models.sh for Linux/macOS, run\_all\_models.bat for Windows) located in the project root. Pass any arguments intended for the underlying Python scripts directly to the wrapper script.

\# Example: Train both models for 5 epochs on Linux/macOS  
chmod \+x run\_all\_models.sh \# Make executable first  
./run\_all\_models.sh \--mode train \--final\_epochs 5

\# Example: Optimize both models on Windows  
run\_all\_models.bat \--mode optimize \--n\_trials 25

**GPU Support**

The scripts will automatically use an available NVIDIA GPU (CUDA) if detected by PyTorch. Ensure your drivers and CUDA toolkit are correctly installed. To force CPU usage:

\# Linux/macOS  
export CUDA\_VISIBLE\_DEVICES=""  
python src/scripts/lstm\_roberta\_main.py \--mode train

\# Windows (Command Prompt)  
set CUDA\_VISIBLE\_DEVICES=""  
python src\\scripts\\lstm\_roberta\_main.py \--mode train

\# Windows (PowerShell)  
$env:CUDA\_VISIBLE\_DEVICES=""  
python src\\scripts\\lstm\_roberta\_main.py \--mode train

**Output**

* **Models:** Trained models are saved in the directory specified by models\_dir in config.yaml (default: src/models/), with distinct names like lstm\_roberta\_model\_final.pt and han\_roberta\_model\_final.pt.  
* **Results:** Evaluation metrics and parameters are logged to src/model\_results.json. Entries are identified by model name (e.g., LSTMRoBERTa\_Train, HANRoBERTa\_Optimize). Confusion matrix plots are saved to the results\_dir (default: src/results/) with model-specific names.  
* **Optuna Databases:** Optimization studies are stored in separate SQLite databases in the results\_dir (e.g., optuna\_lstm\_roberta.db, optuna\_han\_roberta.db).  
* **Logs:** Console output provides detailed logging. Redirect to a file if needed: python ... \> run.log 2\>&1.