# **Hybrid RoBERTa Sentiment Analysis Models**

This repository contains implementations of four hybrid models for performing sentiment analysis on text data, specifically designed for datasets like Twitter posts:

1. **LSTM-RoBERTa:** Combines Long Short-Term Memory (LSTM) layers with RoBERTa embeddings and multi-head attention.  
2. **HAN-RoBERTa:** Combines a Hierarchical Attention Network (HAN) inspired mechanism (GRU/LSTM \+ Attention) with RoBERTa embeddings.  
3. **GNN-RoBERTa:** Implements a Graph Attention Network (GAT)-like mechanism over RoBERTa token embeddings.  
4. **ATAE-RoBERTa:** Adapts the Attention-based LSTM with Aspect Embedding (ATAE) concept, using RoBERTa embeddings followed by a BiLSTM and Attention layer.

All models leverage the contextual embeddings from RoBERTa and sequence modeling/attention capabilities implemented in PyTorch.

The project includes a data processing pipeline that automatically downloads a dataset from Kaggle, preprocesses the text, and stores it locally in an SQLite database for efficient access. It also features hyperparameter optimization using Optuna to fine-tune the models for optimal performance.

## **Features**

* **Hybrid Models:**  
  * LSTM-RoBERTa: RoBERTa \+ BiLSTM \+ Multi-Head Attention.  
  * HAN-RoBERTa: RoBERTa \+ GRU/LSTM \+ Attention.  
  * GNN-RoBERTa: RoBERTa \+ GAT-like Attention Layers. - GNN IS BORKED Right now, I have the fix on a remote machine - too tired to transfer it over and fix it in the repo. Will do it tomorrow at some point
  * ATAE-RoBERTa: RoBERTa \+ BiLSTM \+ Attention.  
* **PyTorch Implementation:** Built using PyTorch and the Hugging Face Transformers library.  
* **Automated Data Handling:** Downloads dataset from Kaggle (zphudzz/tweets-clean-posneg-v1), preprocesses text, and manages data via SQLite.  
* **Hyperparameter Optimization:** Integrated Optuna for efficient hyperparameter search (learning rate, hidden dimensions, dropout, etc.) for all four model types.  
* **Configuration Driven:** Uses a config.yaml file for easy management of paths, model parameters (separate sections for LSTM, HAN, GNN, ATAE), and optimization settings.  
* **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if detected for faster training and inference.  
* **Result Tracking:** Saves model checkpoints, evaluation metrics, and optimization results separately for each model type.  
* **Fine-tuning Mode:** Allows loading a previously trained model (currently implemented for HAN-RoBERTa) and continuing training on a new dataset. *(Note: Fine-tuning might need adaptation for other models)*.  
* **Sequential Runner:** Includes scripts (run\_all\_models.sh/.bat) to execute all four model pipelines back-to-back.

## **Prerequisites**

* **Python:** Python 3.8+ (3.9, 3.10, or 3.11 recommended)  
* **Package Manager:** pip  
* **CUDA (for GPU):** NVIDIA GPU drivers and a compatible CUDA Toolkit version installed and configured (including potentially setting the CUDA\_HOME environment variable).  
* **Kaggle Account & API Credentials:** Needed for automatic dataset download. Configure your credentials (typically \~/.kaggle/kaggle.json).  
* **Key Python Packages:** See requirements.txt (includes torch, transformers, optuna, nltk, pandas, scikit-learn, pyyaml, kagglehub, etc.)

## **Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/yourusername/your-repo-name.git\] \# Replace with your repo URL  
   cd your-repo-name

2. **Create a virtual environment (Recommended):**  
   python \-m venv venv  
   \# Linux/macOS  
   source venv/bin/activate  
   \# Windows (Command Prompt)  
   \# venv\\Scripts\\activate  
   \# Windows (PowerShell)  
   \# .\\venv\\Scripts\\Activate.ps1

3. **Install dependencies:**  
   pip install \-r requirements.txt

   *(Using pip install \-e . might also work if a setup.py is present and configured)*  
4. **Download NLTK data:** Run Python interpreter:  
   import nltk  
   nltk.download('punkt')  
   nltk.download('stopwords')  
   nltk.download('wordnet')  
   exit()

## **Dataset Setup**

The first time you run any main script (src/scripts/\*\_main.py), the DataProcessor will attempt to:

1. Download the zphudzz/tweets-clean-posneg-v1 dataset from Kaggle using kagglehub. Ensure your Kaggle API token is correctly set up (usually by placing kaggle.json in \~/.kaggle/).  
2. Preprocess the text data (cleaning, tokenizing, etc.).  
3. Save the processed data into an SQLite database specified by database\_path in src/config.yaml.

Subsequent runs will load data directly from the SQLite database, skipping the download and preprocessing steps.

## **Configuration (src/config.yaml)**

Most parameters are controlled via the src/config.yaml file. Key sections include:

* **Global Settings:** Project name, paths for results/models, random seed, database path.  
* **Data Settings:** Train/test split sizes, HPO validation split size.  
* **lstm\_roberta:** Configuration specific to the LSTM-RoBERTa model.  
* **han\_roberta:** Configuration specific to the HAN-RoBERTa model.  
* **gnn\_roberta:** Configuration specific to the GNN-RoBERTa model.  
* **atae\_roberta:** Configuration specific to the ATAE-RoBERTa model.

Each model section contains:

* **Model Defaults:** Architecture parameters (base RoBERTa model, hidden dims, layers, dropout, heads, etc.), training defaults (learning rate, batch size, epochs, scheduler settings), freezing RoBERTa layers.  
* **Data Loading:** Default sample size to load from the dataset.  
* **hpo** (Hyperparameter Optimization):  
  * enabled: Set to true to run Optuna when using \--mode optimize.  
  * n\_trials: Number of optimization trials.  
  * hpo\_sample\_size: How much data to use during HPO trials (subset of loaded data).  
  * hpo\_epochs: Epochs per HPO trial.  
  * search\_space: Defines the ranges/choices for hyperparameters Optuna will tune for that specific model.

Modify this file to change default behaviors, paths, or the HPO search space for each model.

## **Usage**

The main scripts are in src/scripts/:

* lstm\_roberta\_main.py  
* han\_roberta\_main.py  
* gnn\_roberta\_main.py  
* atae\_roberta\_main.py

They operate in different modes specified by the \--mode argument.

**Common Modes:**

**1\. Hyperparameter Optimization (--mode optimize)**

Runs Optuna to find the best hyperparameters for the respective model, using settings from its section in config.yaml. After optimization, it trains a final model using the best found parameters on the full training set and evaluates it on the test set.

\# Optimize LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode optimize

\# Optimize HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode optimize

\# Optimize GNN-RoBERTa  
python src/scripts/gnn\_roberta\_main.py \--mode optimize

\# Optimize ATAE-RoBERTa  
python src/scripts/atae\_roberta\_main.py \--mode optimize

\# Optional Overrides (Example for GNN):  
python src/scripts/gnn\_roberta\_main.py \--mode optimize \--n\_trials 15 \--sample\_size 20000

**2\. Training (--mode train)**

Skips Optuna and trains a single model using the default parameters specified in its config.yaml section (or HPO best params if optimization was run previously and defaults weren't manually reset). Trains on the full training set and evaluates on the test set.

\# Train LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode train

\# Train HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode train

\# Train GNN-RoBERTa  
python src/scripts/gnn\_roberta\_main.py \--mode train

\# Train ATAE-RoBERTa  
python src/scripts/atae\_roberta\_main.py \--mode train

\# Optional Overrides (Example for ATAE):  
python src/scripts/atae\_roberta\_main.py \--mode train \--final\_epochs 5 \--sample\_size 50000

**3\. Testing (--mode test)**

Loads the last saved final model for the respective type (e.g., lstm\_roberta\_model\_final.pt, han\_roberta\_model\_final.pt, etc.) and evaluates it on the test set.

\# Test LSTM-RoBERTa  
python src/scripts/lstm\_roberta\_main.py \--mode test

\# Test HAN-RoBERTa  
python src/scripts/han\_roberta\_main.py \--mode test

\# Test GNN-RoBERTa  
python src/scripts/gnn\_roberta\_main.py \--mode test

\# Test ATAE-RoBERTa  
python src/scripts/atae\_roberta\_main.py \--mode test

\# Optional Override (Example for LSTM): Test a specific model checkpoint  
python src/scripts/lstm\_roberta\_main.py \--mode test \--load\_model\_path src/models/some\_other\_lstm\_model.pt

**4\. Fine-tuning (--mode finetune)** *(Currently implemented for HAN-RoBERTa)*

Loads a previously saved model checkpoint and continues training on a new dataset.

\# Fine-tune a saved HAN model on a new dataset  
python src/scripts/han\_roberta\_main.py \--mode finetune \\  
    \--load\_model\_path src/models/han\_roberta\_model\_final.pt \\  
    \--finetune\_data\_path /path/to/new\_larger\_dataset.db \\  
    \--finetune\_epochs 4 \\  
    \--save\_finetuned\_model\_path src/models/han\_roberta\_finetuned\_large.pt

*(Adaptation might be needed to enable fine-tuning for LSTM, GNN, ATAE models)*

**Running All Models Sequentially**

Use the provided wrapper scripts (run\_all\_models.sh for Linux/macOS) located in the project root. Pass any arguments intended for the underlying Python scripts directly to the wrapper script.

\# Example: Train all models for 5 epochs on Linux/macOS  
chmod \+x run\_all\_models.sh \# Make executable first  
./run\_all\_models.sh \--mode train \--final\_epochs 5

\# Example: Optimize all models on Windows  
\# run\_all\_models.bat \--mode optimize \--n\_trials 25

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

## **Output**

* **Models:** Trained models are saved in the directory specified by models\_dir in config.yaml (default: src/models/), with distinct names like lstm\_roberta\_model\_final.pt, han\_roberta\_model\_final.pt, gnn\_roberta\_model\_final.pt, atae\_roberta\_model\_final.pt. Fine-tuned models are saved to specified paths.  
* **Results:** Evaluation metrics and parameters are logged to src/model\_results.json. Entries are identified by model name (e.g., LSTMRoBERTa\_Train, HANRoBERTa\_Optimize, GNNRoBERTa\_Test, ATAERoBERTa\_Train). Confusion matrix plots are saved to the results\_dir (default: src/results/) with model-specific names (e.g., lstm\_roberta\_test\_confusion\_matrix.png).  
* **Optuna Databases:** Optimization studies are stored in separate SQLite databases in the results\_dir (e.g., optuna\_lstm\_roberta.db, optuna\_han\_roberta.db, optuna\_gnn\_roberta.db, optuna\_atae\_roberta.db).  
* **Logs:** Console output provides detailed logging. Redirect to a file if needed: python ... \> run.log 2\>&1