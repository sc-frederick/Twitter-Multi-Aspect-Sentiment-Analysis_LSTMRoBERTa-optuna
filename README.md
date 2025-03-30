# Sentiment Analysis Project

A comprehensive sentiment analysis system with five different models: Multi-Layer Perceptron (MLP) Basic, MLP Enhanced, RoBERTa transformer-based, Kernel Approximation, and Randomized PCA models. Includes a results tracking system that compares model performance.

## Features

- Multiple sentiment analysis models:
  - Multi-Layer Perceptron (MLP) Basic: Simple neural network with TF-IDF features
  - Multi-Layer Perceptron (MLP) Enhanced: Improved neural network with advanced preprocessing
  - RoBERTa: Transformer-based model for state-of-the-art performance
  - Kernel Approximation: Approximate RBF kernel features with linear classification
  - Randomized PCA: Dimension reduction with logistic regression classifier
- Performance tracking and comparison
- CSV and text output for model results

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas
- Scikit-learn
- Transformers (Hugging Face)
- Matplotlib

### Installation

```
pip install -r requirements.txt
```
Run the following inside the python shell:
```
>>> import nltk
>>> nltk.download()
```
Then an installation window appears. Go to the 'Models' tab and select 'punkt' & 'punkt_tab' from under the 'Identifier' column. Then click Download and it will install the necessary files.


## Usage

### Running the Pipeline

The main pipeline can be run with different modes:

```
python src/run_pipeline.py --mode [MODE] [OPTIONS]
```

Available modes:

| Mode | Description |
|------|-------------|
| `train_mlp_basic` | Train the MLP Basic model |
| `train_mlp_enhanced` | Train the MLP Enhanced model |
| `train_roberta` | Train the RoBERTa model |
| `train_kernel` | Train the Kernel Approximation model |
| `train_pca` | Train the Randomized PCA model |
| `train_all` | Train all models and compare them |
| `test` | Test a specific model |
| `compare` | Compare existing trained models |

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample_size` | Number of samples to use for training | 20000 |
| `--verbose` | Verbosity level (0=silent, 1=progress bar, 2=one line per epoch) | 1 |
| `--test_model` | Model to test when in 'test' mode (mlp_basic, mlp_enhanced, roberta, kernel, pca) | mlp_enhanced |
| `--include_roberta` | Include RoBERTa in the comparison (for 'compare' mode) | False |
| `--include_all` | Include all models in the comparison (for 'compare' mode) | False |
| `--timeout` | Timeout in seconds for each model script | 1800 |
| `--skip_training` | Skip model training and compare existing results | False |

### Examples

Train all models with a sample size of 5000:
```
python src/run_pipeline.py --mode train_all --sample_size 5000 --include_all
```

Train just the Kernel Approximation model:
```
python src/run_pipeline.py --mode train_kernel --sample_size 10000
```

Train the Randomized PCA model:
```
python src/run_pipeline.py --mode train_pca --sample_size 10000
```

Test the RoBERTa model:
```
python src/run_pipeline.py --mode test --test_model roberta
```

Compare all models without retraining:
```
python src/run_pipeline.py --mode compare --include_all --skip_training
```

### Individual Scripts

You can also run the individual scripts directly:

```
python src/scripts/mlp_basic_main.py --sample_size 10000
python src/scripts/mlp_enhanced_main.py --sample_size 10000
python src/scripts/roberta_main.py --sample_size 5000
python src/scripts/kernel_approximation_main.py --sample_size 10000
python src/scripts/randomized_pca_main.py --sample_size 10000
python src/scripts/compare_models.py --include_all --skip_training
```

## Results

After running the models, results are stored in:
- `src/model_results.json` (raw data)
- `src/model_results.csv` (CSV format for easy loading into pandas)
- `src/best_model_summary.txt` (text summary of the best model)

The models themselves are stored in the `src/models/` directory. 