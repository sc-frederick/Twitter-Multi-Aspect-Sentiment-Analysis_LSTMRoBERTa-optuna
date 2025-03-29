# Sentiment Analysis Project

A comprehensive sentiment analysis system with three different models: Basic ML model, Enhanced model, and RoBERTa transformer-based model. Includes a results tracking system that compares model performance.

## Features

- Multiple sentiment analysis models:
  - Basic ML model with TF-IDF features
  - Enhanced model with improved preprocessing
  - RoBERTa transformer model for state-of-the-art performance
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

## Usage

### Running the Pipeline

The main pipeline can be run with different modes:

```
python src/run_pipeline.py --mode [MODE] [OPTIONS]
```

Available modes:

| Mode | Description |
|------|-------------|
| `train_basic` | Train the basic model |
| `train_enhanced` | Train the enhanced model |
| `train_roberta` | Train the RoBERTa model |
| `train_all` | Train all models and compare them |
| `test` | Test a specific model |
| `compare` | Compare existing trained models |

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample_size` | Number of samples to use for training | 20000 |
| `--verbose` | Verbosity level (0=silent, 1=progress bar, 2=one line per epoch) | 1 |
| `--test_model` | Model to test when in 'test' mode (basic, enhanced, or roberta) | enhanced |
| `--include_roberta` | Include RoBERTa in the comparison (for 'compare' mode) | False |

### Examples

Train all models with a sample size of 5000:
```
python src/run_pipeline.py --mode train_all --sample_size 5000
```

Train just the RoBERTa model:
```
python src/run_pipeline.py --mode train_roberta --sample_size 1000
```

Test the basic model:
```
python src/run_pipeline.py --mode test --test_model basic
```

Compare all models including RoBERTa:
```
python src/run_pipeline.py --mode compare --include_roberta
```

### Individual Scripts

You can also run the individual scripts directly:

```
python src/scripts/main.py --sample_size 10000
python src/scripts/enhanced_main.py --sample_size 10000
python src/scripts/roberta_main.py --sample_size 5000
python src/scripts/compare_models.py --include_roberta
```

## Results

After running the models, results are stored in:
- `src/model_results.json` (raw data)
- `src/model_results.csv` (CSV format for easy loading into pandas)
- `src/best_model_summary.txt` (text summary of the best model)

The models themselves are stored in the `src/models/` directory. 