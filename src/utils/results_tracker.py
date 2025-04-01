"""
Utility for tracking and comparing model results.
"""

import os
import json
import datetime
import csv
from typing import Dict, List, Any, Union, Optional

# File for storing model results
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_results.json")

def _serialize_for_json(obj: Any) -> Any:
    """
    Convert numpy and tensorflow types to standard Python types for JSON serialization.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON serializable version of the object
    """
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (dict, list)):
        return obj
    elif hasattr(obj, 'item'):  # Handle tensor-like objects with .item() method
        return obj.item()
    else:
        return str(obj)

def _convert_to_serializable(data: Any) -> Any:
    """
    Recursively convert all values in a nested structure to JSON serializable types.
    
    Args:
        data: Data structure to convert
    
    Returns:
        JSON serializable version of the data structure
    """
    if isinstance(data, dict):
        return {k: _convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_to_serializable(item) for item in data]
    else:
        return _serialize_for_json(data)

def save_model_results(
    model_name: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
    example_predictions: Dict[str, Any]
) -> None:
    """
    Save model results to a JSON file.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of performance metrics (accuracy, precision, recall, f1_score)
        parameters: Dictionary of model parameters and training settings
        example_predictions: Dictionary of example texts and their predictions
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create result entry
    result = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "parameters": parameters,
        "example_predictions": example_predictions
    }
    
    # Convert all values to JSON serializable types
    result = _convert_to_serializable(result)
    
    # Load existing results if available
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    
    # Add new result
    results.append(result)
    
    # Sort results by F1 score (descending)
    results.sort(key=lambda x: float(x["metrics"].get("f1_score", 0)), reverse=True)
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results for {model_name} saved to {RESULTS_FILE}")

def get_best_model() -> Dict[str, Any]:
    """
    Get the best performing model according to F1 score.
    
    Returns:
        Dictionary with best model information
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        
        if results:
            # Already sorted by F1 score
            return results[0]
    
    return None

def generate_results_report() -> str:
    """
    Generate a formatted report of all model results.
    
    Returns:
        String containing the formatted results report
    """
    # Load results
    if not os.path.exists(RESULTS_FILE):
        return "No model results found."
    
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    if not results:
        return "No model results found."
    
    # Sort results by accuracy (higher is better)
    results.sort(key=lambda x: x["metrics"].get("accuracy", 0), reverse=True)
    
    # Create report header
    report = "# Model Results Comparison\n\n"
    report += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Create comparison table
    report += "## Model Performance Comparison\n\n"
    report += "| Model | Timestamp | Accuracy | Precision | Recall | F1 Score |\n"
    report += "|-------|-----------|--------|----------|----------|-----------|\n"
    
    for model in results:
        metrics = model["metrics"]
        report += f"| {model['model_name']} | "
        report += f"{model['timestamp']} | "
        
        # Handle each metric, formatting as float if possible
        accuracy = metrics.get('accuracy', 'N/A')
        precision = metrics.get('precision', 'N/A')
        recall = metrics.get('recall', 'N/A')
        f1_score = metrics.get('f1_score', 'N/A')
        
        if isinstance(accuracy, (int, float)):
            report += f"{accuracy:.4f} | "
        else:
            report += f"{accuracy} | "
            
        if isinstance(precision, (int, float)):
            report += f"{precision:.4f} | "
        else:
            report += f"{precision} | "
            
        if isinstance(recall, (int, float)):
            report += f"{recall:.4f} | "
        else:
            report += f"{recall} | "
            
        if isinstance(f1_score, (int, float)):
            report += f"{f1_score:.4f} |\n"
        else:
            report += f"{f1_score} |\n"
    
    report += "\n"
    
    # Add parameter comparison for the best model
    best_model = results[0]
    report += f"\n## Best Model: {best_model['model_name']}\n\n"
    
    report += "### Parameters:\n\n"
    for param, value in best_model['parameters'].items():
        report += f"- **{param}**: {value}\n"
    
    report += "\n### Example Predictions:\n\n"
    for text, prediction in best_model['example_predictions'].items():
        # Truncate long texts
        text_display = text[:100] + "..." if len(text) > 100 else text
        report += f"- \"{text_display}\"\n"
        report += f"  - **Prediction**: {prediction['label']}\n"
        
        confidence = prediction.get('confidence', 'N/A')
        if isinstance(confidence, (int, float)):
            report += f"  - **Confidence**: {confidence:.4f}\n"
        else:
            report += f"  - **Confidence**: {confidence}\n"
    
    return report

def print_model_comparison() -> None:
    """Print a comparison of all trained models to the console."""
    print(generate_results_report())

def save_report_to_file(output_file: str = None) -> None:
    """
    Save the results report to a file.
    
    Args:
        output_file: Path to the output file (default: model_results.csv in src directory)
    """
    if output_file is None:
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_results.csv')
    
    # Load results
    if not os.path.exists(RESULTS_FILE):
        print("No model results found.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No model results found.")
        return
    
    # Sort results by accuracy (higher is better)
    results.sort(key=lambda x: x["metrics"].get("accuracy", 0), reverse=True)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['model_name', 'timestamp', 'accuracy', 'precision', 'recall', 'f1_score']
        
        # Add parameter fields from all models
        param_fields = set()
        for model in results:
            for param in model.get('parameters', {}).keys():
                param_fields.add(f"param_{param}")
        
        fieldnames.extend(sorted(param_fields))
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model in results:
            row = {
                'model_name': model['model_name'],
                'timestamp': model['timestamp'],
                'accuracy': _format_metric(model['metrics'].get('accuracy')),
                'precision': _format_metric(model['metrics'].get('precision')),
                'recall': _format_metric(model['metrics'].get('recall')),
                'f1_score': _format_metric(model['metrics'].get('f1_score'))
            }
            
            # Add parameters
            for param, value in model.get('parameters', {}).items():
                row[f"param_{param}"] = value
                
            writer.writerow(row)
    
    print(f"Results report saved to {output_file}")
    
    # Also save a simple text summary of the best model
    best_model = results[0]
    summary_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write(f"Best Model: {best_model['model_name']}\n")
        f.write(f"Accuracy: {_format_metric(best_model['metrics'].get('accuracy'))}\n")
        f.write(f"F1 Score: {_format_metric(best_model['metrics'].get('f1_score'))}\n\n")
        
        f.write("Example Predictions:\n")
        for text, prediction in best_model['example_predictions'].items():
            # Truncate long texts
            text_display = text[:100] + "..." if len(text) > 100 else text
            f.write(f"- \"{text_display}\"\n")
            f.write(f"  Prediction: {prediction['label']}")
            
            confidence = prediction.get('confidence', 'N/A')
            if isinstance(confidence, (int, float)):
                f.write(f" (Confidence: {confidence:.4f})\n")
            else:
                f.write(f" (Confidence: {confidence})\n")
            
    print(f"Best model summary saved to {summary_file}")

def _format_metric(value):
    """Format a metric value for CSV output."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return value 