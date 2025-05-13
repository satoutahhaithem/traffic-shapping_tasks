#!/usr/bin/env python3
"""
Model Comparison for QoS Prediction

This script compares the performance of different models (LSTM and Transformer)
on the QoS prediction task, evaluating them on test data and generating
visualizations to highlight their strengths and weaknesses.
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix

# Add parent directory to path to import models
import sys
sys.path.append('..')

from models.lstm_model import QoSPredictor
from models.transformer_model import QoSTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FEATURES_DIR = Path("../data/features")
MODELS_DIR = Path("../models")
RESULTS_DIR = Path("../evaluation/results")
COMPARISON_DIR = Path("../evaluation/comparison")

def load_models(model_paths):
    """
    Load trained models for comparison
    
    Args:
        model_paths: Dictionary mapping model names to paths
        
    Returns:
        Dictionary of loaded models
    """
    loaded_models = {}
    
    for name, path_info in model_paths.items():
        model_path = path_info['model_path']
        config_path = path_info.get('config_path')
        model_type = path_info.get('model_type', 'lstm')
        
        logger.info(f"Loading {model_type} model: {name} from {model_path}")
        
        try:
            if model_type == 'lstm':
                model = QoSPredictor()
                model.load(model_path, config_path)
            elif model_type == 'transformer':
                model = QoSTransformer()
                model.load(model_path, config_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                continue
                
            loaded_models[name] = model
            logger.info(f"Successfully loaded model: {name}")
            
        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
    
    return loaded_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models on the same test data
    
    Args:
        models: Dictionary of loaded models
        X_test: Test input sequences
        y_test: Test target values
        
    Returns:
        Dictionary with evaluation results for each model
    """
    results = {}
    predictions = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        metrics = {}
        
        if model.config['model_type'] == 'regression':
            # For regression, calculate MAE and RMSE for each target
            for i, target in enumerate(model.metadata['targets']):
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                
                metrics[f'{target}_mae'] = mae
                metrics[f'{target}_rmse'] = rmse
                
                logger.info(f"{name} - {target}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        else:
            # For classification, calculate accuracy and F1 score
            # Assuming binary classification with threshold 0.5
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            for i, target in enumerate(model.metadata['targets']):
                accuracy = np.mean(y_test[:, i] == y_pred_binary[:, i])
                f1 = f1_score(y_test[:, i], y_pred_binary[:, i], average='binary')
                
                metrics[f'{target}_accuracy'] = accuracy
                metrics[f'{target}_f1'] = f1
                
                logger.info(f"{name} - {target}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        # Calculate lead time for stall prediction if applicable
        if 'stall_event' in model.metadata['targets']:
            lead_time = calculate_lead_time(model, X_test, y_test, y_pred)
            metrics['stall_lead_time'] = lead_time
            logger.info(f"{name} - Average stall prediction lead time: {lead_time:.2f} seconds")
        
        results[name] = metrics
    
    return results, predictions

def calculate_lead_time(model, X_test, y_test, y_pred, threshold=0.5):
    """
    Calculate the average lead time for stall prediction
    
    Args:
        model: The model being evaluated
        X_test: Test input sequences
        y_test: Test target values
        y_pred: Predicted target values
        threshold: Threshold for binary classification
        
    Returns:
        Average lead time in seconds
    """
    # Find the index of stall_event in targets
    stall_idx = model.metadata['targets'].index('stall_event')
    
    # Convert predictions to binary
    y_pred_binary = (y_pred[:, stall_idx] > threshold).astype(int)
    
    # Find actual stall events
    actual_stalls = np.where(y_test[:, stall_idx] == 1)[0]
    
    if len(actual_stalls) == 0:
        logger.warning("No actual stall events found in test data")
        return 0
    
    # Calculate lead time for each stall event
    lead_times = []
    
    for stall_idx in actual_stalls:
        # Look back to find when the model first predicted this stall
        for i in range(max(0, stall_idx - 30), stall_idx):  # Look up to 30 seconds back
            if y_pred_binary[i] == 1:
                # Found a prediction, calculate lead time
                lead_time = stall_idx - i
                lead_times.append(lead_time)
                break
    
    if not lead_times:
        logger.warning("No successful stall predictions found")
        return 0
    
    # Calculate average lead time
    avg_lead_time = np.mean(lead_times)
    
    return avg_lead_time

def compare_metrics(results):
    """
    Compare metrics across different models
    
    Args:
        results: Dictionary with evaluation results for each model
        
    Returns:
        DataFrame with comparative metrics
    """
    # Collect all metrics from all models
    all_metrics = set()
    for model_metrics in results.values():
        all_metrics.update(model_metrics.keys())
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(index=sorted(all_metrics), columns=results.keys())
    
    # Fill in the values
    for model_name, model_metrics in results.items():
        for metric, value in model_metrics.items():
            comparison_df.loc[metric, model_name] = value
    
    return comparison_df

def plot_metric_comparison(comparison_df, metric_pattern):
    """
    Plot comparison of a specific metric across models
    
    Args:
        comparison_df: DataFrame with comparative metrics
        metric_pattern: String pattern to filter metrics
        
    Returns:
        Figure with the comparison plot
    """
    # Filter metrics matching the pattern
    metrics = [m for m in comparison_df.index if metric_pattern in m]
    
    if not metrics:
        logger.warning(f"No metrics found matching pattern: {metric_pattern}")
        return None
    
    # Extract model names
    models = comparison_df.columns
    
    # Create a new DataFrame for plotting
    plot_df = comparison_df.loc[metrics, :].copy()
    
    # Rename index for better labels
    plot_df.index = [m.replace(f'{metric_pattern}_', '') for m in metrics]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind='bar', ax=ax)
    
    plt.title(f'Comparison of {metric_pattern.upper()} across models')
    plt.ylabel(metric_pattern.upper())
    plt.xlabel('Target')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    
    return fig

def plot_predictions(predictions, y_test, metadata, target_idx=0, n_samples=100):
    """
    Plot predictions from different models against actual values
    
    Args:
        predictions: Dictionary with predictions from each model
        y_test: Test target values
        metadata: Dictionary with metadata about the features and targets
        target_idx: Index of the target to plot
        n_samples: Number of samples to plot
        
    Returns:
        Figure with the prediction plot
    """
    target_name = metadata['targets'][target_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.plot(y_test[:n_samples, target_idx], 'k-', label='Actual', linewidth=2)
    
    # Plot predictions from each model
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    for i, (name, y_pred) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        ax.plot(y_pred[:n_samples, target_idx], f'{color}-', label=name, alpha=0.7)
    
    plt.title(f'Predictions for {target_name}')
    plt.xlabel('Time Step')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_error_distribution(predictions, y_test, metadata, target_idx=0):
    """
    Plot error distribution for different models
    
    Args:
        predictions: Dictionary with predictions from each model
        y_test: Test target values
        metadata: Dictionary with metadata about the features and targets
        target_idx: Index of the target to plot
        
    Returns:
        Figure with the error distribution plot
    """
    target_name = metadata['targets'][target_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate errors for each model
    for name, y_pred in predictions.items():
        errors = y_test[:, target_idx] - y_pred[:, target_idx]
        
        # Plot error distribution
        sns.kdeplot(errors, label=name, ax=ax)
    
    plt.title(f'Error Distribution for {target_name}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_confusion_matrices(predictions, y_test, metadata, target_idx=0, threshold=0.5):
    """
    Plot confusion matrices for classification models
    
    Args:
        predictions: Dictionary with predictions from each model
        y_test: Test target values
        metadata: Dictionary with metadata about the features and targets
        target_idx: Index of the target to plot
        threshold: Threshold for binary classification
        
    Returns:
        Figure with confusion matrices
    """
    target_name = metadata['targets'][target_idx]
    
    # Count number of models
    n_models = len(predictions)
    
    # Create subplot grid
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # Plot confusion matrix for each model
    for i, (name, y_pred) in enumerate(predictions.items()):
        # Convert predictions to binary
        y_pred_binary = (y_pred[:, target_idx] > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test[:, target_idx], y_pred_binary)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} - {target_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    
    return fig

def generate_comparison_report(comparison_df, predictions, y_test, metadata):
    """
    Generate a comprehensive comparison report
    
    Args:
        comparison_df: DataFrame with comparative metrics
        predictions: Dictionary with predictions from each model
        y_test: Test target values
        metadata: Dictionary with metadata about the features and targets
    """
    # Create comparison directory if it doesn't exist
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison DataFrame
    comparison_path = COMPARISON_DIR / f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"Saved metric comparison to {comparison_path}")
    
    # Generate and save metric comparison plots
    metric_patterns = ['mae', 'rmse', 'accuracy', 'f1']
    for pattern in metric_patterns:
        fig = plot_metric_comparison(comparison_df, pattern)
        if fig:
            plot_path = COMPARISON_DIR / f"{pattern}_comparison_{timestamp}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved {pattern} comparison plot to {plot_path}")
    
    # Generate and save prediction plots for each target
    for i, target in enumerate(metadata['targets']):
        # Prediction plot
        fig = plot_predictions(predictions, y_test, metadata, target_idx=i)
        plot_path = COMPARISON_DIR / f"{target}_predictions_{timestamp}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved {target} prediction plot to {plot_path}")
        
        # Error distribution plot
        fig = plot_error_distribution(predictions, y_test, metadata, target_idx=i)
        plot_path = COMPARISON_DIR / f"{target}_error_distribution_{timestamp}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved {target} error distribution plot to {plot_path}")
        
        # Confusion matrix for classification targets
        if target in ['stall_event']:
            fig = plot_confusion_matrices(predictions, y_test, metadata, target_idx=i)
            plot_path = COMPARISON_DIR / f"{target}_confusion_matrices_{timestamp}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved {target} confusion matrices to {plot_path}")
    
    # Generate HTML report
    html_report = generate_html_report(comparison_df, timestamp, metadata['targets'])
    report_path = COMPARISON_DIR / f"model_comparison_report_{timestamp}.html"
    with open(report_path, 'w') as f:
        f.write(html_report)
    logger.info(f"Saved HTML report to {report_path}")

def generate_html_report(comparison_df, timestamp, targets):
    """
    Generate an HTML report with comparison results
    
    Args:
        comparison_df: DataFrame with comparative metrics
        timestamp: Timestamp string for file references
        targets: List of target names
        
    Returns:
        HTML string with the report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QoS Prediction Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-section {{ margin-bottom: 30px; }}
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
            .plot {{ max-width: 100%; height: auto; margin-bottom: 10px; }}
            .best {{ font-weight: bold; color: green; }}
        </style>
    </head>
    <body>
        <h1>QoS Prediction Model Comparison Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Model Performance Metrics</h2>
    """
    
    # Add metric tables by category
    metric_categories = {
        'Regression Metrics': ['mae', 'rmse'],
        'Classification Metrics': ['accuracy', 'f1'],
        'Lead Time Metrics': ['lead_time']
    }
    
    for category, patterns in metric_categories.items():
        # Check if we have any metrics in this category
        has_metrics = any(any(pattern in metric for pattern in patterns) for metric in comparison_df.index)
        
        if has_metrics:
            html += f"""
            <div class="metric-section">
                <h3>{category}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
            """
            
            # Add model names as column headers
            for model in comparison_df.columns:
                html += f"<th>{model}</th>"
            
            html += "</tr>"
            
            # Add rows for each metric
            for metric in comparison_df.index:
                if any(pattern in metric for pattern in patterns):
                    html += f"<tr><td>{metric}</td>"
                    
                    # Find the best value for this metric
                    values = comparison_df.loc[metric].dropna()
                    if len(values) > 0:
                        if 'mae' in metric or 'rmse' in metric:
                            best_model = values.idxmin()  # Lower is better
                            best_value = values.min()
                        else:
                            best_model = values.idxmax()  # Higher is better
                            best_value = values.max()
                    
                        # Add values for each model
                        for model in comparison_df.columns:
                            value = comparison_df.loc[metric, model]
                            if pd.notna(value):
                                if model == best_model:
                                    html += f"<td class='best'>{value:.4f}</td>"
                                else:
                                    html += f"<td>{value:.4f}</td>"
                            else:
                                html += "<td>-</td>"
                    
                    html += "</tr>"
            
            html += """
                </table>
            </div>
            """
    
    # Add plots section
    html += """
        <h2>Visualization</h2>
        
        <h3>Metric Comparisons</h3>
        <div class="plot-container">
    """
    
    # Add metric comparison plots
    for pattern in ['mae', 'rmse', 'accuracy', 'f1']:
        plot_path = f"{pattern}_comparison_{timestamp}.png"
        if (COMPARISON_DIR / plot_path).exists():
            html += f"""
            <div>
                <img class="plot" src="{plot_path}" alt="{pattern.upper()} Comparison">
                <p>{pattern.upper()} comparison across models</p>
            </div>
            """
    
    html += """
        </div>
        
        <h3>Predictions</h3>
    """
    
    # Add prediction plots for each target
    for target in targets:
        html += f"""
        <h4>{target}</h4>
        <div class="plot-container">
            <div>
                <img class="plot" src="{target}_predictions_{timestamp}.png" alt="{target} Predictions">
                <p>Model predictions vs actual values</p>
            </div>
            
            <div>
                <img class="plot" src="{target}_error_distribution_{timestamp}.png" alt="{target} Error Distribution">
                <p>Error distribution across models</p>
            </div>
        """
        
        # Add confusion matrix if available
        if target in ['stall_event'] and (COMPARISON_DIR / f"{target}_confusion_matrices_{timestamp}.png").exists():
            html += f"""
            <div>
                <img class="plot" src="{target}_confusion_matrices_{timestamp}.png" alt="{target} Confusion Matrices">
                <p>Confusion matrices for classification</p>
            </div>
            """
        
        html += """
        </div>
        """
    
    # Add conclusion section
    html += """
        <h2>Conclusion</h2>
        <p>
            This report compares different models for QoS prediction. Based on the metrics and visualizations above,
            you can determine which model performs best for your specific use case. Consider the following factors:
        </p>
        <ul>
            <li><strong>Regression accuracy:</strong> Lower MAE and RMSE values indicate better prediction accuracy for continuous targets.</li>
            <li><strong>Classification performance:</strong> Higher accuracy and F1 scores indicate better classification for binary targets.</li>
            <li><strong>Lead time:</strong> Longer lead times for stall prediction allow more time for adaptive strategies.</li>
            <li><strong>Error distribution:</strong> Models with narrower error distributions centered around zero are more consistent.</li>
        </ul>
        <p>
            For deployment in a real-world system, consider not only the raw performance metrics but also
            the computational requirements and inference speed of each model.
        </p>
    </body>
    </html>
    """
    
    return html

def main():
    """Main function to compare different QoS prediction models"""
    logger.info("Starting QoS prediction model comparison")
    
    # Load test data
    dataset_name = '5g_kpi'  # Use 5G KPI dataset as primary
    data_file = FEATURES_DIR / f"{dataset_name}_sequences.npz"
    metadata_file = FEATURES_DIR / f"{dataset_name}_metadata.pkl"
    
    if not data_file.exists() or not metadata_file.exists():
        logger.error(f"Test data not found. Run feature_extraction.py first.")
        return
    
    # Load data
    data = np.load(data_file)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded test data with {X_test.shape[0]} samples")
    
    # Define models to compare
    # In a real scenario, you would have trained models saved to disk
    # Here we're assuming the models exist at these paths
    model_paths = {
        'LSTM': {
            'model_path': str(MODELS_DIR / 'qos_predictor_regression_latest.h5'),
            'model_type': 'lstm'
        },
        'Transformer': {
            'model_path': str(MODELS_DIR / 'qos_transformer_regression_latest.h5'),
            'model_type': 'transformer'
        }
    }
    
    # Check if models exist
    missing_models = []
    for name, path_info in model_paths.items():
        if not Path(path_info['model_path']).exists():
            missing_models.append(name)
    
    if missing_models:
        logger.warning(f"The following models are missing: {missing_models}")
        logger.info("Please train the models first using lstm_model.py and transformer_model.py")
        
        # For demonstration, we'll create a simple comparison with dummy data
        if len(missing_models) == len(model_paths):
            logger.info("Creating dummy comparison for demonstration purposes")
            create_dummy_comparison(metadata)
            return
    
    # Load models
    models = load_models(model_paths)
    
    if not models:
        logger.error("No models could be loaded for comparison")
        return
    
    # Evaluate models
    results, predictions = evaluate_models(models, X_test, y_test)
    
    # Compare metrics
    comparison_df = compare_metrics(results)
    logger.info("\nModel Comparison:")
    logger.info("\n" + str(comparison_df))
    
    # Generate comparison report
    generate_comparison_report(comparison_df, predictions, y_test, metadata)
    
    logger.info("QoS prediction model comparison completed")

def create_dummy_comparison(metadata):
    """
    Create a dummy comparison for demonstration purposes
    
    Args:
        metadata: Dictionary with metadata about the features and targets
    """
    # Create dummy results
    results = {
        'LSTM': {
            'throughput_down_mae': 0.45,
            'throughput_down_rmse': 0.78,
            'rtt_mae': 12.3,
            'rtt_rmse': 18.7,
            'stall_event_accuracy': 0.92,
            'stall_event_f1': 0.85,
            'stall_lead_time': 3.2
        },
        'Transformer': {
            'throughput_down_mae': 0.42,
            'throughput_down_rmse': 0.75,
            'rtt_mae': 11.8,
            'rtt_rmse': 17.9,
            'stall_event_accuracy': 0.94,
            'stall_event_f1': 0.88,
            'stall_lead_time': 3.5
        }
    }
    
    # Create comparison DataFrame
    comparison_df = compare_metrics(results)
    logger.info("\nDummy Model Comparison:")
    logger.info("\n" + str(comparison_df))
    
    # Create dummy predictions
    n_samples = 100
    n_targets = len(metadata['targets'])
    
    # Create synthetic test data
    y_test = np.zeros((n_samples, n_targets))
    for i in range(n_targets):
        if metadata['targets'][i] == 'stall_event':
            # Binary target
            y_test[:, i] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        else:
            # Continuous target
            y_test[:, i] = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 0.2, n_samples)
    
    # Create predictions with slight variations
    predictions = {
        'LSTM': np.zeros((n_samples, n_targets)),
        'Transformer': np.zeros((n_samples, n_targets))
    }
    
    for i in range(n_targets):
        if metadata['targets'][i] == 'stall_event':
            # Binary predictions with different accuracies
            predictions['LSTM'][:, i] = (np.random.rand(n_samples) > 0.15).astype(float)
            predictions['Transformer'][:, i] = (np.random.rand(n_samples) > 0.12).astype(float)
        else:
            # Continuous predictions with different errors
            predictions['LSTM'][:, i] = y_test[:, i] + np.random.normal(0, 0.3, n_samples)
            predictions['Transformer'][:, i] = y_test[:, i] + np.random.normal(0, 0.25, n_samples)
    
    # Generate comparison report
    generate_comparison_report(comparison_df, predictions, y_test, metadata)
    
    logger.info("Dummy QoS prediction model comparison completed")

if __name__ == "__main__":
    main()