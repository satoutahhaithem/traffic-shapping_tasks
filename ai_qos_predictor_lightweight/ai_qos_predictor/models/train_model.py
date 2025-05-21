#!/usr/bin/env python3
"""
Train Model for QoS Prediction

This script trains a model to predict future network QoS metrics
based on sequences of past network measurements.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import DataLoader, TensorDataset

# Import model implementations
from lstm_model import QoSPredictor
from transformer_model import QoSTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Constants
FEATURES_DIR = PROJECT_DIR / "data" / "features"
MODELS_DIR = PROJECT_DIR / "models" / "saved"
RESULTS_DIR = PROJECT_DIR / "models" / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(dataset_name):
    """
    Load prepared sequences and metadata for a dataset
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary with data and metadata
    """
    # Load sequences
    sequences_file = FEATURES_DIR / f"{dataset_name}_sequences.npz"
    if not sequences_file.exists():
        logger.error(f"Sequences file not found: {sequences_file}")
        return None
    
    sequences = np.load(sequences_file)
    
    # Load metadata
    metadata_file = FEATURES_DIR / f"{dataset_name}_metadata.pkl"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return None
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'X_train': sequences['X_train'],
        'y_train': sequences['y_train'],
        'X_val': sequences['X_val'],
        'y_val': sequences['y_val'],
        'X_test': sequences['X_test'],
        'y_test': sequences['y_test'],
        'features': metadata['features'],
        'targets': metadata['targets'],
        'sequence_length': metadata['sequence_length'],
        'prediction_horizon': metadata['prediction_horizon']
    }

def train_model(model_type, data, config=None):
    """
    Train the model with early stopping
    
    Args:
        model_type: Type of model to train ('lstm' or 'transformer')
        data: Dictionary with training and validation data
        config: Dictionary with model configuration
        
    Returns:
        Trained model and training history
    """
    # Create model based on type
    if model_type == 'lstm':
        model = QoSPredictor(config)
    elif model_type == 'transformer':
        model = QoSTransformer(config)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None
    
    # Train the model
    history = model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        {
            'features': data['features'],
            'targets': data['targets'],
            'sequence_length': data['sequence_length'],
            'prediction_horizon': data['prediction_horizon']
        }
    )
    
    return model, history

def evaluate_model(model, data):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        data: Dictionary with test data and metadata
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    metrics = model.evaluate(data['X_test'], data['y_test'])
    
    # Calculate R² score for each target
    y_pred = model.predict(data['X_test'])
    
    for i, target in enumerate(data['targets']):
        y_true = data['y_test'][:, i]
        pred = y_pred[:, i]
        
        r2 = r2_score(y_true, pred)
        metrics[f"{target}_r2"] = r2
        
        logger.info(f"{target} R² score: {r2:.4f}")
    
    return metrics

def plot_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()

def plot_predictions(model, data, num_samples=5, save_path=None):
    """
    Plot predictions vs actual values for a few test samples
    
    Args:
        model: Trained model
        data: Dictionary with test data and metadata
        num_samples: Number of samples to plot
        save_path: Path to save the plot (optional)
    """
    # Get random samples from test set
    indices = np.random.choice(len(data['X_test']), num_samples, replace=False)
    X_samples = data['X_test'][indices]
    y_true = data['y_test'][indices]
    
    # Make predictions
    y_pred = model.predict(X_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, len(data['targets']), figsize=(15, 3*num_samples))
    
    # Plot each sample and target
    for i in range(num_samples):
        for j, target in enumerate(data['targets']):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            
            # Plot input sequence
            input_seq = X_samples[i, :, data['features'].index(target) if target in data['features'] else 0]
            ax.plot(range(len(input_seq)), input_seq, 'b-', label='Input Sequence')
            
            # Plot actual and predicted values
            ax.plot(len(input_seq) + data['prediction_horizon'] - 1, y_true[i, j], 'go', label='Actual')
            ax.plot(len(input_seq) + data['prediction_horizon'] - 1, y_pred[i, j], 'ro', label='Predicted')
            
            # Add vertical line to separate input from prediction
            ax.axvline(x=len(input_seq) - 1, color='k', linestyle='--')
            
            ax.set_title(f"Sample {i+1}, Target: {target}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Normalized Value")
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Predictions plot saved to {save_path}")
    
    plt.close()

def save_results(metrics, history, model_summary, save_path):
    """
    Save evaluation results to a text file
    
    Args:
        metrics: Dictionary with evaluation metrics
        history: Training history
        model_summary: Model summary string
        save_path: Path to save the results
    """
    with open(save_path, 'w') as f:
        f.write("Model Summary:\n")
        f.write(model_summary)
        f.write("\n\n")
        
        f.write("Training History:\n")
        f.write(f"Final training loss: {history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final training MAE: {history['mae'][-1]:.4f}\n")
        f.write(f"Final validation MAE: {history['val_mae'][-1]:.4f}\n")
        f.write("\n")
        
        f.write("Evaluation Metrics:\n")
        # Check if metrics is a nested dictionary or a flat dictionary
        if any(isinstance(v, dict) for v in metrics.values()):
            # Nested dictionary structure
            for target, target_metrics in metrics.items():
                f.write(f"Target: {target}\n")
                for metric_name, value in target_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
        else:
            # Flat dictionary structure
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")
    
    logger.info(f"Results saved to {save_path}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model for QoS prediction')
    
    parser.add_argument('--model-type', choices=['lstm', 'transformer'], default='lstm',
                       help='Type of model to train')
    parser.add_argument('--dataset', default='5g_kpi',
                       help='Dataset to use for training')
    parser.add_argument('--config', type=str,
                       help='Path to model configuration JSON file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    return parser.parse_args()

def main():
    """Main function to train and evaluate the model"""
    # Parse command-line arguments
    args = parse_args()
    
    logger.info(f"Starting {args.model_type} model training on {args.dataset} dataset")
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Add command-line arguments to config
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['patience'] = args.patience
    
    # Load data
    data = load_data(args.dataset)
    
    if data is None:
        logger.error(f"Failed to load data for {args.dataset}")
        return
    
    logger.info(f"Loaded data for {args.dataset}")
    logger.info(f"Features: {data['features']}")
    logger.info(f"Targets: {data['targets']}")
    logger.info(f"Sequence length: {data['sequence_length']}")
    logger.info(f"Prediction horizon: {data['prediction_horizon']}")
    logger.info(f"Training samples: {len(data['X_train'])}")
    logger.info(f"Validation samples: {len(data['X_val'])}")
    logger.info(f"Test samples: {len(data['X_test'])}")
    
    # Train model
    logger.info("Training model...")
    model, history = train_model(args.model_type, data, config)
    
    if model is None:
        logger.error("Model training failed")
        return
    
    logger.info("Model training completed")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, data)
    
    # Get model summary as string
    model_summary = str(model.model)
    
    # Plot and save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_history(
        history,
        save_path=RESULTS_DIR / f"{args.dataset}_{args.model_type}_history_{timestamp}.png"
    )
    
    plot_predictions(
        model,
        data,
        save_path=RESULTS_DIR / f"{args.dataset}_{args.model_type}_predictions_{timestamp}.png"
    )
    
    save_results(
        metrics,
        history,
        model_summary,
        RESULTS_DIR / f"{args.dataset}_{args.model_type}_results_{timestamp}.txt"
    )
    
    # Save model
    model_name = f"{args.dataset}_{args.model_type}_{timestamp}"
    model.save(model_name)
    
    logger.info(f"Model saved as {model_name}")
    logger.info("Training and evaluation completed successfully")

if __name__ == "__main__":
    main()