#!/usr/bin/env python3
"""
Train LSTM Model for QoS Prediction

This script trains an LSTM model to predict future network QoS metrics
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

def build_lstm_model(input_shape, output_shape, lstm_units=64, dropout_rate=0.2):
    """
    Build an LSTM model for time-series prediction
    
    Args:
        input_shape: Shape of input sequences (sequence_length, n_features)
        output_shape: Number of target variables to predict
        lstm_units: Number of LSTM units in each layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(output_shape)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_transformer_model(input_shape, output_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, dropout_rate=0.2):
    """
    Build a Transformer model for time-series prediction
    
    Args:
        input_shape: Shape of input sequences (sequence_length, n_features)
        output_shape: Number of target variables to predict
        head_size: Size of attention heads
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Define Transformer block
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(inputs.shape[-1]),
        ])
        ffn_output = ffn(attention_output)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Add transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Final dense layer
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, data, epochs=100, batch_size=64, patience=10):
    """
    Train the model with early stopping
    
    Args:
        model: Compiled Keras model
        data: Dictionary with training and validation data
        epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        patience: Number of epochs with no improvement before stopping
        
    Returns:
        Training history
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        str(MODELS_DIR / 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        data['X_train'],
        data['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, data):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained Keras model
        data: Dictionary with test data and metadata
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(data['X_test'])
    
    # Calculate metrics for each target
    metrics = {}
    for i, target in enumerate(data['targets']):
        y_true = data['y_test'][:, i]
        pred = y_pred[:, i]
        
        mse = mean_squared_error(y_true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)
        
        metrics[target] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    return metrics

def plot_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
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
        model: Trained Keras model
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
        f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Final training MAE: {history.history['mae'][-1]:.4f}\n")
        f.write(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}\n")
        f.write("\n")
        
        f.write("Evaluation Metrics:\n")
        for target, target_metrics in metrics.items():
            f.write(f"Target: {target}\n")
            for metric_name, value in target_metrics.items():
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
    
    # Build model
    input_shape = (data['sequence_length'], len(data['features']))
    output_shape = len(data['targets'])
    
    if args.model_type == 'lstm':
        # Get LSTM hyperparameters from config or use defaults
        lstm_units = config.get('lstm_units', 64)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        model = build_lstm_model(
            input_shape, 
            output_shape, 
            lstm_units=lstm_units, 
            dropout_rate=dropout_rate
        )
    elif args.model_type == 'transformer':
        # Get Transformer hyperparameters from config or use defaults
        head_size = config.get('head_size', 256)
        num_heads = config.get('num_heads', 4)
        ff_dim = config.get('ff_dim', 4)
        num_transformer_blocks = config.get('num_transformer_blocks', 4)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        model = build_transformer_model(
            input_shape, 
            output_shape, 
            head_size=head_size, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            num_transformer_blocks=num_transformer_blocks, 
            dropout_rate=dropout_rate
        )
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
    
    # Get model summary as string
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    model_summary = "\n".join(model_summary_lines)
    
    logger.info("Model built")
    logger.info(model_summary)
    
    # Train model
    logger.info("Training model...")
    history = train_model(
        model, 
        data, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        patience=args.patience
    )
    logger.info("Model training completed")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, data)
    
    # Log metrics
    for target, target_metrics in metrics.items():
        logger.info(f"Metrics for {target}:")
        for metric_name, value in target_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Plot and save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
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
        save_path=RESULTS_DIR / f"{args.dataset}_{args.model_type}_results_{timestamp}.txt"
    )
    
    # Save model
    model_path = MODELS_DIR / f"{args.dataset}_{args.model_type}_model_{timestamp}.h5"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(MODELS_DIR / f"{args.dataset}_{args.model_type}_model_{timestamp}.json", "w") as json_file:
        json_file.write(model_json)
    
    logger.info("Model training and evaluation completed")

if __name__ == "__main__":
    main()