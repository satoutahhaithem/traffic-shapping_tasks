#!/usr/bin/env python3
"""
Transformer Model for QoS Prediction

This script implements a Transformer-based model for predicting network QoS metrics
based on historical network conditions. Transformers can capture long-range dependencies
in time-series data and may outperform LSTMs for certain QoS prediction tasks.
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Input, GlobalAveragePooling1D, Embedding, Conv1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

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

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head self-attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer models"""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class QoSTransformer:
    """Transformer-based QoS prediction model"""
    
    def __init__(self, config=None):
        """
        Initialize the QoS transformer
        
        Args:
            config: Dictionary with model configuration parameters
        """
        # Default configuration
        self.config = {
            'embed_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_transformer_blocks': 2,
            'mlp_units': [64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'model_type': 'regression'  # 'regression' or 'classification'
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize model
        self.model = None
        self.history = None
        self.metadata = None
    
    def build_model(self, input_shape, output_shape):
        """
        Build the Transformer model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            output_shape: Number of output features to predict
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Initial feature projection to embed_dim
        x = Conv1D(filters=self.config['embed_dim'], kernel_size=1, activation='linear')(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(input_shape[0], self.config['embed_dim'])(x)
        
        # Transformer blocks
        for _ in range(self.config['num_transformer_blocks']):
            x = TransformerBlock(
                self.config['embed_dim'], 
                self.config['num_heads'], 
                self.config['ff_dim'], 
                self.config['dropout_rate']
            )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # MLP head
        for dim in self.config['mlp_units']:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(self.config['dropout_rate'])(x)
        
        # Output layer
        if self.config['model_type'] == 'regression':
            outputs = Dense(output_shape, activation='linear')(x)
        elif self.config['model_type'] == 'classification':
            outputs = Dense(output_shape, activation='sigmoid')(x)
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        
        if self.config['model_type'] == 'regression':
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
        
        self.model = model
        logger.info(f"Built {self.config['model_type']} transformer model with {self.config['num_transformer_blocks']} transformer blocks")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, metadata=None):
        """
        Train the Transformer model
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            metadata: Dictionary with metadata about the features and targets
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_shape = y_train.shape[1]
            self.build_model(input_shape, output_shape)
        
        # Store metadata
        self.metadata = metadata
        
        # Create model directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(MODELS_DIR / 'best_transformer_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting transformer model training")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        
        self.history = history.history
        logger.info("Transformer model training completed")
        
        # Save training history
        self._save_history()
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        if self.config['model_type'] == 'regression':
            # For regression, calculate MAE and RMSE for each target
            for i, target in enumerate(self.metadata['targets']):
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                
                metrics[f'{target}_mae'] = mae
                metrics[f'{target}_rmse'] = rmse
                
                logger.info(f"{target}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        else:
            # For classification, calculate accuracy and F1 score
            # Assuming binary classification with threshold 0.5
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            for i, target in enumerate(self.metadata['targets']):
                accuracy = np.mean(y_test[:, i] == y_pred_binary[:, i])
                f1 = f1_score(y_test[:, i], y_pred_binary[:, i], average='binary')
                
                metrics[f'{target}_accuracy'] = accuracy
                metrics[f'{target}_f1'] = f1
                
                logger.info(f"{target}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        # Calculate lead time for stall prediction
        if 'stall_event' in self.metadata['targets']:
            lead_time = self._calculate_lead_time(X_test, y_test, y_pred)
            metrics['stall_lead_time'] = lead_time
            logger.info(f"Average stall prediction lead time: {lead_time:.2f} seconds")
        
        # Save evaluation results
        self._save_evaluation(metrics, X_test, y_test, y_pred)
        
        return metrics
    
    def _calculate_lead_time(self, X_test, y_test, y_pred, threshold=0.5):
        """
        Calculate the average lead time for stall prediction
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            y_pred: Predicted target values
            threshold: Threshold for binary classification
            
        Returns:
            Average lead time in seconds
        """
        # Find the index of stall_event in targets
        stall_idx = self.metadata['targets'].index('stall_event')
        
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
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted values
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        return self.model.predict(X)
    
    def save(self, model_name=None):
        """
        Save the trained model and its configuration
        
        Args:
            model_name: Name to use for the saved model
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        # Create model directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = self.config['model_type']
            model_name = f"qos_transformer_{model_type}_{timestamp}"
        
        # Save model
        model_path = MODELS_DIR / f"{model_name}.h5"
        self.model.save(str(model_path))
        
        # Save configuration and metadata
        config_path = MODELS_DIR / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'metadata': self.metadata
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Configuration saved to {config_path}")
    
    def load(self, model_path, config_path=None):
        """
        Load a trained model and its configuration
        
        Args:
            model_path: Path to the saved model
            config_path: Path to the saved configuration
        """
        # Load model
        self.model = load_model(model_path, custom_objects={
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        })
        
        # Load configuration and metadata
        if config_path is None:
            # Try to infer config path from model path
            model_path = Path(model_path)
            config_path = model_path.parent / f"{model_path.stem}_config.json"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', self.config)
                self.metadata = data.get('metadata', None)
        
        logger.info(f"Model loaded from {model_path}")
        if self.metadata:
            logger.info(f"Model predicts: {self.metadata['targets']}")
    
    def _save_history(self):
        """Save training history and plots"""
        if self.history is None:
            return
        
        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save history as JSON
        history_path = RESULTS_DIR / f"transformer_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Transformer Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        for metric in self.history:
            if metric not in ['loss', 'val_loss']:
                plt.plot(self.history[metric], label=metric)
        plt.title('Transformer Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_DIR / f"transformer_history_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training history saved to {history_path}")
        logger.info(f"Training plot saved to {plot_path}")
    
    def _save_evaluation(self, metrics, X_test, y_test, y_pred):
        """
        Save evaluation results and plots
        
        Args:
            metrics: Dictionary with evaluation metrics
            X_test: Test input sequences
            y_test: Test target values
            y_pred: Predicted target values
        """
        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        metrics_path = RESULTS_DIR / f"transformer_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot predictions vs actual values
        n_targets = y_test.shape[1]
        plt.figure(figsize=(15, n_targets * 5))
        
        for i, target in enumerate(self.metadata['targets']):
            plt.subplot(n_targets, 1, i + 1)
            
            # Plot actual values
            plt.plot(y_test[:100, i], label='Actual')
            
            # Plot predicted values
            plt.plot(y_pred[:100, i], label='Predicted')
            
            plt.title(f'{target} - Actual vs Predicted (Transformer)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_DIR / f"transformer_predictions_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        logger.info(f"Prediction plot saved to {plot_path}")

def tune_hyperparameters(X_train, y_train, X_val, y_val, metadata, param_grid):
    """
    Perform hyperparameter tuning
    
    Args:
        X_train: Training input sequences
        y_train: Training target values
        X_val: Validation input sequences
        y_val: Validation target values
        metadata: Dictionary with metadata about the features and targets
        param_grid: Dictionary with hyperparameter options to try
        
    Returns:
        Best configuration and its validation loss
    """
    logger.info("Starting hyperparameter tuning for transformer model")
    
    best_val_loss = float('inf')
    best_config = None
    
    # Generate all combinations of hyperparameters
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    logger.info(f"Testing {len(combinations)} hyperparameter combinations")
    
    for i, combination in enumerate(combinations):
        config = dict(zip(keys, combination))
        logger.info(f"Combination {i+1}/{len(combinations)}: {config}")
        
        # Create and train model with this configuration
        transformer = QoSTransformer(config)
        history = transformer.train(X_train, y_train, X_val, y_val, metadata)
        
        # Get best validation loss
        val_loss = min(history.history['val_loss'])
        
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Update best configuration if this one is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            logger.info(f"New best configuration found!")
    
    logger.info(f"Hyperparameter tuning completed")
    logger.info(f"Best configuration: {best_config}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_config, best_val_loss

def main():
    """Main function to train and evaluate the QoS prediction transformer model"""
    logger.info("Starting QoS prediction transformer model training")
    
    # Load prepared data
    dataset_name = '5g_kpi'  # Use 5G KPI dataset as primary
    data_file = FEATURES_DIR / f"{dataset_name}_sequences.npz"
    metadata_file = FEATURES_DIR / f"{dataset_name}_metadata.pkl"
    
    if not data_file.exists() or not metadata_file.exists():
        logger.error(f"Prepared data not found. Run feature_extraction.py first.")
        return
    
    # Load data
    data = np.load(data_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded data with {X_train.shape[0]} training, {X_val.shape[0]} validation, and {X_test.shape[0]} test samples")
    logger.info(f"Input shape: {X_train.shape[1:]} (sequence_length, n_features)")
    logger.info(f"Output shape: {y_train.shape[1:]} (n_targets)")
    logger.info(f"Features: {metadata['features']}")
    logger.info(f"Targets: {metadata['targets']}")
    
    # Define model configuration
    config = {
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_transformer_blocks': 2,
        'mlp_units': [64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'patience': 15,
        'model_type': 'regression'
    }
    
    # Create and train model
    transformer = QoSTransformer(config)
    transformer.train(X_train, y_train, X_val, y_val, metadata)
    
    # Evaluate model
    metrics = transformer.evaluate(X_test, y_test)
    
    # Save model
    transformer.save()
    
    logger.info("QoS prediction transformer model training and evaluation completed")

if __name__ == "__main__":
    main()