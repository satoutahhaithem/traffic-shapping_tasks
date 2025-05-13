#!/usr/bin/env python3
"""
LSTM Model for QoS Prediction

This script implements an LSTM-based model for predicting network QoS metrics
based on historical network conditions. It includes model definition, training,
evaluation, and hyperparameter tuning capabilities.
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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
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

class QoSPredictor:
    """LSTM-based QoS prediction model"""
    
    def __init__(self, config=None):
        """
        Initialize the QoS predictor
        
        Args:
            config: Dictionary with model configuration parameters
        """
        # Default configuration
        self.config = {
            'lstm_units': 64,
            'lstm_layers': 2,
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
        Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            output_shape: Number of output features to predict
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=self.config['lstm_units'],
            input_shape=input_shape,
            return_sequences=self.config['lstm_layers'] > 1,
            activation='tanh',
            recurrent_activation='sigmoid'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layers if specified
        for i in range(1, self.config['lstm_layers']):
            return_sequences = i < self.config['lstm_layers'] - 1
            model.add(LSTM(
                units=self.config['lstm_units'],
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        if self.config['model_type'] == 'regression':
            model.add(Dense(output_shape, activation='linear'))
        elif self.config['model_type'] == 'classification':
            model.add(Dense(output_shape, activation='sigmoid'))
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
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
        logger.info(f"Built {self.config['model_type']} model with {self.config['lstm_layers']} LSTM layers")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, metadata=None):
        """
        Train the LSTM model
        
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
                filepath=str(MODELS_DIR / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting model training")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        
        self.history = history.history
        logger.info("Model training completed")
        
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
            model_name = f"qos_predictor_{model_type}_{timestamp}"
        
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
        self.model = load_model(model_path)
        
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
        history_path = RESULTS_DIR / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        for metric in self.history:
            if metric not in ['loss', 'val_loss']:
                plt.plot(self.history[metric], label=metric)
        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_DIR / f"training_history_{timestamp}.png"
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
        metrics_path = RESULTS_DIR / f"evaluation_metrics_{timestamp}.json"
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
            
            plt.title(f'{target} - Actual vs Predicted')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_DIR / f"predictions_{timestamp}.png"
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
    logger.info("Starting hyperparameter tuning")
    
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
        predictor = QoSPredictor(config)
        history = predictor.train(X_train, y_train, X_val, y_val, metadata)
        
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
    """Main function to train and evaluate the QoS prediction model"""
    logger.info("Starting QoS prediction model training")
    
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
        'lstm_units': 128,
        'lstm_layers': 2,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'patience': 15,
        'model_type': 'regression'
    }
    
    # Create and train model
    predictor = QoSPredictor(config)
    predictor.train(X_train, y_train, X_val, y_val, metadata)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save()
    
    logger.info("QoS prediction model training and evaluation completed")

if __name__ == "__main__":
    main()