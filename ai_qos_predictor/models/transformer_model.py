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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

class PositionalEncoding(nn.Module):
    """Positional encoding layer for transformer models"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent but not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size, output_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, dropout=0.2):
        """
        Initialize the Transformer model
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Feature projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.global_avg_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # Pass through MLP head
        x = self.mlp(x)
        
        return x

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
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, input_shape, output_shape):
        """
        Build the Transformer model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            output_shape: Number of output features to predict
        """
        # Create model
        self.model = TransformerModel(
            input_size=input_shape[1],
            output_size=output_shape,
            d_model=self.config['embed_dim'],
            nhead=self.config['num_heads'],
            num_layers=self.config['num_transformer_blocks'],
            dim_feedforward=self.config['ff_dim'],
            dropout=self.config['dropout_rate']
        ).to(self.device)
        
        logger.info(f"Built {self.config['model_type']} transformer model with {self.config['num_transformer_blocks']} transformer blocks")
        
        return self.model
    
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
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        if self.config['model_type'] == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        logger.info("Starting transformer model training")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                train_loss += loss.item() * inputs.size(0)
                train_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
            
            # Store metrics
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                       f"Loss: {train_loss:.4f} - MAE: {train_mae:.4f} - "
                       f"Val Loss: {val_loss:.4f} - Val MAE: {val_mae:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                # Save best model
                torch.save(self.model.state_dict(), MODELS_DIR / 'best_transformer_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info("Transformer model training completed")
        
        # Save training history
        self._save_history()
        
        return self.history
    
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
        
        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).cpu().numpy()
        
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
        
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        
        return y_pred
    
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
        model_path = MODELS_DIR / f"{model_name}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration and metadata
        config_path = MODELS_DIR / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'metadata': self.metadata,
                'input_size': self.model.input_projection.in_features,
                'd_model': self.model.d_model,
                'output_size': self.model.mlp[-1].out_features
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
                
                # Get model architecture parameters
                input_size = data.get('input_size')
                d_model = data.get('d_model')
                output_size = data.get('output_size')
                
                # Create model with the same architecture
                self.model = TransformerModel(
                    input_size=input_size,
                    output_size=output_size,
                    d_model=d_model,
                    nhead=self.config['num_heads'],
                    num_layers=self.config['num_transformer_blocks'],
                    dim_feedforward=self.config['ff_dim'],
                    dropout=self.config['dropout_rate']
                ).to(self.device)
        else:
            logger.error(f"Configuration file not found: {config_path}")
            return
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        if self.metadata:
            logger.info(f"Model predicts: {self.metadata['targets']}")
    
    def _save_history(self):
        """Save training history and plots"""
        if not self.history['loss']:
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
        plt.plot(self.history['mae'], label='Training MAE')
        plt.plot(self.history['val_mae'], label='Validation MAE')
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
            
            plt.title(f'{target} - Actual vs Predicted')
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
    logger.info("Starting hyperparameter tuning")
    
    best_val_loss = float('inf')
    best_config = None
    
    # Generate all combinations of hyperparameters
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    logger.info(f"Testing {len(combinations)} hyperparameter combinations")
    
    # Test each combination
    for i, combination in enumerate(combinations):
        config = dict(zip(keys, combination))
        logger.info(f"Testing combination {i+1}/{len(combinations)}: {config}")
        
        # Create and train model with this configuration
        transformer = QoSTransformer(config)
        history = transformer.train(X_train, y_train, X_val, y_val, metadata)
        
        # Get the best validation loss
        val_loss = min(history['val_loss'])
        
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Check if this is the best configuration
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            logger.info(f"New best configuration found!")
    
    logger.info(f"Best configuration: {best_config}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_config, best_val_loss

def main():
    """Main function for testing the Transformer model"""
    # Load data
    data_file = FEATURES_DIR / "5g_kpi_sequences.npz"
    metadata_file = FEATURES_DIR / "5g_kpi_metadata.pkl"
    
    if not data_file.exists() or not metadata_file.exists():
        logger.error("Data files not found. Please run the preprocessing script first.")
        return
    
    # Load sequences
    sequences = np.load(data_file)
    X_train = sequences['X_train']
    y_train = sequences['y_train']
    X_val = sequences['X_val']
    y_val = sequences['y_val']
    X_test = sequences['X_test']
    y_test = sequences['y_test']
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded data with {X_train.shape[0]} training samples")
    logger.info(f"Features: {metadata['features']}")
    logger.info(f"Targets: {metadata['targets']}")
    
    # Create and train model
    transformer = QoSTransformer()
    transformer.train(X_train, y_train, X_val, y_val, metadata)
    
    # Evaluate model
    metrics = transformer.evaluate(X_test, y_test)
    
    # Save model
    transformer.save()

if __name__ == "__main__":
    main()