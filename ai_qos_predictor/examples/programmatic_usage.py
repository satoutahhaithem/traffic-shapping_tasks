#!/usr/bin/env python3
"""
Example of programmatic usage of the AI QoS Predictor

This script demonstrates how to load a trained model and use it
to make predictions programmatically, which is useful for integrating
the QoS prediction capabilities into your own applications.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

# Import model classes
from models.lstm_model import QoSPredictor
from models.transformer_model import QoSTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, model_type='lstm'):
    """
    Load a trained model
    
    Args:
        model_path: Path to the trained model
        model_type: Type of model ('lstm' or 'transformer')
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} model from {model_path}")
    
    try:
        if model_type == 'lstm':
            model = QoSPredictor()
        elif model_type == 'transformer':
            model = QoSTransformer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load(model_path)
        logger.info("Model loaded successfully")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def create_input_sequence(network_metrics, sequence_length=10):
    """
    Create an input sequence for the model from network metrics
    
    Args:
        network_metrics: DataFrame with network metrics
        sequence_length: Length of the sequence required by the model
        
    Returns:
        Numpy array with input sequence
    """
    # Ensure we have enough data
    if len(network_metrics) < sequence_length:
        logger.warning(f"Not enough data: {len(network_metrics)}/{sequence_length}")
        return None
    
    # Get the most recent data
    recent_data = network_metrics.iloc[-sequence_length:].values
    
    # Reshape for model input (batch_size, sequence_length, n_features)
    X = np.expand_dims(recent_data, axis=0)
    
    return X

def make_prediction(model, input_sequence):
    """
    Make a prediction using the model
    
    Args:
        model: Trained model
        input_sequence: Input sequence for the model
        
    Returns:
        Dictionary with predictions
    """
    if model is None or input_sequence is None:
        return None
    
    try:
        # Make prediction
        y_pred = model.predict(input_sequence)
        
        # Convert to dictionary
        predictions = {}
        for i, target in enumerate(model.metadata['targets']):
            predictions[target] = float(y_pred[0, i])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def get_recommendation(prediction, current_metrics):
    """
    Get a recommendation based on the prediction
    
    Args:
        prediction: Dictionary with predictions
        current_metrics: Dictionary with current network metrics
        
    Returns:
        Dictionary with recommendation
    """
    if prediction is None or current_metrics is None:
        return None
    
    recommendation = {
        'action': 'maintain',
        'reason': 'Network conditions are stable',
        'confidence': 0.0
    }
    
    # Check for stall prediction
    if 'stall_event' in prediction:
        stall_prob = prediction['stall_event']
        if stall_prob > 0.7:
            recommendation['action'] = 'reduce_bitrate'
            recommendation['reason'] = f'High stall probability: {stall_prob:.2f}'
            recommendation['confidence'] = stall_prob
        elif stall_prob > 0.4:
            recommendation['action'] = 'increase_buffer'
            recommendation['reason'] = f'Moderate stall probability: {stall_prob:.2f}'
            recommendation['confidence'] = stall_prob
    
    # Check for throughput prediction
    if 'throughput_down' in prediction and 'throughput_down' in current_metrics:
        throughput = prediction['throughput_down']
        current_throughput = current_metrics['throughput_down']
        
        # Calculate percent change
        if current_throughput > 0:
            percent_change = (throughput - current_throughput) / current_throughput * 100
            
            if percent_change < -30:
                recommendation['action'] = 'reduce_bitrate'
                recommendation['reason'] = f'Predicted throughput drop: {percent_change:.1f}%'
                recommendation['confidence'] = min(abs(percent_change) / 100, 0.9)
            elif percent_change > 30:
                recommendation['action'] = 'increase_bitrate'
                recommendation['reason'] = f'Predicted throughput increase: {percent_change:.1f}%'
                recommendation['confidence'] = min(abs(percent_change) / 100, 0.9)
    
    # Check for RTT prediction
    if 'rtt' in prediction and 'rtt' in current_metrics:
        rtt = prediction['rtt']
        current_rtt = current_metrics['rtt']
        
        # Calculate percent change
        if current_rtt > 0:
            percent_change = (rtt - current_rtt) / current_rtt * 100
            
            if percent_change > 50 and rtt > 100:
                recommendation['action'] = 'increase_buffer'
                recommendation['reason'] = f'Predicted RTT increase: {percent_change:.1f}%'
                recommendation['confidence'] = min(abs(percent_change) / 100, 0.9)
    
    return recommendation

def simulate_network_metrics(num_samples=100):
    """
    Simulate network metrics for demonstration purposes
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with simulated network metrics
    """
    # Create time index
    timestamps = pd.date_range(start='2025-01-01', periods=num_samples, freq='1s')
    
    # Create base metrics with some randomness
    throughput_down = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, num_samples)) + np.random.normal(0, 2, num_samples)
    throughput_up = 5 + 2 * np.sin(np.linspace(0, 4*np.pi, num_samples)) + np.random.normal(0, 0.5, num_samples)
    rtt = 50 + 20 * np.sin(np.linspace(0, 2*np.pi, num_samples)) + np.random.normal(0, 5, num_samples)
    jitter = 5 + 2 * np.sin(np.linspace(0, 3*np.pi, num_samples)) + np.random.normal(0, 1, num_samples)
    packet_loss = 1 + 1 * np.sin(np.linspace(0, 2*np.pi, num_samples)) + np.random.normal(0, 0.2, num_samples)
    
    # Ensure non-negative values
    throughput_down = np.maximum(throughput_down, 0)
    throughput_up = np.maximum(throughput_up, 0)
    rtt = np.maximum(rtt, 1)
    jitter = np.maximum(jitter, 0)
    packet_loss = np.maximum(packet_loss, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'throughput_down': throughput_down,
        'throughput_up': throughput_up,
        'rtt': rtt,
        'jitter': jitter,
        'packet_loss': packet_loss
    })
    
    return df

def main():
    """Main function demonstrating programmatic usage"""
    # Path to the trained model
    model_dir = Path(__file__).parent.parent / "models"
    
    # Find the most recent LSTM model
    model_files = list(model_dir.glob("*lstm*.h5"))
    if not model_files:
        logger.error("No LSTM model found. Please train a model first.")
        return
    
    model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
    
    # Load the model
    model = load_model(model_path, model_type='lstm')
    if model is None:
        return
    
    # Print model information
    logger.info(f"Model targets: {model.metadata['targets']}")
    logger.info(f"Model features: {model.metadata['features']}")
    logger.info(f"Sequence length: {model.metadata.get('sequence_length', 10)}")
    
    # Simulate network metrics
    logger.info("Simulating network metrics...")
    network_metrics = simulate_network_metrics(num_samples=30)
    
    # Create input sequence
    sequence_length = model.metadata.get('sequence_length', 10)
    input_sequence = create_input_sequence(
        network_metrics[model.metadata['features']], 
        sequence_length=sequence_length
    )
    
    if input_sequence is None:
        logger.error("Failed to create input sequence")
        return
    
    # Make prediction
    logger.info("Making prediction...")
    prediction = make_prediction(model, input_sequence)
    
    if prediction:
        logger.info("Prediction:")
        for target, value in prediction.items():
            logger.info(f"  {target}: {value:.4f}")
        
        # Get current metrics
        current_metrics = {
            col: network_metrics[col].iloc[-1] 
            for col in network_metrics.columns 
            if col != 'timestamp'
        }
        
        # Get recommendation
        recommendation = get_recommendation(prediction, current_metrics)
        
        if recommendation:
            logger.info("\nRecommendation:")
            logger.info(f"  Action: {recommendation['action']}")
            logger.info(f"  Reason: {recommendation['reason']}")
            logger.info(f"  Confidence: {recommendation['confidence']:.2f}")
    
    # Example of integration with a video player
    logger.info("\nExample integration with video player:")
    logger.info("----------------------------------------")
    logger.info("class AdaptiveVideoPlayer:")
    logger.info("    def __init__(self, qos_predictor):")
    logger.info("        self.qos_predictor = qos_predictor")
    logger.info("        self.current_bitrate = 1000000  # 1 Mbps")
    logger.info("        self.buffer_size = 5  # 5 seconds")
    logger.info("")
    logger.info("    def adapt_playback(self, network_metrics):")
    logger.info("        # Create input sequence")
    logger.info("        input_sequence = create_input_sequence(network_metrics)")
    logger.info("")
    logger.info("        # Make prediction")
    logger.info("        prediction = self.qos_predictor.predict(input_sequence)")
    logger.info("")
    logger.info("        # Get recommendation")
    logger.info("        recommendation = get_recommendation(prediction, current_metrics)")
    logger.info("")
    logger.info("        # Apply recommendation")
    logger.info("        if recommendation['action'] == 'reduce_bitrate':")
    logger.info("            self.current_bitrate *= 0.7  # Reduce by 30%")
    logger.info("        elif recommendation['action'] == 'increase_bitrate':")
    logger.info("            self.current_bitrate *= 1.3  # Increase by 30%")
    logger.info("        elif recommendation['action'] == 'increase_buffer':")
    logger.info("            self.buffer_size += 2  # Add 2 seconds to buffer")

if __name__ == "__main__":
    main()