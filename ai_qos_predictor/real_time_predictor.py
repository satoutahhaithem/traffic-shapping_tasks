#!/usr/bin/env python3
"""
Real-Time QoS Predictor for Video Streaming

This script implements a real-time QoS prediction system that can be integrated
with video streaming applications. It continuously monitors network conditions,
predicts future QoS metrics, and provides recommendations for adaptive bitrate
strategies to prevent streaming issues before they occur.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import pickle
import json
import time
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from datetime import datetime
import subprocess
import socket
import argparse

# Add models directory to path
sys.path.append(str(Path(__file__).parent / 'models'))

# Import model classes
from lstm_model import QoSPredictor
from transformer_model import QoSTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
BUFFER_SIZE = 30  # Number of seconds to keep in the buffer
UPDATE_INTERVAL = 1.0  # Seconds between updates
PREDICTION_HORIZON = 3  # Seconds ahead to predict

class NetworkMonitor:
    """Monitor network conditions in real-time"""
    
    def __init__(self, interface='eth0'):
        """
        Initialize the network monitor
        
        Args:
            interface: Network interface to monitor
        """
        self.interface = interface
        self.running = False
        self.data_queue = queue.Queue()
        self.thread = None
        self.last_bytes_rx = 0
        self.last_bytes_tx = 0
        self.last_timestamp = time.time()
    
    def start(self):
        """Start the network monitoring thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Network monitor is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started network monitoring on interface {self.interface}")
    
    def stop(self):
        """Stop the network monitoring thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Stopped network monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect network metrics
                metrics = self._collect_metrics()
                
                # Put metrics in the queue
                if metrics:
                    self.data_queue.put(metrics)
                
                # Sleep until next update
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                time.sleep(1.0)  # Sleep on error to avoid tight loop
    
    def _collect_metrics(self):
        """
        Collect network metrics
        
        Returns:
            Dictionary with network metrics
        """
        metrics = {}
        
        try:
            # Get current timestamp
            current_time = time.time()
            metrics['timestamp'] = current_time
            
            # Measure throughput
            rx_bytes, tx_bytes = self._get_interface_bytes()
            
            if self.last_bytes_rx > 0 and self.last_bytes_tx > 0:
                # Calculate throughput in Mbps
                time_diff = current_time - self.last_timestamp
                rx_diff = rx_bytes - self.last_bytes_rx
                tx_diff = tx_bytes - self.last_bytes_tx
                
                # Convert bytes to bits and then to Mbps
                metrics['throughput_down'] = (rx_diff * 8) / (time_diff * 1_000_000)
                metrics['throughput_up'] = (tx_diff * 8) / (time_diff * 1_000_000)
            else:
                metrics['throughput_down'] = 0.0
                metrics['throughput_up'] = 0.0
            
            # Update last values
            self.last_bytes_rx = rx_bytes
            self.last_bytes_tx = tx_bytes
            self.last_timestamp = current_time
            
            # Measure RTT, jitter, and packet loss
            rtt, jitter, loss = self._measure_network_quality()
            metrics['rtt'] = rtt
            metrics['jitter'] = jitter
            metrics['packet_loss'] = loss
            
            # Try to get signal strength metrics if available
            signal_metrics = self._get_signal_metrics()
            metrics.update(signal_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def _get_interface_bytes(self):
        """
        Get bytes received and transmitted on the interface
        
        Returns:
            Tuple of (rx_bytes, tx_bytes)
        """
        try:
            # Read from /proc/net/dev
            with open('/proc/net/dev', 'r') as f:
                for line in f:
                    if self.interface in line:
                        # Extract bytes
                        parts = line.split(':')[1].strip().split()
                        rx_bytes = int(parts[0])
                        tx_bytes = int(parts[8])
                        return rx_bytes, tx_bytes
            
            # Interface not found
            logger.warning(f"Interface {self.interface} not found in /proc/net/dev")
            return 0, 0
            
        except Exception as e:
            logger.error(f"Error getting interface bytes: {e}")
            return 0, 0
    
    def _measure_network_quality(self):
        """
        Measure RTT, jitter, and packet loss using ping
        
        Returns:
            Tuple of (rtt, jitter, loss)
        """
        try:
            # Use ping to measure RTT and packet loss
            # Ping Google's DNS server as a reliable target
            cmd = ['ping', '-c', '3', '-i', '0.2', '8.8.8.8']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output
            output = result.stdout
            
            # Extract RTT
            rtt = 0.0
            jitter = 0.0
            loss = 0.0
            
            if 'min/avg/max/mdev' in output:
                # Extract RTT statistics
                stats_line = output.split('min/avg/max/mdev')[1].strip()
                stats = stats_line.split('=')[1].strip().split('/')
                
                # min/avg/max/mdev
                rtt = float(stats[1])  # avg
                jitter = float(stats[3].split()[0])  # mdev
            
            if 'packet loss' in output:
                # Extract packet loss
                loss_part = output.split('packet loss')[0].strip().split()[-1]
                loss = float(loss_part.replace('%', ''))
            
            return rtt, jitter, loss
            
        except Exception as e:
            logger.error(f"Error measuring network quality: {e}")
            return 0.0, 0.0, 0.0
    
    def _get_signal_metrics(self):
        """
        Get signal strength metrics if available
        
        Returns:
            Dictionary with signal metrics
        """
        metrics = {}
        
        try:
            # This is a simplified approach - in a real implementation,
            # you would use platform-specific methods to get signal strength
            # For example, on Linux you might use iw/iwconfig, on Windows you'd use WMI
            
            # For now, we'll just return empty metrics
            # In a real implementation, you might add:
            # metrics['sinr'] = ...
            # metrics['rsrp'] = ...
            # metrics['rsrq'] = ...
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting signal metrics: {e}")
            return metrics
    
    def get_latest_metrics(self):
        """
        Get the latest metrics from the queue
        
        Returns:
            Dictionary with the latest metrics, or None if no metrics are available
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

class QoSPredictionSystem:
    """Real-time QoS prediction system"""
    
    def __init__(self, model_path, config_path=None, buffer_size=BUFFER_SIZE):
        """
        Initialize the QoS prediction system
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the model configuration
            buffer_size: Number of seconds to keep in the buffer
        """
        self.buffer_size = buffer_size
        self.metrics_buffer = pd.DataFrame()
        self.predictions = None
        self.model = None
        self.model_type = None
        self.load_model(model_path, config_path)
    
    def load_model(self, model_path, config_path=None):
        """
        Load a trained model
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the model configuration
        """
        try:
            # Determine model type from filename
            model_path = Path(model_path)
            if 'lstm' in model_path.stem.lower():
                logger.info("Loading LSTM model")
                self.model = QoSPredictor()
                self.model_type = 'lstm'
            elif 'transformer' in model_path.stem.lower():
                logger.info("Loading Transformer model")
                self.model = QoSTransformer()
                self.model_type = 'transformer'
            else:
                # Default to LSTM
                logger.info("Model type not specified in filename, defaulting to LSTM")
                self.model = QoSPredictor()
                self.model_type = 'lstm'
            
            # Load the model
            self.model.load(str(model_path), config_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Get metadata
            if self.model.metadata:
                logger.info(f"Model predicts: {self.model.metadata['targets']}")
                logger.info(f"Model features: {self.model.metadata['features']}")
                
                # Initialize predictions DataFrame
                self.predictions = pd.DataFrame(columns=self.model.metadata['targets'])
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def add_metrics(self, metrics):
        """
        Add new metrics to the buffer
        
        Args:
            metrics: Dictionary with network metrics
        """
        if metrics is None:
            return
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics])
        
        # Add to buffer
        if self.metrics_buffer.empty:
            self.metrics_buffer = metrics_df
        else:
            self.metrics_buffer = pd.concat([self.metrics_buffer, metrics_df], ignore_index=True)
        
        # Trim buffer to keep only the most recent data
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer = self.metrics_buffer.iloc[-self.buffer_size:]
    
    def prepare_input_sequence(self):
        """
        Prepare input sequence for the model
        
        Returns:
            Numpy array with input sequence, or None if not enough data
        """
        if self.model is None or self.model.metadata is None:
            logger.error("Model not loaded or missing metadata")
            return None
        
        # Check if we have enough data
        sequence_length = self.model.metadata.get('sequence_length', 10)
        if len(self.metrics_buffer) < sequence_length:
            logger.debug(f"Not enough data for prediction: {len(self.metrics_buffer)}/{sequence_length}")
            return None
        
        # Get the features needed by the model
        features = self.model.metadata['features']
        
        # Check which features are available in the buffer
        available_features = [f for f in features if f in self.metrics_buffer.columns]
        missing_features = [f for f in features if f not in self.metrics_buffer.columns]
        
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            
            # Add missing features with zeros
            for feature in missing_features:
                self.metrics_buffer[feature] = 0.0
        
        # Get the most recent data
        recent_data = self.metrics_buffer.iloc[-sequence_length:][features].values
        
        # Reshape for model input (batch_size, sequence_length, n_features)
        X = np.expand_dims(recent_data, axis=0)
        
        return X
    
    def predict(self):
        """
        Make a prediction based on current metrics
        
        Returns:
            Dictionary with predictions, or None if prediction failed
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        # Prepare input sequence
        X = self.prepare_input_sequence()
        if X is None:
            return None
        
        try:
            # Make prediction
            y_pred = self.model.predict(X)
            
            # Print debug information
            print(f"Raw prediction shape: {y_pred.shape}")
            print(f"Raw prediction values: {y_pred}")
            
            # Convert to dictionary
            predictions = {}
            for i, target in enumerate(self.model.metadata['targets']):
                predictions[target] = float(y_pred[0, i])  # Convert to float to avoid nan issues
                print(f"Prediction for {target}: {predictions[target]}")
            
            # Add timestamp
            predictions['timestamp'] = time.time()
            
            # Add to predictions DataFrame
            pred_df = pd.DataFrame([predictions])
            if self.predictions is None:
                self.predictions = pred_df
            else:
                self.predictions = pd.concat([self.predictions, pred_df], ignore_index=True)
            
            # Trim predictions to keep only recent ones
            if len(self.predictions) > self.buffer_size:
                self.predictions = self.predictions.iloc[-self.buffer_size:]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_recommendation(self, prediction):
        """
        Get a recommendation based on the prediction
        
        Args:
            prediction: Dictionary with predictions
            
        Returns:
            Dictionary with recommendations
        """
        if prediction is None:
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
        if 'throughput_down' in prediction:
            throughput = prediction['throughput_down']
            current_throughput = self.metrics_buffer['throughput_down'].iloc[-1] if not self.metrics_buffer.empty else 0
            
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
        if 'rtt' in prediction:
            rtt = prediction['rtt']
            current_rtt = self.metrics_buffer['rtt'].iloc[-1] if not self.metrics_buffer.empty else 0
            
            # Calculate percent change
            if current_rtt > 0:
                percent_change = (rtt - current_rtt) / current_rtt * 100
                
                if percent_change > 50 and rtt > 100:
                    recommendation['action'] = 'increase_buffer'
                    recommendation['reason'] = f'Predicted RTT increase: {percent_change:.1f}%'
                    recommendation['confidence'] = min(abs(percent_change) / 100, 0.9)
        
        return recommendation

class RealTimePlotter:
    """Real-time visualization of network metrics and predictions"""
    
    def __init__(self, prediction_system, buffer_size=BUFFER_SIZE):
        """
        Initialize the real-time plotter
        
        Args:
            prediction_system: QoSPredictionSystem instance
            buffer_size: Number of seconds to display
        """
        self.prediction_system = prediction_system
        self.buffer_size = buffer_size
        self.fig = None
        self.axes = None
        self.animation = None
        self.recommendation_text = None
    
    def start(self):
        """Start the real-time visualization"""
        # Create figure and axes
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Set up axes
        self.axes[0].set_title('Network Throughput')
        self.axes[0].set_ylabel('Mbps')
        self.axes[0].grid(True)
        
        self.axes[1].set_title('Round-Trip Time')
        self.axes[1].set_ylabel('ms')
        self.axes[1].grid(True)
        
        self.axes[2].set_title('Packet Loss')
        self.axes[2].set_ylabel('%')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].grid(True)
        
        # Add text box for recommendations
        self.recommendation_text = self.fig.text(0.5, 0.01, '', ha='center', va='bottom',
                                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create animation
        self.animation = FuncAnimation(self.fig, self._update_plot, interval=1000, blit=False)
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for recommendation text
        plt.show()
    
    def _update_plot(self, frame):
        """
        Update the plot with new data
        
        Args:
            frame: Animation frame number
        """
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Set up axes again
        self.axes[0].set_title('Network Throughput')
        self.axes[0].set_ylabel('Mbps')
        self.axes[0].grid(True)
        
        self.axes[1].set_title('Round-Trip Time')
        self.axes[1].set_ylabel('ms')
        self.axes[1].grid(True)
        
        self.axes[2].set_title('Packet Loss')
        self.axes[2].set_ylabel('%')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].grid(True)
        
        # Get data from prediction system
        metrics_df = self.prediction_system.metrics_buffer
        predictions_df = self.prediction_system.predictions
        
        if not metrics_df.empty:
            # Plot throughput
            if 'throughput_down' in metrics_df.columns:
                x = range(len(metrics_df))
                self.axes[0].plot(x, metrics_df['throughput_down'], 'b-', label='Download')
            
            if 'throughput_up' in metrics_df.columns:
                x = range(len(metrics_df))
                self.axes[0].plot(x, metrics_df['throughput_up'], 'g-', label='Upload')
            
            self.axes[0].legend()
            
            # Plot RTT
            if 'rtt' in metrics_df.columns:
                x = range(len(metrics_df))
                self.axes[1].plot(x, metrics_df['rtt'], 'r-', label='Current')
            
            # Plot packet loss
            if 'packet_loss' in metrics_df.columns:
                x = range(len(metrics_df))
                self.axes[2].plot(x, metrics_df['packet_loss'], 'm-', label='Current')
            
            # Plot predictions if available
            if predictions_df is not None and not predictions_df.empty:
                # Align predictions with metrics
                pred_offset = len(metrics_df)
                
                # Plot throughput prediction
                if 'throughput_down' in predictions_df.columns:
                    x = [pred_offset]
                    # Make predictions more visible
                    self.axes[0].plot(x, predictions_df['throughput_down'].iloc[-1:], 'bo', markersize=12, label='Predicted')
                    self.axes[0].plot([pred_offset-1, pred_offset],
                                     [metrics_df['throughput_down'].iloc[-1], predictions_df['throughput_down'].iloc[-1]],
                                     'b--', linewidth=2, alpha=0.8)
                    
                    # Print prediction value for debugging
                    print(f"Throughput prediction: {predictions_df['throughput_down'].iloc[-1]}")
                
                # Plot RTT prediction
                if 'rtt' in predictions_df.columns:
                    x = [pred_offset]
                    # Make predictions more visible
                    self.axes[1].plot(x, predictions_df['rtt'].iloc[-1:], 'ro', markersize=12, label='Predicted')
                    self.axes[1].plot([pred_offset-1, pred_offset],
                                     [metrics_df['rtt'].iloc[-1], predictions_df['rtt'].iloc[-1]],
                                     'r--', linewidth=2, alpha=0.8)
                    
                    # Print prediction value for debugging
                    print(f"RTT prediction: {predictions_df['rtt'].iloc[-1]}")
                
                # Plot stall probability if available
                # Plot packet loss prediction
                if 'packet_loss' in predictions_df.columns:
                    x = [pred_offset]
                    self.axes[2].plot(x, predictions_df['packet_loss'].iloc[-1:], 'mo', markersize=12, label='Predicted')
                    self.axes[2].plot([pred_offset-1, pred_offset],
                                     [metrics_df['packet_loss'].iloc[-1], predictions_df['packet_loss'].iloc[-1]],
                                     'm--', linewidth=2, alpha=0.8)
                    
                    # Print prediction value for debugging
                    print(f"Packet loss prediction: {predictions_df['packet_loss'].iloc[-1]}")
                
                # Plot stall probability if available
                if 'stall_event' in predictions_df.columns:
                    stall_prob = predictions_df['stall_event'].iloc[-1]
                    self.axes[2].axhline(y=stall_prob*100, color='r', linestyle='--', linewidth=2,
                                        label=f'Stall Prob: {stall_prob:.2f}')
            
            # Add legends
            for ax in self.axes:
                ax.legend()
            
            # Update recommendation text
            if predictions_df is not None and not predictions_df.empty:
                # Get latest prediction
                latest_pred = {col: predictions_df[col].iloc[-1] for col in predictions_df.columns}
                
                # Get recommendation
                recommendation = self.prediction_system.get_recommendation(latest_pred)
                
                if recommendation:
                    text = f"Recommendation: {recommendation['action'].upper()} - {recommendation['reason']} (Confidence: {recommendation['confidence']:.2f})"
                    self.recommendation_text.set_text(text)
                    
                    # Set color based on action
                    if recommendation['action'] == 'reduce_bitrate':
                        self.recommendation_text.set_bbox(dict(boxstyle='round', facecolor='salmon', alpha=0.5))
                    elif recommendation['action'] == 'increase_buffer':
                        self.recommendation_text.set_bbox(dict(boxstyle='round', facecolor='khaki', alpha=0.5))
                    elif recommendation['action'] == 'increase_bitrate':
                        self.recommendation_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                    else:
                        self.recommendation_text.set_bbox(dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def main():
    """Main function to run the real-time QoS prediction system"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-Time QoS Predictor for Video Streaming')
    parser.add_argument('--model', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--interface', type=str, default='eth0', help='Network interface to monitor')
    parser.add_argument('--buffer-size', type=int, default=BUFFER_SIZE, help='Number of seconds to keep in the buffer')
    parser.add_argument('--update-interval', type=float, default=UPDATE_INTERVAL, help='Seconds between updates')
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model
    if model_path is None:
        # Look for models in the models directory
        model_files = list(MODELS_DIR.glob('*.h5'))
        if model_files:
            # Use the most recent model
            model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
            logger.info(f"Using most recent model: {model_path}")
        else:
            logger.error("No models found. Please train a model first or specify a model path.")
            return
    
    try:
        # Initialize QoS prediction system
        prediction_system = QoSPredictionSystem(model_path, buffer_size=args.buffer_size)
        
        # Initialize network monitor
        monitor = NetworkMonitor(interface=args.interface)
        
        # Start network monitoring
        monitor.start()
        
        # Initialize plotter
        plotter = RealTimePlotter(prediction_system, buffer_size=args.buffer_size)
        
        # Main loop
        logger.info("Starting real-time QoS prediction")
        
        # Start plotter in a separate thread
        plot_thread = threading.Thread(target=plotter.start)
        plot_thread.daemon = True
        plot_thread.start()
        
        try:
            while True:
                # Get latest metrics
                metrics = monitor.get_latest_metrics()
                
                if metrics:
                    # Add to prediction system
                    prediction_system.add_metrics(metrics)
                    
                    # Make prediction
                    prediction = prediction_system.predict()
                    
                    if prediction:
                        # Get recommendation
                        recommendation = prediction_system.get_recommendation(prediction)
                        
                        if recommendation:
                            logger.info(f"Recommendation: {recommendation['action']} - {recommendation['reason']}")
                
                # Sleep until next update
                time.sleep(args.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping real-time QoS prediction")
        
        finally:
            # Stop network monitoring
            monitor.stop()
    
    except Exception as e:
        logger.error(f"Error in real-time QoS prediction: {e}")

if __name__ == "__main__":
    main()