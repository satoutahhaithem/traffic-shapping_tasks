#!/usr/bin/env python3
"""
Example: Integrating AI QoS Predictor with a Video Player

This script demonstrates how to integrate the AI QoS Predictor with a video player
to enable proactive quality adaptation based on predicted network conditions.
"""

import os
import sys
import time
import logging
import threading
import numpy as np
from pathlib import Path
import tensorflow as tf

# Add parent directory to path to import from ai_qos_predictor
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock video player class (in a real implementation, this would be replaced with an actual video player API)
class VideoPlayer:
    """Mock video player for demonstration purposes"""
    
    QUALITY_LEVELS = {
        'low': {'bitrate': 500000, 'resolution': '480p'},
        'medium': {'bitrate': 1500000, 'resolution': '720p'},
        'high': {'bitrate': 3000000, 'resolution': '1080p'},
        'ultra': {'bitrate': 6000000, 'resolution': '4K'}
    }
    
    def __init__(self):
        self.current_quality = 'medium'
        self.buffer_level = 5.0  # seconds
        self.is_playing = False
        self.playback_position = 0.0
        
    def start_playback(self):
        """Start video playback"""
        self.is_playing = True
        logger.info(f"Started playback at {self.current_quality} quality")
        
    def stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        logger.info("Stopped playback")
        
    def set_quality(self, quality_level):
        """Set video quality level"""
        if quality_level in self.QUALITY_LEVELS:
            self.current_quality = quality_level
            logger.info(f"Quality changed to {quality_level}: {self.QUALITY_LEVELS[quality_level]}")
        else:
            logger.error(f"Unknown quality level: {quality_level}")
    
    def get_buffer_level(self):
        """Get current buffer level in seconds"""
        return self.buffer_level
    
    def update_buffer(self, downloaded_bytes, network_throughput):
        """Update buffer based on downloaded data and network conditions"""
        quality_bitrate = self.QUALITY_LEVELS[self.current_quality]['bitrate']
        
        # Calculate how many seconds of video were downloaded
        seconds_downloaded = (downloaded_bytes * 8) / quality_bitrate
        
        # If playing, reduce buffer by elapsed time
        if self.is_playing:
            self.buffer_level = max(0, self.buffer_level + seconds_downloaded - 1.0)
            self.playback_position += 1.0
        else:
            self.buffer_level += seconds_downloaded
            
        logger.info(f"Buffer level: {self.buffer_level:.2f}s, Position: {self.playback_position:.2f}s")
        
        return self.buffer_level

# QoS Predictor class
class QoSPredictor:
    """QoS Predictor for network conditions"""
    
    def __init__(self, model_path, sequence_length=10, features=None):
        """
        Initialize the QoS predictor
        
        Args:
            model_path: Path to the trained model
            sequence_length: Length of input sequences
            features: List of feature names (if None, default features are used)
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.features = features or ['throughput_down', 'throughput_up', 'rtt', 'jitter', 
                                    'packet_loss', 'sinr', 'rsrp', 'rsrq']
        self.targets = ['throughput_down', 'rtt', 'packet_loss']
        
        # Load the model
        self.load_model()
        
        # Initialize the measurement buffer
        self.measurements = []
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def add_measurement(self, measurement):
        """
        Add a new network measurement
        
        Args:
            measurement: Dictionary with network metrics
        """
        # Ensure all required features are present
        for feature in self.features:
            if feature not in measurement:
                measurement[feature] = 0.0
        
        # Add to measurements buffer
        self.measurements.append([measurement[f] for f in self.features])
        
        # Keep only the most recent measurements
        if len(self.measurements) > self.sequence_length:
            self.measurements.pop(0)
    
    def predict(self):
        """
        Predict future network conditions
        
        Returns:
            Dictionary with predicted values for each target
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        if len(self.measurements) < self.sequence_length:
            logger.warning(f"Not enough measurements ({len(self.measurements)}/{self.sequence_length})")
            return None
        
        # Prepare input sequence
        X = np.array([self.measurements])
        
        # Make prediction
        y_pred = self.model.predict(X, verbose=0)[0]
        
        # Create result dictionary
        result = {target: y_pred[i] for i, target in enumerate(self.targets)}
        
        return result

# Adaptive Bitrate Controller
class ABRController:
    """Adaptive Bitrate Controller using QoS predictions"""
    
    def __init__(self, video_player, qos_predictor):
        """
        Initialize the ABR controller
        
        Args:
            video_player: VideoPlayer instance
            qos_predictor: QoSPredictor instance
        """
        self.video_player = video_player
        self.qos_predictor = qos_predictor
        self.running = False
        self.network_monitor_thread = None
        
        # Thresholds for quality adaptation
        self.throughput_thresholds = {
            'low': 1000000,    # 1 Mbps
            'medium': 2500000, # 2.5 Mbps
            'high': 5000000    # 5 Mbps
        }
        
        # Buffer thresholds (in seconds)
        self.buffer_thresholds = {
            'critical': 2.0,
            'low': 5.0,
            'high': 15.0
        }
    
    def start(self):
        """Start the ABR controller"""
        self.running = True
        self.network_monitor_thread = threading.Thread(target=self._network_monitor_loop)
        self.network_monitor_thread.daemon = True
        self.network_monitor_thread.start()
        logger.info("ABR controller started")
    
    def stop(self):
        """Stop the ABR controller"""
        self.running = False
        if self.network_monitor_thread:
            self.network_monitor_thread.join(timeout=1.0)
        logger.info("ABR controller stopped")
    
    def _network_monitor_loop(self):
        """Network monitoring and adaptation loop"""
        while self.running:
            # Simulate network measurement (in a real implementation, this would come from actual network monitoring)
            measurement = self._simulate_network_measurement()
            
            # Add measurement to predictor
            self.qos_predictor.add_measurement(measurement)
            
            # Get prediction
            prediction = self.qos_predictor.predict()
            
            if prediction:
                # Log prediction
                logger.info(f"Predicted throughput: {prediction['throughput_down']*10000000:.2f} bps, "
                           f"RTT: {prediction['rtt']*100:.2f} ms, "
                           f"Packet loss: {prediction['packet_loss']*100:.2f}%")
                
                # Update video player buffer
                self.video_player.update_buffer(
                    downloaded_bytes=measurement['throughput_down'] / 8,  # Convert bits to bytes
                    network_throughput=measurement['throughput_down']
                )
                
                # Adapt quality based on prediction
                self._adapt_quality(prediction)
            
            # Sleep for 1 second
            time.sleep(1.0)
    
    def _simulate_network_measurement(self):
        """
        Simulate network measurement
        
        In a real implementation, this would be replaced with actual network monitoring
        
        Returns:
            Dictionary with simulated network metrics
        """
        # Base values
        base_throughput = 3000000  # 3 Mbps
        base_rtt = 50.0  # 50 ms
        base_packet_loss = 0.5  # 0.5%
        
        # Add some random variation
        throughput = max(100000, base_throughput + np.random.normal(0, 500000))
        rtt = max(10.0, base_rtt + np.random.normal(0, 10.0))
        packet_loss = max(0.0, base_packet_loss + np.random.normal(0, 0.2))
        
        # Every 30 seconds, simulate a network degradation event
        if int(time.time()) % 30 < 5:
            throughput *= 0.3
            rtt *= 2.0
            packet_loss *= 3.0
        
        # Normalize values (as the model expects normalized inputs)
        normalized_throughput = throughput / 10000000
        normalized_rtt = rtt / 100
        normalized_packet_loss = packet_loss / 100
        
        # Create measurement dictionary
        measurement = {
            'throughput_down': normalized_throughput,
            'throughput_up': normalized_throughput * 0.2,  # Upload is typically lower
            'rtt': normalized_rtt,
            'jitter': normalized_rtt * 0.1,
            'packet_loss': normalized_packet_loss,
            'sinr': 0.7,  # Placeholder values for signal metrics
            'rsrp': 0.6,
            'rsrq': 0.5
        }
        
        return measurement
    
    def _adapt_quality(self, prediction):
        """
        Adapt video quality based on predicted network conditions
        
        Args:
            prediction: Dictionary with predicted network metrics
        """
        # Get current buffer level
        buffer_level = self.video_player.get_buffer_level()
        
        # Get predicted throughput (denormalize)
        predicted_throughput = prediction['throughput_down'] * 10000000  # Scale back to bps
        
        # Determine appropriate quality level based on predicted throughput
        if predicted_throughput < self.throughput_thresholds['low']:
            target_quality = 'low'
        elif predicted_throughput < self.throughput_thresholds['medium']:
            target_quality = 'medium'
        elif predicted_throughput < self.throughput_thresholds['high']:
            target_quality = 'high'
        else:
            target_quality = 'ultra'
        
        # Adjust based on buffer level
        if buffer_level < self.buffer_thresholds['critical']:
            # Critical buffer level, drop to lowest quality to prevent stalling
            target_quality = 'low'
        elif buffer_level < self.buffer_thresholds['low'] and target_quality != 'low':
            # Low buffer, drop one quality level
            qualities = list(self.video_player.QUALITY_LEVELS.keys())
            current_index = qualities.index(self.video_player.current_quality)
            target_quality = qualities[max(0, current_index - 1)]
        
        # Apply quality change if different from current
        if target_quality != self.video_player.current_quality:
            self.video_player.set_quality(target_quality)

def main():
    """Main function"""
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Look for the most recent model in the saved models directory
        models_dir = Path(__file__).resolve().parent.parent / "models" / "saved"
        model_files = list(models_dir.glob("*lstm*.h5"))
        if not model_files:
            logger.error("No model found. Please train a model first.")
            return
        
        # Use the most recent model
        model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
    
    logger.info(f"Using model: {model_path}")
    
    # Create video player
    video_player = VideoPlayer()
    
    # Create QoS predictor
    qos_predictor = QoSPredictor(model_path)
    
    # Create ABR controller
    abr_controller = ABRController(video_player, qos_predictor)
    
    try:
        # Start video playback
        video_player.start_playback()
        
        # Start ABR controller
        abr_controller.start()
        
        # Run for 60 seconds
        logger.info("Running for 60 seconds...")
        time.sleep(60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Stop ABR controller
        abr_controller.stop()
        
        # Stop video playback
        video_player.stop_playback()
        
        logger.info("Done")

if __name__ == "__main__":
    main()