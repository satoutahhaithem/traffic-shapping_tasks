#!/usr/bin/env python3

"""
Script to generate sample live data for demonstration purposes.
This script simulates changing network conditions and metrics over time.
"""

import os
import json
import time
import random
from datetime import datetime, timedelta

# Configuration
RESULTS_DIR = "test_results"
LIVE_DATA_FILE = os.path.join(RESULTS_DIR, "live_data.json")

# Network conditions to cycle through
NETWORK_CONDITIONS = [
    {"name": "Excellent", "rate": "10mbit", "delay": "20ms", "loss": "0%"},
    {"name": "Good", "rate": "6mbit", "delay": "40ms", "loss": "0.5%"},
    {"name": "Fair", "rate": "4mbit", "delay": "80ms", "loss": "1%"},
    {"name": "Poor", "rate": "2mbit", "delay": "150ms", "loss": "3%"}
]

# Quality parameters to cycle through
RESOLUTION_SCALES = [1.0, 0.9, 0.75, 0.5]
JPEG_QUALITIES = [95, 85, 75, 60]
FRAME_RATES = [30, 20, 15, 10]

def create_sample_data_point(timestamp, network_index, quality_index):
    """Create a sample data point with the given parameters."""
    network = NETWORK_CONDITIONS[network_index]
    resolution_scale = RESOLUTION_SCALES[quality_index]
    jpeg_quality = JPEG_QUALITIES[quality_index]
    frame_rate = FRAME_RATES[quality_index]
    
    # Calculate sample metrics based on network and quality parameters
    # Bandwidth usage (higher with higher resolution, quality, and fps)
    base_bandwidth = 5000000  # 5 Mbps base for full quality
    resolution_factor = {0.5: 0.25, 0.75: 0.5625, 0.9: 0.81, 1.0: 1.0}
    quality_factor = {60: 0.4, 75: 0.6, 85: 0.8, 95: 1.0}
    fps_factor = {10: 0.33, 15: 0.5, 20: 0.67, 30: 1.0}
    
    bandwidth = (base_bandwidth * 
                resolution_factor.get(resolution_scale, 1.0) * 
                quality_factor.get(jpeg_quality, 1.0) * 
                fps_factor.get(frame_rate, 1.0))
    
    # Add some random variation
    bandwidth *= random.uniform(0.9, 1.1)
    
    # Frame delivery time (higher with higher delay and bandwidth)
    base_delay = float(network["delay"].replace("ms", ""))
    delivery_time = base_delay * (1 + (bandwidth / 10000000) * 0.5)  # Delay increases with bandwidth
    delivery_time *= random.uniform(0.9, 1.1)  # Add some random variation
    
    # Frame drop rate (higher with higher loss and bandwidth)
    base_loss = float(network["loss"].replace("%", ""))
    drop_rate = base_loss * (1 + (bandwidth / 10000000) * 0.5)  # Loss increases with bandwidth
    drop_rate *= random.uniform(0.9, 1.1)  # Add some random variation
    
    # Visual quality score
    resolution_score = resolution_scale * 100
    quality_score = jpeg_quality
    visual_quality = (resolution_score * 0.6) + (quality_score * 0.4)
    
    # Smoothness score
    fps_score = {10: 30, 15: 50, 20: 70, 30: 100}
    network_score = {"Poor": 30, "Fair": 60, "Good": 80, "Excellent": 100}
    smoothness = (fps_score.get(frame_rate, 50) * 0.7) + (network_score.get(network["name"], 50) * 0.3)
    
    # Create the data point
    data_point = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "network_condition": network["name"],
        "network_rate": network["rate"],
        "network_delay": network["delay"],
        "network_loss": network["loss"],
        "resolution_scale": resolution_scale,
        "jpeg_quality": jpeg_quality,
        "frame_rate": frame_rate,
        "metrics": {
            "bandwidth_usage": bandwidth,
            "frame_delivery_time": delivery_time,
            "frame_drop_rate": drop_rate,
            "visual_quality_score": visual_quality,
            "smoothness_score": smoothness
        }
    }
    
    return data_point

def generate_sample_live_data(num_points=30, interval_seconds=10):
    """Generate sample live data points over time."""
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Start with an empty list
    live_data = []
    
    # Generate data points
    start_time = datetime.now() - timedelta(seconds=interval_seconds * num_points)
    
    for i in range(num_points):
        # Calculate timestamp
        timestamp = start_time + timedelta(seconds=interval_seconds * i)
        
        # Cycle through network conditions (change every 5 points)
        network_index = (i // 5) % len(NETWORK_CONDITIONS)
        
        # Cycle through quality parameters (change every 3 points)
        quality_index = (i // 3) % len(RESOLUTION_SCALES)
        
        # Create data point
        data_point = create_sample_data_point(timestamp, network_index, quality_index)
        
        # Add to list
        live_data.append(data_point)
        
        print(f"Generated data point {i+1}/{num_points}: {data_point['timestamp']} - {data_point['network_condition']}")
    
    # Save to file
    with open(LIVE_DATA_FILE, 'w') as f:
        json.dump(live_data, f, indent=2)
    
    print(f"Saved {num_points} sample data points to {LIVE_DATA_FILE}")

def main():
    """Main function."""
    print("Generating Sample Live Data")
    print("==========================")
    
    # Generate sample data
    generate_sample_live_data()
    
    print("\nSample data generation complete.")
    print("To view the data, run: python3 dynamic_quality_testing/live_monitor.py")

if __name__ == "__main__":
    main()