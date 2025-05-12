#!/usr/bin/env python3

"""
Script to update live data during testing.
This script is called by dynamic_tc_control.sh to update the live data file with new metrics.
"""

import os
import json
import sys
import time
from datetime import datetime

# Configuration
RESULTS_DIR = "test_results"
LIVE_DATA_FILE = os.path.join(RESULTS_DIR, "live_data.json")

def update_live_data(network_condition, network_rate, network_delay, network_loss, 
                    resolution_scale, jpeg_quality, frame_rate, 
                    bandwidth_usage, frame_delivery_time, frame_drop_rate, 
                    visual_quality_score, smoothness_score):
    """Update the live data file with new metrics."""
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load existing data
    live_data = []
    if os.path.exists(LIVE_DATA_FILE):
        try:
            with open(LIVE_DATA_FILE, 'r') as f:
                live_data = json.load(f)
        except json.JSONDecodeError:
            # If the file is corrupted, start with an empty list
            live_data = []
    
    # Create a new data point
    data_point = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "network_condition": network_condition,
        "network_rate": network_rate,
        "network_delay": network_delay,
        "network_loss": network_loss,
        "resolution_scale": float(resolution_scale),
        "jpeg_quality": int(jpeg_quality),
        "frame_rate": int(frame_rate),
        "metrics": {
            "bandwidth_usage": float(bandwidth_usage),
            "frame_delivery_time": float(frame_delivery_time),
            "frame_drop_rate": float(frame_drop_rate),
            "visual_quality_score": float(visual_quality_score),
            "smoothness_score": float(smoothness_score)
        }
    }
    
    # Add the new data point
    live_data.append(data_point)
    
    # Save the updated data
    with open(LIVE_DATA_FILE, 'w') as f:
        json.dump(live_data, f, indent=2)
    
    print(f"Updated live data file with new metrics at {data_point['timestamp']}")

def main():
    """Main function."""
    if len(sys.argv) < 13:
        print("Usage: update_live_data.py <network_condition> <network_rate> <network_delay> <network_loss> "
              "<resolution_scale> <jpeg_quality> <frame_rate> "
              "<bandwidth_usage> <frame_delivery_time> <frame_drop_rate> "
              "<visual_quality_score> <smoothness_score>")
        return
    
    update_live_data(
        sys.argv[1],   # network_condition
        sys.argv[2],   # network_rate
        sys.argv[3],   # network_delay
        sys.argv[4],   # network_loss
        sys.argv[5],   # resolution_scale
        sys.argv[6],   # jpeg_quality
        sys.argv[7],   # frame_rate
        sys.argv[8],   # bandwidth_usage
        sys.argv[9],   # frame_delivery_time
        sys.argv[10],  # frame_drop_rate
        sys.argv[11],  # visual_quality_score
        sys.argv[12]   # smoothness_score
    )

if __name__ == "__main__":
    main()