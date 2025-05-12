#!/usr/bin/env python3

"""
Generate graphs to visualize the relationship between controlled network parameters
(bandwidth, delay, packet loss) and measured performance metrics.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime

# Results directory
RESULTS_DIR = "test_results"

def format_bandwidth(x, pos):
    """Format bandwidth in Mbps."""
    return f"{x/1000000:.1f}"

def extract_rate_value(rate_str):
    """Extract numeric value from rate string (e.g., '2mbit' -> 2000000)."""
    if not rate_str:
        return 0
    
    value = float(rate_str.replace('mbit', '').replace('kbit', '').strip())
    if 'mbit' in rate_str:
        return value * 1000000
    elif 'kbit' in rate_str:
        return value * 1000
    return value

def extract_delay_value(delay_str):
    """Extract numeric value from delay string (e.g., '150ms' -> 150)."""
    if not delay_str:
        return 0
    
    value = float(delay_str.replace('ms', '').replace('s', '').strip())
    if 's' in delay_str and 'ms' not in delay_str:
        return value * 1000
    return value

def extract_loss_value(loss_str):
    """Extract numeric value from loss string (e.g., '3%' -> 3)."""
    if not loss_str:
        return 0
    
    return float(loss_str.replace('%', '').strip())

def load_test_results():
    """Load all test results from JSON files."""
    results = []
    
    # Find all JSON result files
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def prepare_data(results):
    """Prepare data for graphing."""
    data = {
        'bandwidth': {
            'controlled': [],
            'measured': [],
            'resolution': [],
            'quality': [],
            'fps': []
        },
        'delay': {
            'controlled': [],
            'measured': [],
            'resolution': [],
            'quality': [],
            'fps': []
        },
        'loss': {
            'controlled': [],
            'measured': [],
            'resolution': [],
            'quality': [],
            'fps': []
        }
    }
    
    for result in results:
        # Skip results without metrics
        if 'metrics' not in result:
            continue
        
        # Extract controlled parameters
        controlled_bandwidth = extract_rate_value(result.get('network_rate', ''))
        controlled_delay = extract_delay_value(result.get('network_delay', ''))
        controlled_loss = extract_loss_value(result.get('network_loss', ''))
        
        # Extract measured metrics
        measured_bandwidth = result['metrics'].get('bandwidth_usage', 0)
        measured_delay = result['metrics'].get('frame_delivery_time', 0)
        measured_loss = result['metrics'].get('frame_drop_rate', 0)
        
        # Extract quality parameters
        resolution = result.get('resolution_scale', 0)
        quality = result.get('jpeg_quality', 0)
        fps = result.get('frame_rate', 0)
        
        # Add to data
        data['bandwidth']['controlled'].append(controlled_bandwidth)
        data['bandwidth']['measured'].append(measured_bandwidth)
        data['bandwidth']['resolution'].append(resolution)
        data['bandwidth']['quality'].append(quality)
        data['bandwidth']['fps'].append(fps)
        
        data['delay']['controlled'].append(controlled_delay)
        data['delay']['measured'].append(measured_delay)
        data['delay']['resolution'].append(resolution)
        data['delay']['quality'].append(quality)
        data['delay']['fps'].append(fps)
        
        data['loss']['controlled'].append(controlled_loss)
        data['loss']['measured'].append(measured_loss)
        data['loss']['resolution'].append(resolution)
        data['loss']['quality'].append(quality)
        data['loss']['fps'].append(fps)
    
    return data

def generate_bandwidth_graphs(data):
    """Generate graphs for bandwidth."""
    plt.figure(figsize=(12, 8))
    
    # Controlled vs Measured Bandwidth
    ax1 = plt.subplot(2, 2, 1)
    
    # Sort data points for line plot
    sorted_indices = np.argsort(data['bandwidth']['controlled'])
    sorted_controlled = np.array(data['bandwidth']['controlled'])[sorted_indices]
    sorted_measured = np.array(data['bandwidth']['measured'])[sorted_indices]
    
    # Plot line with markers
    plt.plot(sorted_controlled, sorted_measured, 'o-', linewidth=2, markersize=6, alpha=0.7, label='Actual Measurement')
    plt.plot([0, max(data['bandwidth']['controlled'])], [0, max(data['bandwidth']['controlled'])], 'r--', label='Ideal 1:1 Relationship')
    plt.legend()
    
    ax1.xaxis.set_major_formatter(FuncFormatter(format_bandwidth))
    ax1.yaxis.set_major_formatter(FuncFormatter(format_bandwidth))
    plt.xlabel('Controlled Bandwidth (Mbps)')
    plt.ylabel('Measured Bandwidth (Mbps)')
    plt.title('Controlled vs Measured Bandwidth')
    plt.grid(True)
    
    # Bandwidth vs Resolution
    ax2 = plt.subplot(2, 2, 2)
    
    # Group by resolution and calculate average bandwidth
    resolutions = sorted(set(data['bandwidth']['resolution']))
    avg_bandwidths = []
    
    for res in resolutions:
        indices = [i for i, r in enumerate(data['bandwidth']['resolution']) if r == res]
        avg_bw = np.mean([data['bandwidth']['measured'][i] for i in indices])
        avg_bandwidths.append(avg_bw)
    
    # Plot line with markers
    plt.plot(resolutions, avg_bandwidths, 'o-', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.yaxis.set_major_formatter(FuncFormatter(format_bandwidth))
    plt.xlabel('Resolution Scale')
    plt.ylabel('Measured Bandwidth (Mbps)')
    plt.title('Resolution Scale vs Bandwidth')
    plt.grid(True)
    
    # Bandwidth vs Quality
    ax3 = plt.subplot(2, 2, 3)
    
    # Group by quality and calculate average bandwidth
    qualities = sorted(set(data['bandwidth']['quality']))
    avg_bandwidths = []
    
    for quality in qualities:
        indices = [i for i, q in enumerate(data['bandwidth']['quality']) if q == quality]
        avg_bw = np.mean([data['bandwidth']['measured'][i] for i in indices])
        avg_bandwidths.append(avg_bw)
    
    # Plot line with markers
    plt.plot(qualities, avg_bandwidths, 'o-', linewidth=2, markersize=6, alpha=0.7)
    
    ax3.yaxis.set_major_formatter(FuncFormatter(format_bandwidth))
    plt.xlabel('JPEG Quality')
    plt.ylabel('Measured Bandwidth (Mbps)')
    plt.title('JPEG Quality vs Bandwidth')
    plt.grid(True)
    
    # Bandwidth vs FPS
    ax4 = plt.subplot(2, 2, 4)
    
    # Group by FPS and calculate average bandwidth
    fps_values = sorted(set(data['bandwidth']['fps']))
    avg_bandwidths = []
    
    for fps in fps_values:
        indices = [i for i, f in enumerate(data['bandwidth']['fps']) if f == fps]
        avg_bw = np.mean([data['bandwidth']['measured'][i] for i in indices])
        avg_bandwidths.append(avg_bw)
    
    # Plot line with markers
    plt.plot(fps_values, avg_bandwidths, 'o-', linewidth=2, markersize=6, alpha=0.7)
    
    ax4.yaxis.set_major_formatter(FuncFormatter(format_bandwidth))
    plt.xlabel('Frame Rate (FPS)')
    plt.ylabel('Measured Bandwidth (Mbps)')
    plt.title('Frame Rate vs Bandwidth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bandwidth_graphs.png'))
    print(f"Saved bandwidth graphs to {os.path.join(RESULTS_DIR, 'bandwidth_graphs.png')}")

def generate_delay_graphs(data):
    """Generate graphs for delay."""
    plt.figure(figsize=(12, 8))
    
    # Controlled vs Measured Delay
    plt.subplot(2, 2, 1)
    
    # Sort data points for line plot
    sorted_indices = np.argsort(data['delay']['controlled'])
    sorted_controlled = np.array(data['delay']['controlled'])[sorted_indices]
    sorted_measured = np.array(data['delay']['measured'])[sorted_indices]
    
    # Plot line with markers
    plt.plot(sorted_controlled, sorted_measured, 'o-', linewidth=2, markersize=6, alpha=0.7, label='Actual Measurement')
    plt.plot([0, max(data['delay']['controlled'])], [0, max(data['delay']['measured'])], 'r--', label='Ideal 1:1 Relationship')
    plt.legend()
    
    plt.xlabel('Controlled Delay (ms)')
    plt.ylabel('Measured Frame Delivery Time (ms)')
    plt.title('Controlled Delay vs Measured Frame Delivery Time')
    plt.grid(True)
    
    # Delay vs Resolution
    plt.subplot(2, 2, 2)
    for delay in sorted(set(data['delay']['controlled'])):
        indices = [i for i, x in enumerate(data['delay']['controlled']) if x == delay]
        
        # Get resolution and measured delay values
        res_delay_pairs = [(data['delay']['resolution'][i], data['delay']['measured'][i]) for i in indices]
        # Sort by resolution
        res_delay_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        resolutions = [pair[0] for pair in res_delay_pairs]
        measured_delays = [pair[1] for pair in res_delay_pairs]
        
        # Plot line with markers
        plt.plot(resolutions, measured_delays, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{delay}ms')
    
    plt.xlabel('Resolution Scale')
    plt.ylabel('Measured Frame Delivery Time (ms)')
    plt.title('Resolution Scale vs Frame Delivery Time')
    plt.legend()
    plt.grid(True)
    
    # Delay vs Quality
    plt.subplot(2, 2, 3)
    for delay in sorted(set(data['delay']['controlled'])):
        indices = [i for i, x in enumerate(data['delay']['controlled']) if x == delay]
        
        # Get quality and measured delay values
        quality_delay_pairs = [(data['delay']['quality'][i], data['delay']['measured'][i]) for i in indices]
        # Sort by quality
        quality_delay_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        qualities = [pair[0] for pair in quality_delay_pairs]
        measured_delays = [pair[1] for pair in quality_delay_pairs]
        
        # Plot line with markers
        plt.plot(qualities, measured_delays, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{delay}ms')
    
    plt.xlabel('JPEG Quality')
    plt.ylabel('Measured Frame Delivery Time (ms)')
    plt.title('JPEG Quality vs Frame Delivery Time')
    plt.legend()
    plt.grid(True)
    
    # Delay vs FPS
    plt.subplot(2, 2, 4)
    for delay in sorted(set(data['delay']['controlled'])):
        indices = [i for i, x in enumerate(data['delay']['controlled']) if x == delay]
        
        # Get FPS and measured delay values
        fps_delay_pairs = [(data['delay']['fps'][i], data['delay']['measured'][i]) for i in indices]
        # Sort by FPS
        fps_delay_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        fps_values = [pair[0] for pair in fps_delay_pairs]
        measured_delays = [pair[1] for pair in fps_delay_pairs]
        
        # Plot line with markers
        plt.plot(fps_values, measured_delays, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{delay}ms')
    
    plt.xlabel('Frame Rate (FPS)')
    plt.ylabel('Measured Frame Delivery Time (ms)')
    plt.title('Frame Rate vs Frame Delivery Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'delay_graphs.png'))
    print(f"Saved delay graphs to {os.path.join(RESULTS_DIR, 'delay_graphs.png')}")

def generate_loss_graphs(data):
    """Generate graphs for packet loss."""
    plt.figure(figsize=(12, 8))
    
    # Controlled vs Measured Loss
    plt.subplot(2, 2, 1)
    
    # Sort data points for line plot
    sorted_indices = np.argsort(data['loss']['controlled'])
    sorted_controlled = np.array(data['loss']['controlled'])[sorted_indices]
    sorted_measured = np.array(data['loss']['measured'])[sorted_indices]
    
    # Plot line with markers
    plt.plot(sorted_controlled, sorted_measured, 'o-', linewidth=2, markersize=6, alpha=0.7, label='Actual Measurement')
    plt.plot([0, max(max(data['loss']['controlled']), 0.1)], [0, max(max(data['loss']['measured']), 0.1)], 'r--', label='Ideal 1:1 Relationship')
    plt.legend()
    
    plt.xlabel('Controlled Packet Loss (%)')
    plt.ylabel('Measured Frame Drop Rate (%)')
    plt.title('Controlled Packet Loss vs Measured Frame Drop Rate')
    plt.grid(True)
    
    # Loss vs Resolution
    plt.subplot(2, 2, 2)
    for loss in sorted(set(data['loss']['controlled'])):
        indices = [i for i, x in enumerate(data['loss']['controlled']) if x == loss]
        
        # Get resolution and measured loss values
        res_loss_pairs = [(data['loss']['resolution'][i], data['loss']['measured'][i]) for i in indices]
        # Sort by resolution
        res_loss_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        resolutions = [pair[0] for pair in res_loss_pairs]
        measured_losses = [pair[1] for pair in res_loss_pairs]
        
        # Plot line with markers
        plt.plot(resolutions, measured_losses, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{loss}%')
    
    plt.xlabel('Resolution Scale')
    plt.ylabel('Measured Frame Drop Rate (%)')
    plt.title('Resolution Scale vs Frame Drop Rate')
    plt.legend()
    plt.grid(True)
    
    # Loss vs Quality
    plt.subplot(2, 2, 3)
    for loss in sorted(set(data['loss']['controlled'])):
        indices = [i for i, x in enumerate(data['loss']['controlled']) if x == loss]
        
        # Get quality and measured loss values
        quality_loss_pairs = [(data['loss']['quality'][i], data['loss']['measured'][i]) for i in indices]
        # Sort by quality
        quality_loss_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        qualities = [pair[0] for pair in quality_loss_pairs]
        measured_losses = [pair[1] for pair in quality_loss_pairs]
        
        # Plot line with markers
        plt.plot(qualities, measured_losses, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{loss}%')
    
    plt.xlabel('JPEG Quality')
    plt.ylabel('Measured Frame Drop Rate (%)')
    plt.title('JPEG Quality vs Frame Drop Rate')
    plt.legend()
    plt.grid(True)
    
    # Loss vs FPS
    plt.subplot(2, 2, 4)
    for loss in sorted(set(data['loss']['controlled'])):
        indices = [i for i, x in enumerate(data['loss']['controlled']) if x == loss]
        
        # Get FPS and measured loss values
        fps_loss_pairs = [(data['loss']['fps'][i], data['loss']['measured'][i]) for i in indices]
        # Sort by FPS
        fps_loss_pairs.sort(key=lambda x: x[0])
        
        # Extract sorted values
        fps_values = [pair[0] for pair in fps_loss_pairs]
        measured_losses = [pair[1] for pair in fps_loss_pairs]
        
        # Plot line with markers
        plt.plot(fps_values, measured_losses, 'o-', linewidth=2, markersize=6, alpha=0.7, label=f'{loss}%')
    
    plt.xlabel('Frame Rate (FPS)')
    plt.ylabel('Measured Frame Drop Rate (%)')
    plt.title('Frame Rate vs Frame Drop Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_graphs.png'))
    print(f"Saved loss graphs to {os.path.join(RESULTS_DIR, 'loss_graphs.png')}")

def generate_combined_graph(data):
    """Generate a combined graph showing the effect of all controlled parameters."""
    plt.figure(figsize=(15, 10))
    
    # Create a figure with multiple subplots for better visualization
    fig = plt.figure(figsize=(15, 10))
    
    # First subplot: 3D Surface plot of Bandwidth vs Delay vs Frame Delivery Time
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Get unique bandwidth and delay values
    bandwidths = sorted(set(data['bandwidth']['controlled']))
    delays = sorted(set(data['delay']['controlled']))
    
    # Create a grid of bandwidth and delay values
    X, Y = np.meshgrid(bandwidths, delays)
    Z = np.zeros_like(X)
    
    # Fill in the grid with frame delivery time values
    for i, delay in enumerate(delays):
        for j, bw in enumerate(bandwidths):
            # Find indices where both bandwidth and delay match
            indices = [idx for idx in range(len(data['bandwidth']['controlled']))
                      if data['bandwidth']['controlled'][idx] == bw and data['delay']['controlled'][idx] == delay]
            
            if indices:
                # Average the frame delivery times for these conditions
                Z[i, j] = np.mean([data['delay']['measured'][idx] for idx in indices])
    
    # Create a surface plot with a colormap
    surf = ax1.plot_surface(X/1000000, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add labels and title
    ax1.set_xlabel('Bandwidth (Mbps)')
    ax1.set_ylabel('Delay (ms)')
    ax1.set_zlabel('Frame Delivery Time (ms)')
    ax1.set_title('Bandwidth vs Delay vs Frame Delivery Time')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Frame Delivery Time (ms)')
    
    # Second subplot: 3D Surface plot of Bandwidth vs Packet Loss vs Frame Drop Rate
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Get unique bandwidth and loss values
    bandwidths = sorted(set(data['bandwidth']['controlled']))
    losses = sorted(set(data['loss']['controlled']))
    
    # Create a grid of bandwidth and loss values
    X, Y = np.meshgrid(bandwidths, losses)
    Z = np.zeros_like(X)
    
    # Fill in the grid with frame drop rate values
    for i, loss in enumerate(losses):
        for j, bw in enumerate(bandwidths):
            # Find indices where both bandwidth and loss match
            indices = [idx for idx in range(len(data['bandwidth']['controlled']))
                      if data['bandwidth']['controlled'][idx] == bw and data['loss']['controlled'][idx] == loss]
            
            if indices:
                # Average the frame drop rates for these conditions
                Z[i, j] = np.mean([data['loss']['measured'][idx] for idx in indices])
    
    # Create a surface plot with a colormap
    surf = ax2.plot_surface(X/1000000, Y, Z, cmap='plasma', edgecolor='none', alpha=0.8)
    
    # Add labels and title
    ax2.set_xlabel('Bandwidth (Mbps)')
    ax2.set_ylabel('Packet Loss (%)')
    ax2.set_zlabel('Frame Drop Rate (%)')
    ax2.set_title('Bandwidth vs Packet Loss vs Frame Drop Rate')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Frame Drop Rate (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'combined_graph.png'))
    print(f"Saved combined graph to {os.path.join(RESULTS_DIR, 'combined_graph.png')}")

def generate_time_series_graphs(data):
    """Generate time-series graphs showing how metrics change over time."""
    plt.figure(figsize=(15, 10))
    
    # Extract timestamps and convert to datetime objects
    timestamps = []
    for result in data:
        if 'timestamp' in result:
            try:
                timestamp = datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S")
                timestamps.append(timestamp)
            except ValueError:
                # Skip invalid timestamps
                timestamps.append(None)
        else:
            timestamps.append(None)
    
    # Skip time series graphs if no valid timestamps
    if not any(timestamps):
        print("No valid timestamps found. Skipping time series graphs.")
        return
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract metrics for each timestamp
    bandwidth_usage = []
    frame_delivery_time = []
    frame_drop_rate = []
    visual_quality = []
    smoothness = []
    
    for result in data:
        if 'metrics' in result:
            bandwidth_usage.append(result['metrics'].get('bandwidth_usage', 0) / 1000000)  # Convert to Mbps
            frame_delivery_time.append(result['metrics'].get('frame_delivery_time', 0))
            frame_drop_rate.append(result['metrics'].get('frame_drop_rate', 0))
            visual_quality.append(result['metrics'].get('visual_quality_score', 0))
            smoothness.append(result['metrics'].get('smoothness_score', 0))
        else:
            bandwidth_usage.append(0)
            frame_delivery_time.append(0)
            frame_drop_rate.append(0)
            visual_quality.append(0)
            smoothness.append(0)
    
    # Filter out None timestamps and corresponding metrics
    valid_data = [(t, b, f, d, v, s) for t, b, f, d, v, s in zip(timestamps, bandwidth_usage, frame_delivery_time, frame_drop_rate, visual_quality, smoothness) if t is not None]
    
    if not valid_data:
        print("No valid data found. Skipping time series graphs.")
        return
    
    # Sort by timestamp
    valid_data.sort(key=lambda x: x[0])
    
    # Extract sorted data
    sorted_timestamps = [d[0] for d in valid_data]
    sorted_bandwidth = [d[1] for d in valid_data]
    sorted_delivery_time = [d[2] for d in valid_data]
    sorted_drop_rate = [d[3] for d in valid_data]
    sorted_visual_quality = [d[4] for d in valid_data]
    sorted_smoothness = [d[5] for d in valid_data]
    
    # Plot Bandwidth Usage over time
    axs[0, 0].plot(sorted_timestamps, sorted_bandwidth, 'o-', linewidth=2, markersize=6, color='blue')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Bandwidth Usage (Mbps)')
    axs[0, 0].set_title('Bandwidth Usage Over Time')
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot Frame Delivery Time over time
    axs[0, 1].plot(sorted_timestamps, sorted_delivery_time, 'o-', linewidth=2, markersize=6, color='red')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Frame Delivery Time (ms)')
    axs[0, 1].set_title('Frame Delivery Time Over Time')
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot Frame Drop Rate over time
    axs[1, 0].plot(sorted_timestamps, sorted_drop_rate, 'o-', linewidth=2, markersize=6, color='orange')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frame Drop Rate (%)')
    axs[1, 0].set_title('Frame Drop Rate Over Time')
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot Visual Quality and Smoothness over time
    ax1 = axs[1, 1]
    ax1.plot(sorted_timestamps, sorted_visual_quality, 'o-', linewidth=2, markersize=6, color='green', label='Visual Quality')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Visual Quality Score')
    ax1.set_title('Quality Scores Over Time')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(sorted_timestamps, sorted_smoothness, 'o-', linewidth=2, markersize=6, color='purple', label='Smoothness')
    ax2.set_ylabel('Smoothness Score')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'time_series_graphs.png'))
    print(f"Saved time series graphs to {os.path.join(RESULTS_DIR, 'time_series_graphs.png')}")

def create_sample_results():
    """Create sample test results for demonstration purposes."""
    print("Creating sample test results for demonstration...")
    
    # Sample network conditions
    network_conditions = [
        {"name": "Poor", "rate": "2mbit", "delay": "150ms", "loss": "3%"},
        {"name": "Fair", "rate": "4mbit", "delay": "80ms", "loss": "1%"},
        {"name": "Good", "rate": "6mbit", "delay": "40ms", "loss": "0.5%"},
        {"name": "Excellent", "rate": "10mbit", "delay": "20ms", "loss": "0%"}
    ]
    
    # Sample quality parameters
    resolution_scales = [0.5, 0.75, 0.9, 1.0]
    jpeg_qualities = [60, 75, 85, 95]
    frame_rates = [10, 15, 20, 30]
    
    # Create sample results
    results = []
    
    for network in network_conditions:
        for resolution in resolution_scales:
            for quality in jpeg_qualities:
                for fps in frame_rates:
                    # Create a sample result with varying timestamps
                    # Calculate a timestamp that varies based on the parameters
                    # This creates a sequence of timestamps over a 2-hour period
                    hour = 14 + (network_conditions.index(network) * resolution_scales.index(resolution) *
                                jpeg_qualities.index(quality) * frame_rates.index(fps)) % 2
                    minute = (network_conditions.index(network) * 15 +
                             resolution_scales.index(resolution) * 10 +
                             jpeg_qualities.index(quality) * 5 +
                             frame_rates.index(fps) * 2) % 60
                    second = (frame_rates.index(fps) * 15) % 60
                    
                    timestamp = f"2025-05-09 {hour:02d}:{minute:02d}:{second:02d}"
                    
                    result = {
                        "timestamp": timestamp,
                        "resolution_scale": resolution,
                        "jpeg_quality": quality,
                        "frame_rate": fps,
                        "network_condition": network["name"],
                        "network_rate": network["rate"],
                        "network_delay": network["delay"],
                        "network_loss": network["loss"],
                        "metrics": {}
                    }
                    
                    # Calculate sample metrics
                    # Bandwidth usage (higher with higher resolution, quality, and fps)
                    base_bandwidth = 5000000  # 5 Mbps base for full quality
                    resolution_factor = {0.5: 0.25, 0.75: 0.5625, 0.9: 0.81, 1.0: 1.0}
                    quality_factor = {60: 0.4, 75: 0.6, 85: 0.8, 95: 1.0}
                    fps_factor = {10: 0.33, 15: 0.5, 20: 0.67, 30: 1.0}
                    
                    bandwidth = (base_bandwidth *
                                resolution_factor.get(resolution, 1.0) *
                                quality_factor.get(quality, 1.0) *
                                fps_factor.get(fps, 1.0))
                    
                    # Frame delivery time (higher with higher delay and bandwidth)
                    base_delay = extract_delay_value(network["delay"])
                    delivery_time = base_delay * (1 + (bandwidth / 10000000) * 0.5)  # Delay increases with bandwidth
                    
                    # Frame drop rate (higher with higher loss and bandwidth)
                    base_loss = extract_loss_value(network["loss"])
                    drop_rate = base_loss * (1 + (bandwidth / 10000000) * 0.5)  # Loss increases with bandwidth
                    
                    # Visual quality score
                    resolution_score = resolution * 100
                    quality_score = quality
                    visual_quality = (resolution_score * 0.6) + (quality_score * 0.4)
                    
                    # Smoothness score
                    fps_score = {10: 30, 15: 50, 20: 70, 30: 100}
                    network_score = {"Poor": 30, "Fair": 60, "Good": 80, "Excellent": 100}
                    smoothness = (fps_score.get(fps, 50) * 0.7) + (network_score.get(network["name"], 50) * 0.3)
                    
                    # Add metrics to result
                    result["metrics"] = {
                        "bandwidth_usage": bandwidth,
                        "frame_delivery_time": delivery_time,
                        "frame_drop_rate": drop_rate,
                        "visual_quality_score": visual_quality,
                        "smoothness_score": smoothness
                    }
                    
                    # Add to results
                    results.append(result)
    
    # Save sample results to JSON file
    sample_file = os.path.join(RESULTS_DIR, "sample_test_results.json")
    with open(sample_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Created sample test results with {len(results)} combinations.")
    print(f"Saved to {sample_file}")
    
    return results

def main():
    """Main function."""
    print("Generating graphs from test results...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    # Load test results
    results = load_test_results()
    
    if not results:
        print("No test results found. Creating sample results for demonstration...")
        results = create_sample_results()
    
    print(f"Loaded {len(results)} test results.")
    
    # Prepare data for graphing
    data = prepare_data(results)
    
    # Generate graphs
    generate_bandwidth_graphs(data)
    generate_delay_graphs(data)
    generate_loss_graphs(data)
    generate_combined_graph(data)
    generate_time_series_graphs(results)  # Pass the original results for time series
    
    print("Graph generation complete.")
    print("To view the graphs, check the test_results directory.")

if __name__ == "__main__":
    main()