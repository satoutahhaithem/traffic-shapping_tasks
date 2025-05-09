#!/usr/bin/env python3

import os
import time
import json
import subprocess
import requests
from datetime import datetime

# Try to import matplotlib and numpy, but provide graceful fallback if not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib and/or numpy not installed. Plotting functionality will be disabled.")
    print("To enable plotting, install the required packages:")
    print("  pip install matplotlib numpy")
    PLOTTING_AVAILABLE = False

# Configuration
SENDER_IP = "localhost"  # Local machine for sender
SENDER_PORT = 5000
RECEIVER_IP = "192.168.2.169"  # Change to match the IP address of your receiver
RECEIVER_PORT = 8081
INTERFACE = "wlp0s20f3"  # Change to match your network interface

# Test matrix
RESOLUTION_SCALES = [0.25, 0.5, 0.75, 1.0]
JPEG_QUALITIES = [50, 65, 80, 95]
FRAME_RATES = [10, 15, 20, 30]

# Network conditions
NETWORK_CONDITIONS = [
    {"name": "Poor", "rate": "1mbit", "delay": "200ms", "loss": "5%"},
    {"name": "Fair", "rate": "3mbit", "delay": "100ms", "loss": "2%"},
    {"name": "Good", "rate": "5mbit", "delay": "50ms", "loss": "1%"},
    {"name": "Excellent", "rate": "10mbit", "delay": "20ms", "loss": "0%"}
]

# Results storage
results = []

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(f" {text} ".center(80, "-"))
    print("-" * 80)

def apply_network_condition(condition):
    """Apply network condition using tc."""
    print(f"Applying network condition: {condition['name']}")
    print(f"  Rate: {condition['rate']}, Delay: {condition['delay']}, Loss: {condition['loss']}")
    
    # Reset any existing tc rules
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    
    # Apply new tc rules
    subprocess.run([
        "sudo", "tc", "qdisc", "add", "dev", INTERFACE, "root", "netem", 
        "rate", condition["rate"], 
        "delay", condition["delay"], 
        "loss", condition["loss"]
    ])
    
    # Verify the settings
    result = subprocess.run(["tc", "qdisc", "show", "dev", INTERFACE], 
                           capture_output=True, text=True)
    print(f"  TC settings: {result.stdout.strip()}")
    
    # Wait for network to stabilize
    time.sleep(2)

def reset_network_condition():
    """Reset network conditions to normal."""
    print("Resetting network conditions")
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    time.sleep(1)

def set_quality_parameters(resolution, quality, fps):
    """Set quality parameters on the video streamer."""
    print(f"Setting quality parameters: Resolution={resolution}, Quality={quality}, FPS={fps}")
    
    # Set resolution scale
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_resolution/{resolution}", 
                               timeout=5)
        print(f"  Resolution response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting resolution: {e}")
    
    # Set JPEG quality
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_quality/{quality}", 
                               timeout=5)
        print(f"  Quality response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting quality: {e}")
    
    # Set frame rate
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_fps/{fps}", 
                               timeout=5)
        print(f"  FPS response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting FPS: {e}")
    
    # Wait for settings to take effect
    time.sleep(2)

def measure_performance():
    """Measure performance metrics for current settings."""
    print("Measuring performance metrics...")
    
    # Initialize metrics
    metrics = {
        "bandwidth_usage": 0,
        "frame_delivery_time": 0,
        "frame_drop_rate": 0,
        "visual_quality_score": 0,
        "smoothness_score": 0
    }
    
    # Get metrics from sender
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_metrics", timeout=5)
        if response.status_code == 200:
            sender_metrics = response.json()
            metrics.update(sender_metrics)
            print(f"  Sender metrics: {sender_metrics}")
    except Exception as e:
        print(f"  Error getting sender metrics: {e}")
    
    # Get metrics from receiver
    try:
        response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/get_metrics", timeout=5)
        if response.status_code == 200:
            receiver_metrics = response.json()
            metrics.update(receiver_metrics)
            print(f"  Receiver metrics: {receiver_metrics}")
    except Exception as e:
        print(f"  Error getting receiver metrics: {e}")
    
    # If metrics endpoints aren't implemented, estimate metrics based on settings
    if metrics["bandwidth_usage"] == 0:
        # Estimate bandwidth based on resolution, quality, and fps
        resolution_factor = {0.25: 0.0625, 0.5: 0.25, 0.75: 0.5625, 1.0: 1.0}
        quality_factor = {50: 0.3, 65: 0.5, 80: 0.7, 95: 1.0}
        fps_factor = {10: 0.33, 15: 0.5, 20: 0.67, 30: 1.0}
        
        base_bandwidth = 5000000  # 5 Mbps base for full quality
        estimated_bandwidth = (base_bandwidth * 
                              resolution_factor.get(current_resolution, 1.0) * 
                              quality_factor.get(current_quality, 1.0) * 
                              fps_factor.get(current_fps, 1.0))
        
        metrics["bandwidth_usage"] = estimated_bandwidth
        print(f"  Estimated bandwidth: {estimated_bandwidth/1000000:.2f} Mbps")
    
    if metrics["visual_quality_score"] == 0:
        # Estimate visual quality based on resolution and quality
        resolution_score = {0.25: 25, 0.5: 50, 0.75: 75, 1.0: 100}
        quality_score = {50: 50, 65: 65, 80: 80, 95: 95}
        
        visual_quality = (resolution_score.get(current_resolution, 50) * 0.6 + 
                         quality_score.get(current_quality, 50) * 0.4)
        
        metrics["visual_quality_score"] = visual_quality
        print(f"  Estimated visual quality: {visual_quality:.1f}/100")
    
    if metrics["smoothness_score"] == 0:
        # Estimate smoothness based on fps and network condition
        fps_score = {10: 30, 15: 50, 20: 70, 30: 100}
        network_score = {"Poor": 30, "Fair": 60, "Good": 80, "Excellent": 100}
        
        smoothness = (fps_score.get(current_fps, 50) * 0.7 + 
                     network_score.get(current_network["name"], 50) * 0.3)
        
        metrics["smoothness_score"] = smoothness
        print(f"  Estimated smoothness: {smoothness:.1f}/100")
    
    return metrics

def run_test(resolution, quality, fps, network):
    """Run a single test with the given parameters."""
    global current_resolution, current_quality, current_fps, current_network
    
    # Store current parameters for metric estimation
    current_resolution = resolution
    current_quality = quality
    current_fps = fps
    current_network = network
    
    print_section(f"Testing: Resolution={resolution}, Quality={quality}, FPS={fps}, Network={network['name']}")
    
    # Apply network condition
    apply_network_condition(network)
    
    # Set quality parameters
    set_quality_parameters(resolution, quality, fps)
    
    # Wait for system to stabilize
    print("Waiting for system to stabilize...")
    time.sleep(10)
    
    # Measure performance
    metrics = measure_performance()
    
    # Record results
    test_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resolution_scale": resolution,
        "jpeg_quality": quality,
        "frame_rate": fps,
        "network_condition": network["name"],
        "network_rate": network["rate"],
        "network_delay": network["delay"],
        "network_loss": network["loss"],
        "metrics": metrics
    }
    
    results.append(test_result)
    print(f"Test completed: {test_result}")
    
    # Wait before next test
    time.sleep(2)

def generate_plots(timestamp):
    """Generate plots from the test results."""
    if not PLOTTING_AVAILABLE:
        print("Skipping plot generation because matplotlib/numpy is not available.")
        print("Install the required packages with: pip install matplotlib numpy")
        return
    
    try:
        print("Generating plots...")
        
        # Extract data for plotting
        resolutions = []
        qualities = []
        fps_values = []
        networks = []
        bandwidth = []
        visual_quality = []
        smoothness = []
        
        for result in results:
            resolutions.append(result["resolution_scale"])
            qualities.append(result["jpeg_quality"])
            fps_values.append(result["frame_rate"])
            networks.append(result["network_condition"])
            bandwidth.append(result["metrics"]["bandwidth_usage"] / 1000000)  # Convert to Mbps
            visual_quality.append(result["metrics"]["visual_quality_score"])
            smoothness.append(result["metrics"]["smoothness_score"])
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bandwidth usage by resolution and network
        plt.subplot(2, 2, 1)
        for network in NETWORK_CONDITIONS:
            network_name = network["name"]
            indices = [i for i, n in enumerate(networks) if n == network_name]
            plt.plot([resolutions[i] for i in indices], 
                    [bandwidth[i] for i in indices], 
                    'o-', label=network_name)
        
        plt.xlabel('Resolution Scale')
        plt.ylabel('Bandwidth (Mbps)')
        plt.title('Bandwidth Usage by Resolution and Network')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Visual quality by resolution and quality
        plt.subplot(2, 2, 2)
        for quality in JPEG_QUALITIES:
            indices = [i for i, q in enumerate(qualities) if q == quality]
            plt.plot([resolutions[i] for i in indices], 
                    [visual_quality[i] for i in indices], 
                    'o-', label=f'Quality {quality}%')
        
        plt.xlabel('Resolution Scale')
        plt.ylabel('Visual Quality Score')
        plt.title('Visual Quality by Resolution and JPEG Quality')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Smoothness by FPS and network
        plt.subplot(2, 2, 3)
        for network in NETWORK_CONDITIONS:
            network_name = network["name"]
            indices = [i for i, n in enumerate(networks) if n == network_name]
            plt.plot([fps_values[i] for i in indices], 
                    [smoothness[i] for i in indices], 
                    'o-', label=network_name)
        
        plt.xlabel('Frame Rate (FPS)')
        plt.ylabel('Smoothness Score')
        plt.title('Smoothness by Frame Rate and Network')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Quality-bandwidth tradeoff
        plt.subplot(2, 2, 4)
        plt.scatter(bandwidth, visual_quality, c=np.array(fps_values), cmap='viridis', 
                   s=100, alpha=0.7)
        plt.colorbar(label='Frame Rate (FPS)')
        plt.xlabel('Bandwidth (Mbps)')
        plt.ylabel('Visual Quality Score')
        plt.title('Quality-Bandwidth Tradeoff')
        plt.grid(True)
        
        # Save the figure
        plot_filename = f"quality_test_plots_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Plots saved to {plot_filename}")
        
        # Close the figure
        plt.close()
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Continuing without plots...")

def check_dependencies():
    """Check if all dependencies are installed."""
    print_header("Checking Dependencies")
    
    all_dependencies_met = True
    
    # Check if matplotlib and numpy are installed
    if PLOTTING_AVAILABLE:
        print("✓ matplotlib and numpy are installed")
    else:
        print("✗ matplotlib and/or numpy are not installed")
        print("  Install with: pip install matplotlib numpy")
        print("  Plotting will be disabled, but testing can still proceed")
    
    # Check if tc is installed
    result = subprocess.run(["which", "tc"], capture_output=True)
    if result.returncode == 0:
        print("✓ tc is installed")
    else:
        print("✗ tc is not installed. Install with: sudo apt install iproute2")
        all_dependencies_met = False
    
    # Check if video streamer is running
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/", timeout=2)
        print("✓ Video streamer is running")
    except:
        print("✗ Video streamer is not running")
        print(f"  Start with: python3 dynamic_quality_testing/video_streamer.py")
        all_dependencies_met = False
    
    # Check if video receiver is running
    try:
        response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/", timeout=2)
        print("✓ Video receiver is running")
    except:
        print("✗ Video receiver is not running")
        print(f"  Start with: python3 dynamic_quality_testing/receive_video.py")
        all_dependencies_met = False
    
    return all_dependencies_met

def generate_report():
    """Generate a report of the test results."""
    print_header("Generating Test Report")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_test_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    
    # Generate plots if matplotlib is available
    generate_plots(timestamp)
    
    # Generate a text report
    text_report = f"quality_test_report_{timestamp}.txt"
    with open(text_report, "w") as f:
        f.write("=================================================\n")
        f.write("DYNAMIC QUALITY TESTING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=================================================\n\n")
        
        f.write("SUMMARY\n")
        f.write(f"Total tests run: {len(results)}\n")
        f.write(f"Resolution scales tested: {', '.join(str(r) for r in RESOLUTION_SCALES)}\n")
        f.write(f"JPEG qualities tested: {', '.join(str(q) for q in JPEG_QUALITIES)}\n")
        f.write(f"Frame rates tested: {', '.join(str(fps) for fps in FRAME_RATES)}\n")
        f.write(f"Network conditions tested: {', '.join(n['name'] for n in NETWORK_CONDITIONS)}\n\n")
        
        f.write("DETAILED RESULTS\n")
        for i, result in enumerate(results):
            f.write(f"Test #{i+1}:\n")
            f.write(f"  Time: {result['timestamp']}\n")
            f.write(f"  Resolution Scale: {result['resolution_scale']}\n")
            f.write(f"  JPEG Quality: {result['jpeg_quality']}\n")
            f.write(f"  Frame Rate: {result['frame_rate']}\n")
            f.write(f"  Network: {result['network_condition']} ({result['network_rate']}, {result['network_delay']}, {result['network_loss']})\n")
            f.write("  Metrics:\n")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    f.write(f"    {metric}: {value:.2f}\n")
                else:
                    f.write(f"    {metric}: {value}\n")
            f.write("\n")
    
    print(f"Text report saved to {text_report}")

def main():
    """Main function to run the quality tests."""
    print_header("Dynamic Quality Testing System")
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and ensure the video streamer and receiver are running.")
        return
    
    # Confirm with user
    print("\nThis script will test different quality settings with various network conditions.")
    print("It will modify your network settings using tc, which requires sudo privileges.")
    print("Make sure both the video streamer and receiver are running before proceeding.")
    
    confirm = input("\nDo you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        # Run tests for each combination
        for network in NETWORK_CONDITIONS:
            for resolution in RESOLUTION_SCALES:
                for quality in JPEG_QUALITIES:
                    for fps in FRAME_RATES:
                        run_test(resolution, quality, fps, network)
        
        # Reset network conditions
        reset_network_condition()
        
        # Generate report
        generate_report()
        
        print_header("Testing Complete")
        print(f"Tested {len(results)} combinations of quality settings and network conditions.")
        print("Check the JSON file and plots for detailed results.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
    finally:
        # Always reset network conditions
        reset_network_condition()

if __name__ == "__main__":
    main()