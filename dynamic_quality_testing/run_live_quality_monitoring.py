#!/usr/bin/env python3

"""
Comprehensive script to run the entire dynamic quality testing system with live monitoring.
This script:
1. Starts the video streamer (if not already running)
2. Starts the video receiver (if not already running)
3. Starts the live monitoring dashboard
4. Allows changing network conditions using tc
5. Updates live data in real-time
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
import webbrowser
from datetime import datetime
import argparse

# Configuration
SENDER_IP = "localhost"  # Local machine for sender
SENDER_PORT = 5000
RECEIVER_IP = "192.168.2.169"  # Change to match the IP address of your receiver
RECEIVER_PORT = 8081
INTERFACE = "wlp0s20f3"  # Change to match your network interface
RESULTS_DIR = "test_results"
LIVE_DATA_FILE = os.path.join(RESULTS_DIR, "live_data.json")
MONITOR_PORT = 8090

# Network conditions presets
NETWORK_CONDITIONS = [
    {"name": "Excellent", "rate": "10mbit", "delay": "20ms", "loss": "0%"},
    {"name": "Good", "rate": "6mbit", "delay": "40ms", "loss": "0.5%"},
    {"name": "Fair", "rate": "4mbit", "delay": "80ms", "loss": "1%"},
    {"name": "Poor", "rate": "2mbit", "delay": "150ms", "loss": "3%"},
    {"name": "Custom", "rate": "", "delay": "", "loss": ""}
]

# Global variables
processes = {}
stop_event = threading.Event()
current_network = None

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

def check_dependencies():
    """Check if all dependencies are installed."""
    print_section("Checking Dependencies")
    
    all_dependencies_met = True
    
    # Check if tc is installed
    result = subprocess.run(["which", "tc"], capture_output=True)
    if result.returncode == 0:
        print("✓ tc is installed")
    else:
        print("✗ tc is not installed. Install with: sudo apt install iproute2")
        all_dependencies_met = False
    
    # Check if jq is installed
    result = subprocess.run(["which", "jq"], capture_output=True)
    if result.returncode == 0:
        print("✓ jq is installed")
    else:
        print("✗ jq is not installed. Install with: sudo apt install jq")
        print("  Live data updates will be limited without jq.")
    
    # Check if matplotlib and numpy are installed
    try:
        import matplotlib
        import numpy
        print("✓ matplotlib and numpy are installed")
    except ImportError:
        print("✗ matplotlib and/or numpy are not installed")
        print("  Install with: pip install matplotlib numpy")
        print("  Graphing will be limited without these packages.")
        all_dependencies_met = False
    
    return all_dependencies_met

def start_video_streamer():
    """Start the video streamer if not already running."""
    print_section("Starting Video Streamer")
    
    # Check if already running
    try:
        import requests
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/", timeout=2)
        print("✓ Video streamer is already running")
        return True
    except:
        pass
    
    # Start the video streamer
    try:
        cmd = ["python3", "dynamic_quality_testing/video_streamer.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes["streamer"] = process
        
        # Wait for it to start
        time.sleep(2)
        
        # Check if it's running
        try:
            response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/", timeout=2)
            print("✓ Video streamer started successfully")
            return True
        except:
            print("✗ Failed to start video streamer")
            return False
    except Exception as e:
        print(f"✗ Error starting video streamer: {e}")
        return False

def start_video_receiver():
    """Start the video receiver if not already running."""
    print_section("Starting Video Receiver")
    
    # Check if already running
    try:
        import requests
        response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/", timeout=2)
        print("✓ Video receiver is already running")
        return True
    except:
        pass
    
    # Start the video receiver
    try:
        cmd = ["python3", "dynamic_quality_testing/receive_video.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes["receiver"] = process
        
        # Wait for it to start
        time.sleep(2)
        
        # Check if it's running
        try:
            response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/", timeout=2)
            print("✓ Video receiver started successfully")
            return True
        except:
            print("✗ Failed to start video receiver")
            return False
    except Exception as e:
        print(f"✗ Error starting video receiver: {e}")
        return False

def start_live_monitor():
    """Start the live monitoring dashboard."""
    print_section("Starting Live Monitoring Dashboard")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Create initial live data file if it doesn't exist
    if not os.path.exists(LIVE_DATA_FILE):
        with open(LIVE_DATA_FILE, 'w') as f:
            json.dump([], f)
    
    # Ask the user which monitoring method they prefer
    print("Choose a monitoring method:")
    print("1. Terminal-based monitoring (recommended for real-time updates)")
    print("2. Browser-based monitoring")
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == "1":
        # Start the terminal-based monitor in a new terminal window
        try:
            # Start in a new terminal window
            if os.name == 'nt':  # Windows
                cmd = ["start", "cmd", "/k", "python", "dynamic_quality_testing/terminal_live_monitor.py"]
                subprocess.Popen(cmd, shell=True)
            else:  # Linux/Mac
                cmd = ["gnome-terminal", "--", "python3", "dynamic_quality_testing/terminal_live_monitor.py"]
                try:
                    subprocess.Popen(cmd)
                except FileNotFoundError:
                    # Try with xterm if gnome-terminal is not available
                    try:
                        cmd = ["xterm", "-e", "python3 dynamic_quality_testing/terminal_live_monitor.py"]
                        subprocess.Popen(cmd)
                    except FileNotFoundError:
                        # If no terminal is available, run in the current terminal
                        print("Could not open a new terminal window. Running in the current terminal...")
                        cmd = ["python3", "dynamic_quality_testing/terminal_live_monitor.py"]
                        process = subprocess.Popen(cmd)
                        processes["monitor"] = process
            
            print("✓ Terminal-based live monitoring started")
            print("  Check the new terminal window for real-time updates")
            
        except Exception as e:
            print(f"Error starting terminal monitor: {e}")
            return False
    else:
        # Start the browser-based monitor
        monitor_thread = threading.Thread(target=run_live_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("✓ Browser-based live monitoring dashboard started")
        print(f"  Open http://localhost:{MONITOR_PORT}/live_monitor.html in your browser")
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open the browser
        webbrowser.open(f"http://localhost:{MONITOR_PORT}/live_monitor.html")
    
    return True

def run_live_monitor():
    """Run the browser-based live monitor script."""
    try:
        cmd = ["python3", "dynamic_quality_testing/live_monitor.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes["monitor"] = process
        
        # Wait for the process to finish
        process.wait()
    except Exception as e:
        print(f"Error running browser-based monitor: {e}")

def apply_network_condition(condition):
    """Apply network condition using tc."""
    global current_network
    
    print_section(f"Applying Network Condition: {condition['name']}")
    
    # Get the network condition parameters
    rate = condition["rate"]
    delay = condition["delay"]
    loss = condition["loss"]
    
    # If custom, prompt for values
    if condition["name"] == "Custom":
        rate = input("Enter the rate (e.g., '1mbit'): ")
        delay = input("Enter the delay (e.g., '100ms'): ")
        loss = input("Enter the loss (e.g., '10%'): ")
    
    print(f"Applying: Rate={rate}, Delay={delay}, Loss={loss}")
    
    # Reset any existing tc rules
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    
    # Apply new tc rules
    subprocess.run([
        "sudo", "tc", "qdisc", "add", "dev", INTERFACE, "root", "netem", 
        "rate", rate, 
        "delay", delay, 
        "loss", loss
    ])
    
    # Verify the settings
    result = subprocess.run(["tc", "qdisc", "show", "dev", INTERFACE], 
                           capture_output=True, text=True)
    print(f"TC settings: {result.stdout.strip()}")
    
    # Update current network
    current_network = {
        "name": condition["name"] if condition["name"] != "Custom" else "Custom",
        "rate": rate,
        "delay": delay,
        "loss": loss
    }
    
    # Update live data
    update_live_data()
    
    # Wait for network to stabilize
    time.sleep(2)

def reset_network_condition():
    """Reset network conditions to normal."""
    print_section("Resetting Network Conditions")
    
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    
    print("Network conditions reset to normal")
    
    # Update current network
    global current_network
    current_network = None
    
    # Wait for network to stabilize
    time.sleep(1)

def update_live_data():
    """Update the live data file with current metrics."""
    if current_network is None:
        return
    
    print("Updating live data...")
    
    try:
        import requests
        
        # Default values
        resolution_scale = 0.75
        jpeg_quality = 85
        frame_rate = 20
        metrics = {
            "bandwidth_usage": 5000000,  # 5 Mbps
            "frame_delivery_time": float(current_network["delay"].replace("ms", "")),
            "frame_drop_rate": float(current_network["loss"].replace("%", "")),
            "visual_quality_score": 80,
            "smoothness_score": 70
        }
        
        # Try to get actual values if available
        try:
            # Get current video quality settings from the sender
            try:
                res_response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_resolution", timeout=2)
                if res_response.status_code == 200 and res_response.text.strip():
                    resolution_scale = float(res_response.text)
            except:
                print("Could not get resolution, using default value")
            
            try:
                qual_response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_quality", timeout=2)
                if qual_response.status_code == 200 and qual_response.text.strip():
                    jpeg_quality = int(qual_response.text)
            except:
                print("Could not get quality, using default value")
            
            try:
                fps_response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_fps", timeout=2)
                if fps_response.status_code == 200 and fps_response.text.strip():
                    frame_rate = int(fps_response.text)
            except:
                print("Could not get FPS, using default value")
            
            # Get current metrics from the sender and receiver
            try:
                sender_metrics = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_metrics", timeout=2).json()
                metrics.update(sender_metrics)
            except:
                print("Could not get sender metrics, using estimates")
            
            try:
                receiver_metrics = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/get_metrics", timeout=2).json()
                metrics.update(receiver_metrics)
            except:
                print("Could not get receiver metrics, using estimates")
                
        except Exception as e:
            print(f"Could not get actual values: {e}")
            print("Using estimated values instead")
            
        # Calculate estimated metrics based on network conditions
        # Bandwidth usage (higher with higher resolution, quality, and fps)
        base_bandwidth = 5000000  # 5 Mbps base for full quality
        resolution_factor = {0.5: 0.25, 0.75: 0.5625, 0.9: 0.81, 1.0: 1.0}
        quality_factor = {60: 0.4, 75: 0.6, 85: 0.8, 95: 1.0}
        fps_factor = {10: 0.33, 15: 0.5, 20: 0.67, 30: 1.0}
        
        bandwidth = (base_bandwidth *
                    resolution_factor.get(resolution_scale, 1.0) *
                    quality_factor.get(jpeg_quality, 1.0) *
                    fps_factor.get(frame_rate, 1.0))
        
        # Frame delivery time (higher with higher delay and bandwidth)
        base_delay = float(current_network["delay"].replace("ms", ""))
        delivery_time = base_delay * (1 + (bandwidth / 10000000) * 0.5)
        
        # Frame drop rate (higher with higher loss and bandwidth)
        base_loss = float(current_network["loss"].replace("%", ""))
        drop_rate = base_loss * (1 + (bandwidth / 10000000) * 0.5)
        
        # Visual quality score
        resolution_score = resolution_scale * 100
        quality_score = jpeg_quality
        visual_quality = (resolution_score * 0.6) + (quality_score * 0.4)
        
        # Smoothness score
        fps_score = {10: 30, 15: 50, 20: 70, 30: 100}
        network_score = {"Poor": 30, "Fair": 60, "Good": 80, "Excellent": 100}
        smoothness = (fps_score.get(frame_rate, 50) * 0.7) + (network_score.get(current_network["name"], 50) * 0.3)
        
        # Update metrics with calculated values if not already present
        if "bandwidth_usage" not in metrics:
            metrics["bandwidth_usage"] = bandwidth
        if "frame_delivery_time" not in metrics:
            metrics["frame_delivery_time"] = delivery_time
        if "frame_drop_rate" not in metrics:
            metrics["frame_drop_rate"] = drop_rate
        if "visual_quality_score" not in metrics:
            metrics["visual_quality_score"] = visual_quality
        if "smoothness_score" not in metrics:
            metrics["smoothness_score"] = smoothness
        
        # Create a data point
        data_point = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "network_condition": current_network["name"],
            "network_rate": current_network["rate"],
            "network_delay": current_network["delay"],
            "network_loss": current_network["loss"],
            "resolution_scale": float(resolution_scale),
            "jpeg_quality": int(jpeg_quality),
            "frame_rate": int(frame_rate),
            "metrics": metrics
        }
        
        # Load existing data
        live_data = []
        if os.path.exists(LIVE_DATA_FILE):
            try:
                with open(LIVE_DATA_FILE, 'r') as f:
                    live_data = json.load(f)
            except json.JSONDecodeError:
                live_data = []
        
        # Add the new data point
        live_data.append(data_point)
        
        # Save the updated data
        with open(LIVE_DATA_FILE, 'w') as f:
            json.dump(live_data, f, indent=2)
        
        print(f"Updated live data file with new metrics at {data_point['timestamp']}")
    
    except Exception as e:
        print(f"Error updating live data: {e}")

def show_menu():
    """Show the main menu."""
    print_section("Dynamic Quality Testing with Live Monitoring")
    
    print("1. Apply Excellent Network Condition (10mbit, 20ms, 0%)")
    print("2. Apply Good Network Condition (6mbit, 40ms, 0.5%)")
    print("3. Apply Fair Network Condition (4mbit, 80ms, 1%)")
    print("4. Apply Poor Network Condition (2mbit, 150ms, 3%)")
    print("5. Apply Custom Network Condition")
    print("6. Reset Network Condition")
    print("7. View Current Network Condition")
    print("8. Generate Sample Live Data")
    print("9. Exit")
    
    choice = input("\nEnter your choice (1-9): ")
    
    if choice == "1":
        apply_network_condition(NETWORK_CONDITIONS[0])
    elif choice == "2":
        apply_network_condition(NETWORK_CONDITIONS[1])
    elif choice == "3":
        apply_network_condition(NETWORK_CONDITIONS[2])
    elif choice == "4":
        apply_network_condition(NETWORK_CONDITIONS[3])
    elif choice == "5":
        apply_network_condition(NETWORK_CONDITIONS[4])
    elif choice == "6":
        reset_network_condition()
    elif choice == "7":
        show_current_condition()
    elif choice == "8":
        generate_sample_data()
    elif choice == "9":
        cleanup_and_exit()
    else:
        print("Invalid choice. Please try again.")

def show_current_condition():
    """Show the current network condition."""
    print_section("Current Network Condition")
    
    if current_network is None:
        print("No network condition is currently applied.")
        return
    
    print(f"Name: {current_network['name']}")
    print(f"Rate: {current_network['rate']}")
    print(f"Delay: {current_network['delay']}")
    print(f"Loss: {current_network['loss']}")
    
    # Show tc qdisc statistics
    result = subprocess.run(["tc", "-s", "qdisc", "show", "dev", INTERFACE], 
                           capture_output=True, text=True)
    print("\nTC Statistics:")
    print(result.stdout.strip())

def generate_sample_data():
    """Generate sample live data for demonstration."""
    print_section("Generating Sample Live Data")
    
    try:
        cmd = ["python3", "dynamic_quality_testing/generate_sample_live_data.py"]
        process = subprocess.run(cmd, check=True)
        print("✓ Sample live data generated successfully")
    except Exception as e:
        print(f"✗ Error generating sample live data: {e}")

def cleanup_and_exit():
    """Clean up resources and exit."""
    print_section("Cleaning Up")
    
    # Reset network conditions
    reset_network_condition()
    
    # Stop all processes
    for name, process in processes.items():
        print(f"Stopping {name}...")
        process.terminate()
    
    print("Goodbye!")
    sys.exit(0)

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    print("\nInterrupted by user. Cleaning up...")
    cleanup_and_exit()

def main():
    """Main function."""
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print_header("Dynamic Quality Testing with Live Monitoring")
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    # Start components
    if not start_video_streamer():
        print("Failed to start video streamer. Exiting.")
        return
    
    if not start_video_receiver():
        print("Failed to start video receiver. Exiting.")
        return
    
    if not start_live_monitor():
        print("Failed to start live monitoring dashboard. Exiting.")
        return
    
    # Main loop
    while not stop_event.is_set():
        show_menu()
        time.sleep(1)

if __name__ == "__main__":
    main()