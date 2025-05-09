#!/usr/bin/env python3

"""
Terminal-based live monitoring script for dynamic quality testing.
This script displays graphs directly in the terminal, avoiding browser caching issues.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import threading
import signal

# Configuration
RESULTS_DIR = "test_results"
LIVE_DATA_FILE = os.path.join(RESULTS_DIR, "live_data.json")
UPDATE_INTERVAL = 1  # seconds
MAX_POINTS = 10  # Reduced to 10 points to make recent changes more visible

# Global variables
live_data = []
last_update_time = 0
stop_event = threading.Event()

def clear_terminal():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

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

def load_live_data():
    """Load the live data from the file."""
    global live_data, last_update_time
    
    if os.path.exists(LIVE_DATA_FILE):
        file_mod_time = os.path.getmtime(LIVE_DATA_FILE)
        
        if file_mod_time > last_update_time:
            try:
                with open(LIVE_DATA_FILE, 'r') as f:
                    live_data = json.load(f)
                
                last_update_time = file_mod_time
                return True
            except json.JSONDecodeError:
                print("Error: Could not decode JSON data.")
                return False
    
    return False

def generate_ascii_graph(values, title, max_width=70, max_height=10):
    """Generate an ASCII graph from a list of values."""
    if not values:
        return [f"{title}: No data available"]
    
    # Normalize values to fit in the graph
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        normalized = [max_height // 2] * len(values)
    else:
        normalized = [int((v - min_val) / (max_val - min_val) * (max_height - 1)) for v in values]
    
    # Generate the graph
    graph = []
    graph.append(f"{title}: Min={min_val:.2f}, Max={max_val:.2f}, Current={values[-1]:.2f}")
    
    # Add a legend for the timeline
    timeline = "Timeline: "
    timeline += "Oldest " + "-" * (len(values) - 6) + " Newest"
    graph.append(timeline)
    
    for h in range(max_height - 1, -1, -1):
        line = "│"
        for i, n in enumerate(normalized):
            if n >= h:
                # Highlight the most recent value
                if i == len(normalized) - 1:
                    line += "▓"  # Use a different character for the most recent value
                else:
                    line += "█"
            else:
                line += " "
        graph.append(line)
    
    # Add the x-axis
    graph.append("└" + "─" * len(values))
    
    return graph

def display_help():
    """Display help information about how to interpret the graphs."""
    print_section("How to Interpret the Graphs")
    
    print("Each graph shows the most recent data points (up to 20), with the newest points on the right.")
    print("The height of each bar represents the value of the metric at that point in time.")
    print()
    
    print("1. Bandwidth Usage (Mbps):")
    print("   - Shows the actual bandwidth used by the video stream")
    print("   - Higher values mean more data is being transmitted")
    print("   - Should decrease when network conditions worsen")
    print()
    
    print("2. Frame Delivery Time (ms):")
    print("   - Shows how long it takes for frames to be delivered")
    print("   - Higher values mean more delay in the video")
    print("   - Should increase when network delay increases")
    print()
    
    print("3. Frame Drop Rate (%):")
    print("   - Shows the percentage of frames that are dropped")
    print("   - Higher values mean more frames are being lost")
    print("   - Should increase when network loss increases")
    print()
    
    print("4. Visual Quality Score:")
    print("   - Shows the estimated visual quality of the video")
    print("   - Higher values mean better visual quality")
    print("   - Should decrease when network conditions worsen")
    print()
    
    print("5. Smoothness Score:")
    print("   - Shows how smooth the video playback is")
    print("   - Higher values mean smoother playback")
    print("   - Should decrease when network conditions worsen")
    print()
    
    print("Graphs will appear in a few seconds...")
    time.sleep(5)  # Give the user time to read the help

def display_live_graphs():
    """Display the live graphs in the terminal."""
    if not live_data:
        print("No live data available yet.")
        return
    
    # Extract the most recent data points (limited to MAX_POINTS)
    recent_data = live_data[-MAX_POINTS:]
    
    # Extract timestamps and metrics
    timestamps = []
    bandwidth_usage = []
    frame_delivery_time = []
    frame_drop_rate = []
    visual_quality = []
    smoothness = []
    network_conditions = []
    
    for result in recent_data:
        if 'timestamp' in result:
            try:
                timestamp = datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S")
                timestamps.append(timestamp)
            except ValueError:
                timestamps.append(None)
        else:
            timestamps.append(None)
        
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
        
        network_conditions.append(f"{result.get('network_condition', 'Unknown')}: {result.get('network_rate', 'N/A')}, {result.get('network_delay', 'N/A')}, {result.get('network_loss', 'N/A')}")
    
    # Clear the terminal
    clear_terminal()
    
    # Print the header
    print_header("Terminal Live Quality Monitoring")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data points: {len(live_data)} (showing last {len(recent_data)})")
    
    # Print the most recent network condition with highlighting
    if network_conditions:
        condition = network_conditions[-1]
        print("\n" + "*" * 80)
        print(f"CURRENT NETWORK CONDITION: {condition}")
        print("*" * 80)
    
    # Generate and display ASCII graphs
    print_section("Bandwidth Usage (Mbps)")
    for line in generate_ascii_graph(bandwidth_usage, "Bandwidth Usage (Mbps)"):
        print(line)
    
    print_section("Frame Delivery Time (ms)")
    for line in generate_ascii_graph(frame_delivery_time, "Frame Delivery Time (ms)"):
        print(line)
    
    print_section("Frame Drop Rate (%)")
    for line in generate_ascii_graph(frame_drop_rate, "Frame Drop Rate (%)"):
        print(line)
    
    print_section("Quality Scores")
    for line in generate_ascii_graph(visual_quality, "Visual Quality Score"):
        print(line)
    
    for line in generate_ascii_graph(smoothness, "Smoothness Score"):
        print(line)
    
    print("\nPress Enter to continue, 'm' for menu, Ctrl+C to exit.")

def show_menu():
    """Show the main menu."""
    print_section("Terminal Live Quality Monitoring - Menu")
    
    print("1. Apply Excellent Network Condition (10mbit, 20ms, 0%)")
    print("2. Apply Good Network Condition (6mbit, 40ms, 0.5%)")
    print("3. Apply Fair Network Condition (4mbit, 80ms, 1%)")
    print("4. Apply Poor Network Condition (2mbit, 150ms, 3%)")
    print("5. Apply Custom Network Condition")
    print("6. Reset Network Condition")
    print("7. View Current Network Condition")
    print("8. Show Help")
    print("9. Continue Monitoring")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-9): ")
    
    if choice == "1":
        apply_network_condition("Excellent", "10mbit", "20ms", "0%")
    elif choice == "2":
        apply_network_condition("Good", "6mbit", "40ms", "0.5%")
    elif choice == "3":
        apply_network_condition("Fair", "4mbit", "80ms", "1%")
    elif choice == "4":
        apply_network_condition("Poor", "2mbit", "150ms", "3%")
    elif choice == "5":
        rate = input("Enter the rate (e.g., '1mbit'): ")
        delay = input("Enter the delay (e.g., '100ms'): ")
        loss = input("Enter the loss (e.g., '10%'): ")
        apply_network_condition("Custom", rate, delay, loss)
    elif choice == "6":
        reset_network_condition()
    elif choice == "7":
        show_current_condition()
    elif choice == "8":
        display_help()
    elif choice == "9":
        return True  # Continue monitoring
    elif choice == "0":
        print("\nExiting...")
        stop_event.set()
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")
    
    input("\nPress Enter to continue...")
    return True

def apply_network_condition(name, rate, delay, loss):
    """Apply network condition using tc."""
    print_section(f"Applying Network Condition: {name}")
    
    print(f"Applying: Rate={rate}, Delay={delay}, Loss={loss}")
    
    # Reset any existing tc rules
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "wlp0s20f3", "root"],
                  stderr=subprocess.DEVNULL)
    
    # Apply new tc rules
    subprocess.run([
        "sudo", "tc", "qdisc", "add", "dev", "wlp0s20f3", "root", "netem",
        "rate", rate,
        "delay", delay,
        "loss", loss
    ])
    
    # Verify the settings
    result = subprocess.run(["tc", "qdisc", "show", "dev", "wlp0s20f3"],
                           capture_output=True, text=True)
    print(f"TC settings: {result.stdout.strip()}")

def reset_network_condition():
    """Reset network conditions to normal."""
    print_section("Resetting Network Conditions")
    
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "wlp0s20f3", "root"],
                  stderr=subprocess.DEVNULL)
    
    print("Network conditions reset to normal")

def show_current_condition():
    """Show the current network condition."""
    print_section("Current Network Condition")
    
    # Show tc qdisc statistics
    result = subprocess.run(["tc", "-s", "qdisc", "show", "dev", "wlp0s20f3"],
                           capture_output=True, text=True)
    print("\nTC Statistics:")
    print(result.stdout.strip())
    
    input("\nPress Enter to continue...")

def monitor_loop():
    """Main monitoring loop."""
    show_help_first = True
    show_menu_option = False
    
    while not stop_event.is_set():
        if show_help_first:
            display_help()
            show_help_first = False
        
        if load_live_data():
            display_live_graphs()
        
        if show_menu_option:
            # Show menu after displaying graphs
            clear_terminal()
            show_menu()
            show_menu_option = False
        else:
            # Ask user if they want to show menu
            choice = input("\nEnter 'm' for menu, or press Enter to continue monitoring: ")
            if choice.lower() == 'm':
                show_menu_option = True
            
            # Wait a bit before updating again
            time.sleep(UPDATE_INTERVAL)

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    print("\nExiting...")
    stop_event.set()
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print_header("Terminal Live Quality Monitoring")
    print("Starting monitoring...")
    print("\nHelp information will be displayed first, then the graphs will appear.")
    print("Press Enter to continue, 'm' for menu, Ctrl+C to exit at any time.")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Create initial live data file if it doesn't exist
    if not os.path.exists(LIVE_DATA_FILE):
        with open(LIVE_DATA_FILE, 'w') as f:
            json.dump([], f)
    
    # Start the monitoring loop
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()