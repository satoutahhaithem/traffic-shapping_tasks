#!/usr/bin/env python3
"""
Performance Monitor for WebRTC Streaming

This script monitors the performance of the WebRTC streaming system by fetching
metrics from the sender and receiver APIs, and generates graphs to visualize
the performance under different network conditions.

Usage:
    python performance_monitor.py [--sender-ip SENDER_IP] [--receiver-ip RECEIVER_IP]
                                 [--sender-port SENDER_PORT] [--receiver-port RECEIVER_PORT]
                                 [--interval INTERVAL] [--duration DURATION]
                                 [--output OUTPUT_DIR] [--live]

Author: Roo AI Assistant
Date: May 2025
"""

import argparse
import time
import json
import os
import subprocess
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import threading

# Default settings
DEFAULT_SENDER_IP = "localhost"
DEFAULT_RECEIVER_IP = "192.168.2.169"
DEFAULT_SENDER_PORT = 8000
DEFAULT_RECEIVER_PORT = 8001
DEFAULT_INTERVAL = 1.0  # seconds
DEFAULT_DURATION = 300  # seconds (5 minutes)
DEFAULT_OUTPUT_DIR = "./performance_graphs"

# Global variables
running = True
data = {
    "timestamps": [],
    "sender": {
        "bandwidth_usage": [],
        "actual_fps": [],
        "buffer_fullness": [],
        "frame_size": []
    },
    "receiver": {
        "frame_delivery_time": [],
        "actual_fps": [],
        "frame_drop_rate": [],
        "buffer_fullness": []
    },
    "network": {
        "rate": [],
        "delay": [],
        "loss": []
    }
}

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Function to get metrics from the sender
def get_sender_metrics(sender_ip, sender_port):
    try:
        response = requests.get(f"http://{sender_ip}:{sender_port}/metrics", timeout=1)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}Error getting sender metrics: HTTP {response.status_code}{Colors.ENDC}")
            return None
    except Exception as e:
        print(f"{Colors.RED}Error connecting to sender metrics API: {e}{Colors.ENDC}")
        return None

# Function to get metrics from the receiver
def get_receiver_metrics(receiver_ip, receiver_port):
    try:
        response = requests.get(f"http://{receiver_ip}:{receiver_port}/metrics", timeout=1)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}Error getting receiver metrics: HTTP {response.status_code}{Colors.ENDC}")
            return None
    except Exception as e:
        print(f"{Colors.RED}Error connecting to receiver metrics API: {e}{Colors.ENDC}")
        return None

# Function to get current network conditions using tc
def get_network_conditions():
    try:
        # Get the default interface
        result = subprocess.run(["ip", "route", "get", "8.8.8.8"], 
                               capture_output=True, text=True, check=True)
        interface = result.stdout.split()[4]
        
        # Get current tc settings
        result = subprocess.run(["tc", "qdisc", "show", "dev", interface], 
                               capture_output=True, text=True, check=True)
        
        # Parse the output to extract metrics
        conditions = {
            "rate": "0",
            "delay": "0ms",
            "loss": "0%"
        }
        
        # Extract rate
        rate_match = result.stdout.find("rate ")
        if rate_match != -1:
            rate_end = result.stdout.find(" ", rate_match + 5)
            conditions["rate"] = result.stdout[rate_match + 5:rate_end]
        
        # Extract delay
        delay_match = result.stdout.find("delay ")
        if delay_match != -1:
            delay_end = result.stdout.find(" ", delay_match + 6)
            conditions["delay"] = result.stdout[delay_match + 6:delay_end]
        
        # Extract loss
        loss_match = result.stdout.find("loss ")
        if loss_match != -1:
            loss_end = result.stdout.find(" ", loss_match + 5)
            conditions["loss"] = result.stdout[loss_match + 5:loss_end]
        
        return conditions
    except Exception as e:
        print(f"{Colors.YELLOW}Error getting network conditions: {e}{Colors.ENDC}")
        return {"rate": "0", "delay": "0ms", "loss": "0%"}

# Function to collect metrics
def collect_metrics(sender_ip, sender_port, receiver_ip, receiver_port, interval, duration):
    global running, data
    
    start_time = time.time()
    count = 0
    
    print(f"{Colors.GREEN}Starting metrics collection...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press Ctrl+C to stop collection and generate graphs{Colors.ENDC}")
    
    try:
        while running and (duration <= 0 or time.time() - start_time < duration):
            # Get current timestamp
            current_time = time.time() - start_time
            data["timestamps"].append(current_time)
            
            # Get sender metrics
            sender_metrics = get_sender_metrics(sender_ip, sender_port)
            if sender_metrics:
                data["sender"]["bandwidth_usage"].append(sender_metrics.get("bandwidth_usage", 0))
                data["sender"]["actual_fps"].append(sender_metrics.get("actual_fps", 0))
                data["sender"]["buffer_fullness"].append(sender_metrics.get("buffer_fullness", 0))
                data["sender"]["frame_size"].append(sender_metrics.get("frame_size", 0))
            else:
                # Use previous values or 0 if no previous values
                data["sender"]["bandwidth_usage"].append(data["sender"]["bandwidth_usage"][-1] if data["sender"]["bandwidth_usage"] else 0)
                data["sender"]["actual_fps"].append(data["sender"]["actual_fps"][-1] if data["sender"]["actual_fps"] else 0)
                data["sender"]["buffer_fullness"].append(data["sender"]["buffer_fullness"][-1] if data["sender"]["buffer_fullness"] else 0)
                data["sender"]["frame_size"].append(data["sender"]["frame_size"][-1] if data["sender"]["frame_size"] else 0)
            
            # Get receiver metrics
            receiver_metrics = get_receiver_metrics(receiver_ip, receiver_port)
            if receiver_metrics:
                data["receiver"]["frame_delivery_time"].append(receiver_metrics.get("frame_delivery_time", 0))
                data["receiver"]["actual_fps"].append(receiver_metrics.get("actual_fps", 0))
                data["receiver"]["frame_drop_rate"].append(receiver_metrics.get("frame_drop_rate", 0))
                data["receiver"]["buffer_fullness"].append(receiver_metrics.get("buffer_fullness", 0))
            else:
                # Use previous values or 0 if no previous values
                data["receiver"]["frame_delivery_time"].append(data["receiver"]["frame_delivery_time"][-1] if data["receiver"]["frame_delivery_time"] else 0)
                data["receiver"]["actual_fps"].append(data["receiver"]["actual_fps"][-1] if data["receiver"]["actual_fps"] else 0)
                data["receiver"]["frame_drop_rate"].append(data["receiver"]["frame_drop_rate"][-1] if data["receiver"]["frame_drop_rate"] else 0)
                data["receiver"]["buffer_fullness"].append(data["receiver"]["buffer_fullness"][-1] if data["receiver"]["buffer_fullness"] else 0)
            
            # Get network conditions
            network_conditions = get_network_conditions()
            
            # Convert rate to Mbps for graphing
            rate_str = network_conditions["rate"]
            if "Kbit" in rate_str or "kbit" in rate_str:
                rate_value = float(rate_str.replace("Kbit", "").replace("kbit", "")) / 1000
            elif "Mbit" in rate_str or "mbit" in rate_str:
                rate_value = float(rate_str.replace("Mbit", "").replace("mbit", ""))
            elif "Gbit" in rate_str or "gbit" in rate_str:
                rate_value = float(rate_str.replace("Gbit", "").replace("gbit", "")) * 1000
            else:
                rate_value = 0
            
            # Convert delay to ms for graphing
            delay_str = network_conditions["delay"]
            if "us" in delay_str:
                delay_value = float(delay_str.replace("us", "")) / 1000
            elif "ms" in delay_str:
                delay_value = float(delay_str.replace("ms", ""))
            elif "s" in delay_str:
                delay_value = float(delay_str.replace("s", "")) * 1000
            else:
                delay_value = 0
            
            # Convert loss to percentage for graphing
            loss_str = network_conditions["loss"]
            loss_value = float(loss_str.replace("%", ""))
            
            data["network"]["rate"].append(rate_value)
            data["network"]["delay"].append(delay_value)
            data["network"]["loss"].append(loss_value)
            
            # Print current metrics every 5 seconds
            if count % 5 == 0:
                print(f"\n{Colors.BLUE}======================================================{Colors.ENDC}")
                print(f"{Colors.BLUE}PERFORMANCE METRICS - {time.strftime('%H:%M:%S')}{Colors.ENDC}")
                print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
                
                print(f"{Colors.CYAN}Network Conditions:{Colors.ENDC}")
                print(f"  Rate: {network_conditions['rate']}")
                print(f"  Delay: {network_conditions['delay']}")
                print(f"  Loss: {network_conditions['loss']}")
                
                if sender_metrics:
                    print(f"\n{Colors.CYAN}Sender Metrics:{Colors.ENDC}")
                    print(f"  Bandwidth Usage: {sender_metrics.get('bandwidth_usage', 0):.2f} MB/s")
                    print(f"  Actual FPS: {sender_metrics.get('actual_fps', 0):.1f}")
                    print(f"  Buffer Fullness: {sender_metrics.get('buffer_fullness', 0):.1f}%")
                    print(f"  Frame Size: {sender_metrics.get('frame_size', 0):.1f} KB")
                
                if receiver_metrics:
                    print(f"\n{Colors.CYAN}Receiver Metrics:{Colors.ENDC}")
                    print(f"  Frame Delivery Time: {receiver_metrics.get('frame_delivery_time', 0):.1f} ms")
                    print(f"  Actual FPS: {receiver_metrics.get('actual_fps', 0):.1f}")
                    print(f"  Frame Drop Rate: {receiver_metrics.get('frame_drop_rate', 0):.1f}%")
                    print(f"  Buffer Fullness: {receiver_metrics.get('buffer_fullness', 0):.1f}%")
            
            # Wait for the next interval
            time.sleep(interval)
            count += 1
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Metrics collection stopped by user{Colors.ENDC}")
    
    except Exception as e:
        print(f"\n{Colors.RED}Error collecting metrics: {e}{Colors.ENDC}")
    
    finally:
        print(f"\n{Colors.GREEN}Collected {len(data['timestamps'])} data points over {data['timestamps'][-1]:.1f} seconds{Colors.ENDC}")

# Function to generate static graphs
def generate_graphs(output_dir):
    global data
    
    print(f"{Colors.GREEN}Generating performance graphs...{Colors.ENDC}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2)
    
    # Plot 1: Network Conditions
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("Network Conditions")
    ax1.plot(data["timestamps"], data["network"]["rate"], 'b-', label="Rate (Mbps)")
    ax1.set_ylabel("Rate (Mbps)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(data["timestamps"], data["network"]["delay"], 'r-', label="Delay (ms)")
    ax1_twin.set_ylabel("Delay (ms)", color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Packet Loss
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Packet Loss")
    ax2.plot(data["timestamps"], data["network"]["loss"], 'g-', label="Loss (%)")
    ax2.plot(data["timestamps"], data["receiver"]["frame_drop_rate"], 'm-', label="Frame Drop Rate (%)")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend()
    
    # Plot 3: Bandwidth Usage vs Network Rate
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title("Bandwidth Usage vs Network Rate")
    ax3.plot(data["timestamps"], data["sender"]["bandwidth_usage"], 'b-', label="Actual Bandwidth (MB/s)")
    ax3.set_ylabel("Bandwidth (MB/s)", color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(data["timestamps"], data["network"]["rate"], 'r--', label="Network Rate (Mbps)")
    ax3_twin.set_ylabel("Rate (Mbps)", color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 4: Frame Delivery Time vs Network Delay
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title("Frame Delivery Time vs Network Delay")
    ax4.plot(data["timestamps"], data["receiver"]["frame_delivery_time"], 'b-', label="Frame Delivery Time (ms)")
    ax4.set_ylabel("Delivery Time (ms)", color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(data["timestamps"], data["network"]["delay"], 'r--', label="Network Delay (ms)")
    ax4_twin.set_ylabel("Delay (ms)", color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 5: FPS Comparison
    ax5 = plt.subplot(gs[2, 0])
    ax5.set_title("FPS Comparison")
    ax5.plot(data["timestamps"], data["sender"]["actual_fps"], 'b-', label="Sender FPS")
    ax5.plot(data["timestamps"], data["receiver"]["actual_fps"], 'r-', label="Receiver FPS")
    ax5.set_ylabel("Frames Per Second")
    ax5.legend()
    
    # Plot 6: Buffer Fullness
    ax6 = plt.subplot(gs[2, 1])
    ax6.set_title("Buffer Fullness")
    ax6.plot(data["timestamps"], data["sender"]["buffer_fullness"], 'b-', label="Sender Buffer (%)")
    ax6.plot(data["timestamps"], data["receiver"]["buffer_fullness"], 'r-', label="Receiver Buffer (%)")
    ax6.set_ylabel("Buffer Fullness (%)")
    ax6.legend()
    
    # Plot 7: Frame Size
    ax7 = plt.subplot(gs[3, 0])
    ax7.set_title("Frame Size")
    ax7.plot(data["timestamps"], data["sender"]["frame_size"], 'g-', label="Frame Size (KB)")
    ax7.set_ylabel("Size (KB)")
    
    # Plot 8: Performance Impact
    ax8 = plt.subplot(gs[3, 1])
    ax8.set_title("Performance Impact")
    
    # Calculate performance impact as ratio of receiver FPS to sender FPS
    performance_impact = []
    for i in range(len(data["timestamps"])):
        if data["sender"]["actual_fps"][i] > 0:
            impact = data["receiver"]["actual_fps"][i] / data["sender"]["actual_fps"][i] * 100
        else:
            impact = 0
        performance_impact.append(impact)
    
    ax8.plot(data["timestamps"], performance_impact, 'b-', label="Performance Ratio (%)")
    ax8.set_ylabel("Ratio (%)")
    ax8.set_ylim([0, 110])
    
    # Add horizontal line at 100%
    ax8.axhline(y=100, color='r', linestyle='--')
    
    # Add overall title and adjust layout
    plt.suptitle("WebRTC Streaming Performance Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    output_file = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved performance graph to: {output_file}{Colors.ENDC}")
    
    # Generate individual graphs for better detail
    
    # Network Conditions Graph
    plt.figure(figsize=(10, 6))
    plt.title("Network Conditions Over Time")
    plt.plot(data["timestamps"], data["network"]["rate"], 'b-', label="Rate (Mbps)")
    plt.plot(data["timestamps"], data["network"]["delay"], 'r-', label="Delay (ms)")
    plt.plot(data["timestamps"], data["network"]["loss"], 'g-', label="Loss (%)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, f"network_conditions_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved network conditions graph to: {output_file}{Colors.ENDC}")
    
    # Performance Comparison Graph
    plt.figure(figsize=(10, 6))
    plt.title("Performance Comparison")
    plt.plot(data["timestamps"], data["sender"]["actual_fps"], 'b-', label="Sender FPS")
    plt.plot(data["timestamps"], data["receiver"]["actual_fps"], 'r-', label="Receiver FPS")
    plt.plot(data["timestamps"], performance_impact, 'g-', label="Performance Ratio (%)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, f"performance_comparison_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved performance comparison graph to: {output_file}{Colors.ENDC}")
    
    # Save raw data as JSON for later analysis
    data_file = os.path.join(output_dir, f"performance_data_{timestamp}.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{Colors.GREEN}Saved raw performance data to: {data_file}{Colors.ENDC}")
    
    return output_file

# Function to update live graph
def update_live_graph(frame_num, *args):
    global data
    
    # Unpack arguments
    fig, axes = args[0], args[1]
    
    # Clear all axes
    for ax in axes:
        ax.clear()
    
    # Plot data on each axis
    
    # Plot 1: Network Conditions
    axes[0].set_title("Network Conditions")
    axes[0].plot(data["timestamps"], data["network"]["rate"], 'b-', label="Rate (Mbps)")
    axes[0].set_ylabel("Rate (Mbps)", color='b')
    axes[0].tick_params(axis='y', labelcolor='b')
    
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(data["timestamps"], data["network"]["delay"], 'r-', label="Delay (ms)")
    ax0_twin.set_ylabel("Delay (ms)", color='r')
    ax0_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: FPS Comparison
    axes[1].set_title("FPS Comparison")
    axes[1].plot(data["timestamps"], data["sender"]["actual_fps"], 'b-', label="Sender FPS")
    axes[1].plot(data["timestamps"], data["receiver"]["actual_fps"], 'r-', label="Receiver FPS")
    axes[1].set_ylabel("Frames Per Second")
    axes[1].legend()
    
    # Plot 3: Buffer Fullness
    axes[2].set_title("Buffer Fullness")
    axes[2].plot(data["timestamps"], data["sender"]["buffer_fullness"], 'b-', label="Sender Buffer (%)")
    axes[2].plot(data["timestamps"], data["receiver"]["buffer_fullness"], 'r-', label="Receiver Buffer (%)")
    axes[2].set_ylabel("Buffer Fullness (%)")
    axes[2].legend()
    
    # Plot 4: Performance Impact
    axes[3].set_title("Performance Impact")
    
    # Calculate performance impact as ratio of receiver FPS to sender FPS
    performance_impact = []
    for i in range(len(data["timestamps"])):
        if data["sender"]["actual_fps"][i] > 0:
            impact = data["receiver"]["actual_fps"][i] / data["sender"]["actual_fps"][i] * 100
        else:
            impact = 0
        performance_impact.append(impact)
    
    axes[3].plot(data["timestamps"], performance_impact, 'b-', label="Performance Ratio (%)")
    axes[3].set_ylabel("Ratio (%)")
    axes[3].set_ylim([0, 110])
    
    # Add horizontal line at 100%
    axes[3].axhline(y=100, color='r', linestyle='--')
    
    # Add overall title
    fig.suptitle("WebRTC Streaming Performance Analysis (Live)", fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return axes

# Function to run live graph
def run_live_graph():
    global running, data
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_live_graph, fargs=(fig, axes), interval=1000)
    
    # Show the plot
    plt.show()
    
    # When plot window is closed, stop the collection
    running = False

# Main function
def main():
    global running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Performance Monitor for WebRTC Streaming")
    parser.add_argument("--sender-ip", default=DEFAULT_SENDER_IP, help="Sender IP address")
    parser.add_argument("--receiver-ip", default=DEFAULT_RECEIVER_IP, help="Receiver IP address")
    parser.add_argument("--sender-port", type=int, default=DEFAULT_SENDER_PORT, help="Sender metrics port")
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT, help="Receiver metrics port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Metrics collection interval (seconds)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Collection duration (seconds, 0=unlimited)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs")
    parser.add_argument("--live", action="store_true", help="Show live graphs during collection")
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
    print(f"{Colors.BLUE}WEBRTC STREAMING PERFORMANCE MONITOR{Colors.ENDC}")
    print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
    print(f"Sender: {args.sender_ip}:{args.sender_port}")
    print(f"Receiver: {args.receiver_ip}:{args.receiver_port}")
    print(f"Interval: {args.interval} seconds")
    print(f"Duration: {args.duration} seconds (0=unlimited)")
    print(f"Output directory: {args.output}")
    print(f"Live graphs: {'Enabled' if args.live else 'Disabled'}")
    print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
    
    # Check if sender and receiver metrics are available
    print(f"\n{Colors.CYAN}Checking sender metrics...{Colors.ENDC}")
    sender_metrics = get_sender_metrics(args.sender_ip, args.sender_port)
    if sender_metrics:
        print(f"{Colors.GREEN}✓ Sender metrics available{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠ Sender metrics not available. Make sure the sender is running with metrics enabled.{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Start sender with: python direct_sender.py --metrics-port {args.sender_port}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Checking receiver metrics...{Colors.ENDC}")
    receiver_metrics = get_receiver_metrics(args.receiver_ip, args.receiver_port)
    if receiver_metrics:
        print(f"{Colors.GREEN}✓ Receiver metrics available{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠ Receiver metrics not available. Make sure the receiver is running with metrics enabled.{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Start receiver with: python direct_receiver.py --display --metrics-port {args.receiver_port}{Colors.ENDC}")
    
    # Start live graph in a separate thread if requested
    if args.live:
        live_thread = threading.Thread(target=run_live_graph)
        live_thread.daemon = True
        live_thread.start()
    
    # Collect metrics
    collect_metrics(args.sender_ip, args.sender_port, args.receiver_ip, args.receiver_port, 
                   args.interval, args.duration)
    
    # Generate graphs
    if data["timestamps"]:
        output_file = generate_graphs(args.output)
        
        # Open the graph file if we're on a system with a GUI
        try:
            if os.name == 'posix':  # Linux/Unix
                subprocess.run(["xdg-open", output_file], check=False)
            elif os.name == 'nt':  # Windows
                os.startfile(output_file)
            elif os.name == 'darwin':  # macOS
                subprocess.run(["open", output_file], check=False)
        except Exception as e:
            print(f"{Colors.YELLOW}Could not open graph file automatically: {e}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Graph saved to: {output_file}{Colors.ENDC}")
    else:
        print(f"{Colors.RED}No data collected. Cannot generate graphs.{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}Performance monitoring complete.{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Performance monitoring stopped by user.{Colors.ENDC}")
        running = False
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        running = False