import argparse
import time
import json
import os
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess

# Default settings
DEFAULT_SENDER_IP = "localhost"
DEFAULT_RECEIVER_IP = "localhost"
DEFAULT_SENDER_PORT = 8000
DEFAULT_RECEIVER_PORT = 8001
DEFAULT_INTERVAL = 1.0  # seconds per measurement
DEFAULT_DURATION = 120  # seconds (2 minutes)
DEFAULT_OUTPUT_DIR = "./performance_graphs"

# Global variables
running = True
data = {
    "timestamps": [],
    "measured": {
        "bandwidth": [],  # MB/s (will convert to Mbps)
        "latency": [],    # ms
        "loss_rate": []   # %
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
    except Exception:
        return None

# Function to get metrics from the receiver
def get_receiver_metrics(receiver_ip, receiver_port):
    try:
        response = requests.get(f"http://{receiver_ip}:{receiver_port}/metrics", timeout=1)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None

# Function to run the measurement cycle
def run_measurement_cycle(sender_ip, sender_port, receiver_ip, receiver_port, interval, duration):
    global running, data
    
    start_time = time.time()
    
    print(f"{Colors.GREEN}Starting metrics collection for {duration} seconds...{Colors.ENDC}")
    
    try:
        while running and (duration <= 0 or time.time() - start_time < duration):
            current_time = time.time() - start_time
            data["timestamps"].append(current_time)
            
            # Get measured values
            sender_metrics = get_sender_metrics(sender_ip, sender_port)
            receiver_metrics = get_receiver_metrics(receiver_ip, receiver_port)
            
            if sender_metrics and receiver_metrics:
                bandwidth_mbps = sender_metrics.get("bandwidth_usage", 0) * 8
                latency_ms = receiver_metrics.get("network_latency", 0)
                loss_rate = receiver_metrics.get("frame_drop_rate", 0)
                
                data["measured"]["bandwidth"].append(bandwidth_mbps)
                data["measured"]["latency"].append(latency_ms)
                data["measured"]["loss_rate"].append(loss_rate)
                
                print(f"  Measured - Bandwidth: {bandwidth_mbps:.2f} Mbps, Latency: {latency_ms:.2f} ms, Loss: {loss_rate:.2f}%")
            else:
                data["measured"]["bandwidth"].append(data["measured"]["bandwidth"][-1] if data["measured"]["bandwidth"] else 0)
                data["measured"]["latency"].append(data["measured"]["latency"][-1] if data["measured"]["latency"] else 0)
                data["measured"]["loss_rate"].append(data["measured"]["loss_rate"][-1] if data["measured"]["loss_rate"] else 0)
                print(f"  {Colors.YELLOW}Could not get complete metrics{Colors.ENDC}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Metrics collection stopped by user{Colors.ENDC}")
    
    finally:
        if data["timestamps"]:
            print(f"\n{Colors.GREEN}Collected {len(data['timestamps'])} data points over {data['timestamps'][-1]:.1f} seconds{Colors.ENDC}")

# Function to generate graphs
def generate_graphs(output_dir):
    global data
    
    print(f"{Colors.GREEN}Generating performance graphs...{Colors.ENDC}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.title("Measured Bandwidth")
    plt.plot(data["timestamps"], data["measured"]["bandwidth"], 'r-')
    plt.ylabel("Mbps")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title("Measured Latency")
    plt.plot(data["timestamps"], data["measured"]["latency"], 'r-')
    plt.ylabel("ms")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title("Measured Packet Loss")
    plt.plot(data["timestamps"], data["measured"]["loss_rate"], 'r-')
    plt.xlabel("Time (seconds)")
    plt.ylabel("%")
    plt.grid(True)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"performance_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved performance graph to: {output_file}{Colors.ENDC}")
    
    data_file = os.path.join(output_dir, f"performance_data_{timestamp}.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{Colors.GREEN}Saved raw data to: {data_file}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}Displaying graph...{Colors.ENDC}")
    plt.show()

# Main function
def main():
    global running
    
    parser = argparse.ArgumentParser(description="Performance Monitor for WebRTC Streaming")
    parser.add_argument("--sender-ip", default=DEFAULT_SENDER_IP, help="Sender IP address")
    parser.add_argument("--receiver-ip", default=DEFAULT_RECEIVER_IP, help="Receiver IP address")
    parser.add_argument("--sender-port", type=int, default=DEFAULT_SENDER_PORT, help="Sender metrics port")
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT, help="Receiver metrics port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Metrics collection interval in seconds")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Total duration in seconds (0 for unlimited)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}Performance Monitor for WebRTC Streaming{Colors.ENDC}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    print(f"Sender: {args.sender_ip}:{args.sender_port}")
    print(f"Receiver: {args.receiver_ip}:{args.receiver_port}")
    print(f"Interval: {args.interval} seconds")
    print(f"Total Duration: {args.duration} seconds (0 = unlimited)")
    print(f"Output Directory: {args.output}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Checking metrics APIs...{Colors.ENDC}")
    
    if not get_sender_metrics(args.sender_ip, args.sender_port):
        print(f"{Colors.YELLOW}Warning: Could not connect to sender metrics API{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}Successfully connected to sender metrics API{Colors.ENDC}")

    if not get_receiver_metrics(args.receiver_ip, args.receiver_port):
        print(f"{Colors.YELLOW}Warning: Could not connect to receiver metrics API{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}Successfully connected to receiver metrics API{Colors.ENDC}")
    
    run_measurement_cycle(args.sender_ip, args.sender_port, args.receiver_ip, args.receiver_port, args.interval, args.duration)
    
    if len(data["timestamps"]) > 0:
        generate_graphs(args.output)
        print(f"\n{Colors.GREEN}Analysis complete!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}No data collected. Cannot generate graphs.{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Program interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
