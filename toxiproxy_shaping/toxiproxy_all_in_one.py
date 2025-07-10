import argparse
import time
import json
import os
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from toxiproxy import Toxiproxy

# Default settings
DEFAULT_SENDER_IP = "localhost"
DEFAULT_RECEIVER_IP = "localhost"
DEFAULT_SENDER_PORT = 8000
DEFAULT_RECEIVER_PORT = 8001
DEFAULT_INTERVAL = 10.0  # seconds
DEFAULT_DURATION = 120  # seconds (2 minutes)
DEFAULT_OUTPUT_DIR = "./toxiproxy_performance_graphs"

# Global variables
running = True
data = {
    "timestamps": [],
    "commanded": {
        "rate": [],      # Mbps
        "delay": [],     # ms
        "loss": []       # %
    },
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

# Define the network condition presets
NETWORK_PRESETS = [
    {"name": "VERY POOR", "rate": 0.5, "delay": 300, "loss": 0},
    {"name": "POOR", "rate": 1, "delay": 150, "loss": 0},
    {"name": "FAIR", "rate": 2, "delay": 80, "loss": 0},
    {"name": "GOOD", "rate": 5, "delay": 40, "loss": 0},
    {"name": "EXCELLENT", "rate": 10, "delay": 20, "loss": 0},
    {"name": "ULTRA", "rate": 50, "delay": 1, "loss": 0}
]

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

# Function to apply conditions and collect metrics
def run_test_cycle(sender_ip, sender_port, receiver_ip, receiver_port, interval, duration):
    global running, data
    
    # Connect to Toxiproxy and set up the proxy
    toxiproxy = Toxiproxy()
    proxy = toxiproxy.create_proxy(
        name="video_stream",
        listen="tcp://0.0.0.0:8666",
        upstream="tcp://127.0.0.1:9999"
    )
    proxy.add_toxic("latency", "latency", latency=0)
    proxy.add_toxic("bandwidth", "bandwidth", rate=100000)

    start_time = time.time()
    
    print(f"{Colors.GREEN}Starting metrics collection for {duration} seconds...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press Ctrl+C at any time to stop collection and generate graphs immediately{Colors.ENDC}")
    
    try:
        while running and (duration <= 0 or time.time() - start_time < duration):
            for preset in NETWORK_PRESETS:
                # Apply the network conditions
                print(f"\n{Colors.BLUE}======================================================{Colors.ENDC}")
                print(f"{Colors.BLUE}APPLYING PRESET: {preset['name']}{Colors.ENDC}")
                print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
                proxy.update_toxic("latency", latency=preset["delay"])
                proxy.update_toxic("bandwidth", rate=int(preset["rate"] * 1000)) # Convert Mbps to KBps

                # Wait for a moment for the conditions to take effect
                time.sleep(2)

                # Collect metrics for the duration of the interval
                cycle_start_time = time.time()
                while time.time() - cycle_start_time < interval:
                    current_time = time.time() - start_time
                    data["timestamps"].append(current_time)
                    
                    # Store commanded values
                    data["commanded"]["rate"].append(preset["rate"])
                    data["commanded"]["delay"].append(preset["delay"])
                    data["commanded"]["loss"].append(preset["loss"])
                    
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

                    time.sleep(1) # Collect metrics every second

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Metrics collection stopped by user{Colors.ENDC}")
    
    except Exception as e:
        print(f"\n{Colors.RED}Error during test cycle: {e}{Colors.ENDC}")
    
    finally:
        if data["timestamps"]:
            print(f"\n{Colors.GREEN}Collected {len(data['timestamps'])} data points over {data['timestamps'][-1]:.1f} seconds{Colors.ENDC}")
        proxy.delete()
        print("Toxiproxy proxy deleted.")

# Function to generate graphs
def generate_graphs(output_dir):
    global data
    
    print(f"{Colors.GREEN}Generating performance comparison graphs...{Colors.ENDC}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.title("Bandwidth Comparison")
    plt.plot(data["timestamps"], data["commanded"]["rate"], 'b-', label="Commanded")
    plt.plot(data["timestamps"], data["measured"]["bandwidth"], 'r-', label="Measured")
    plt.ylabel("Mbps")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title("Latency Comparison")
    plt.plot(data["timestamps"], data["commanded"]["delay"], 'b-', label="Commanded")
    plt.plot(data["timestamps"], data["measured"]["latency"], 'r-', label="Measured")
    plt.ylabel("ms")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title("Packet Loss Comparison")
    plt.plot(data["timestamps"], data["commanded"]["loss"], 'b-', label="Commanded")
    plt.plot(data["timestamps"], data["measured"]["loss_rate"], 'r-', label="Measured")
    plt.xlabel("Time (seconds)")
    plt.ylabel("%")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"toxiproxy_performance_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved performance graph to: {output_file}{Colors.ENDC}")
    
    data_file = os.path.join(output_dir, f"toxiproxy_data_{timestamp}.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{Colors.GREEN}Saved raw data to: {data_file}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}Displaying graph...{Colors.ENDC}")
    plt.show()
    
    return output_file

# Main function
def main():
    global running
    
    parser = argparse.ArgumentParser(description="Toxiproxy All-in-One Performance Measurement")
    parser.add_argument("--sender-ip", default=DEFAULT_SENDER_IP, help="Sender IP address")
    parser.add_argument("--receiver-ip", default=DEFAULT_RECEIVER_IP, help="Receiver IP address")
    parser.add_argument("--sender-port", type=int, default=DEFAULT_SENDER_PORT, help="Sender metrics port")
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT, help="Receiver metrics port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Metrics collection interval per preset in seconds")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Total duration in seconds (0 for unlimited)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}Toxiproxy All-in-One Performance Measurement{Colors.ENDC}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    print(f"Sender: {args.sender_ip}:{args.sender_port}")
    print(f"Receiver: {args.receiver_ip}:{args.receiver_port}")
    print(f"Interval per preset: {args.interval} seconds")
    print(f"Total Duration: {args.duration} seconds (0 = unlimited)")
    print(f"Output Directory: {args.output}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Checking metrics APIs...{Colors.ENDC}")
    
    sender_metrics = get_sender_metrics(args.sender_ip, args.sender_port)
    if sender_metrics:
        print(f"{Colors.GREEN}Successfully connected to sender metrics API{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}Warning: Could not connect to sender metrics API{Colors.ENDC}")
        print(f"{Colors.YELLOW}Make sure the sender is running with --metrics-port {args.sender_port}{Colors.ENDC}")
    
    receiver_metrics = get_receiver_metrics(args.receiver_ip, args.receiver_port)
    if receiver_metrics:
        print(f"{Colors.GREEN}Successfully connected to receiver metrics API{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}Warning: Could not connect to receiver metrics API{Colors.ENDC}")
        print(f"{Colors.YELLOW}Make sure the receiver is running with --metrics-port {args.receiver_port}{Colors.ENDC}")
    
    run_test_cycle(args.sender_ip, args.sender_port, args.receiver_ip, args.receiver_port, args.interval, args.duration)
    
    if len(data["timestamps"]) > 0:
        output_file = generate_graphs(args.output)
        print(f"\n{Colors.GREEN}Analysis complete!{Colors.ENDC}")
        print(f"{Colors.GREEN}Graphs have been saved in the '{args.output}' directory{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}No data collected. Cannot generate graphs.{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Program interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")