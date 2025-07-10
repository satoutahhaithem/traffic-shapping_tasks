import argparse
import time
import json
import os
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Default settings
DEFAULT_SENDER_IP = "localhost"
DEFAULT_RECEIVER_IP = "localhost"
DEFAULT_SENDER_PORT = 8000
DEFAULT_RECEIVER_PORT = 8001
DEFAULT_INTERVAL = 10.0  # seconds
DEFAULT_DURATION = 120  # seconds (2 minutes)
DEFAULT_OUTPUT_DIR = "./toxiproxy_performance_graphs"
DEFAULT_CYCLE_DURATION = 20  # seconds per network condition

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

# Define the network condition presets from toxiproxy_controller.py
NETWORK_PRESETS = [
    {"name": "VERY POOR", "rate": 0.5, "delay": 300, "loss": 0}, # Toxiproxy doesn't directly support packet loss, so we set it to 0
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

# Function to get commanded network conditions based on time
def get_commanded_conditions(elapsed_time):
    cycle_time = elapsed_time % (len(NETWORK_PRESETS) * DEFAULT_CYCLE_DURATION)
    preset_index = int(cycle_time / DEFAULT_CYCLE_DURATION)
    preset = NETWORK_PRESETS[preset_index]
    return {
        "name": preset["name"],
        "rate": preset["rate"],
        "delay": preset["delay"],
        "loss": preset["loss"]
    }

# Function to collect metrics
def collect_metrics(sender_ip, sender_port, receiver_ip, receiver_port, interval, duration):
    global running, data
    
    start_time = time.time()
    count = 0
    
    print(f"{Colors.GREEN}Starting metrics collection for {duration} seconds...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press Ctrl+C at any time to stop collection and generate graphs immediately{Colors.ENDC}")
    
    try:
        while running and (duration <= 0 or time.time() - start_time < duration):
            current_time = time.time() - start_time
            data["timestamps"].append(current_time)
            
            commanded = get_commanded_conditions(current_time)
            
            data["commanded"]["rate"].append(commanded["rate"])
            data["commanded"]["delay"].append(commanded["delay"])
            data["commanded"]["loss"].append(commanded["loss"])
            
            sender_metrics = get_sender_metrics(sender_ip, sender_port)
            receiver_metrics = get_receiver_metrics(receiver_ip, receiver_port)
            
            if sender_metrics and receiver_metrics:
                bandwidth_mbps = sender_metrics.get("bandwidth_usage", 0) * 8
                latency_ms = receiver_metrics.get("network_latency", 0)
                loss_rate = receiver_metrics.get("frame_drop_rate", 0)
                
                data["measured"]["bandwidth"].append(bandwidth_mbps)
                data["measured"]["latency"].append(latency_ms)
                data["measured"]["loss_rate"].append(loss_rate)
            else:
                data["measured"]["bandwidth"].append(data["measured"]["bandwidth"][-1] if data["measured"]["bandwidth"] else 0)
                data["measured"]["latency"].append(data["measured"]["latency"][-1] if data["measured"]["latency"] else 0)
                data["measured"]["loss_rate"].append(data["measured"]["loss_rate"][-1] if data["measured"]["loss_rate"] else 0)
            
            if count % 5 == 0:
                print(f"\n{Colors.BLUE}======================================================{Colors.ENDC}")
                print(f"{Colors.BLUE}TOXIPROXY PERFORMANCE - {time.strftime('%H:%M:%S')}{Colors.ENDC}")
                print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
                
                print(f"{Colors.CYAN}Commanded Network Conditions: {commanded['name']}{Colors.ENDC}")
                print(f"  Rate: {commanded['rate']} Mbps")
                print(f"  Delay: {commanded['delay']} ms")
                print(f"  Loss: {commanded['loss']}%")
                
                print(f"\n{Colors.CYAN}Measured Performance:{Colors.ENDC}")
                if sender_metrics and receiver_metrics:
                    print(f"  Bandwidth: {bandwidth_mbps:.2f} Mbps")
                    print(f"  Latency: {latency_ms:.2f} ms")
                    print(f"  Loss Rate: {loss_rate:.2f}%")
                else:
                    print(f"  {Colors.YELLOW}Could not get complete metrics{Colors.ENDC}")
            
            time.sleep(interval)
            count += 1
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Metrics collection stopped by user{Colors.ENDC}")
    
    except Exception as e:
        print(f"\n{Colors.RED}Error collecting metrics: {e}{Colors.ENDC}")
    
    finally:
        if data["timestamps"]:
            print(f"\n{Colors.GREEN}Collected {len(data['timestamps'])} data points over {data['timestamps'][-1]:.1f} seconds{Colors.ENDC}")

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
    
    parser = argparse.ArgumentParser(description="Toxiproxy Performance Measurement")
    parser.add_argument("--sender-ip", default=DEFAULT_SENDER_IP, help="Sender IP address")
    parser.add_argument("--receiver-ip", default=DEFAULT_RECEIVER_IP, help="Receiver IP address")
    parser.add_argument("--sender-port", type=int, default=DEFAULT_SENDER_PORT, help="Sender metrics port")
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT, help="Receiver metrics port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Metrics collection interval in seconds")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Total duration in seconds (0 for unlimited)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}Toxiproxy Performance Measurement{Colors.ENDC}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    print(f"Sender: {args.sender_ip}:{args.sender_port}")
    print(f"Receiver: {args.receiver_ip}:{args.receiver_port}")
    print(f"Interval: {args.interval} seconds")
    print(f"Duration: {args.duration} seconds (0 = unlimited)")
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
    
    print(f"\n{Colors.CYAN}Starting metrics collection...{Colors.ENDC}")
    
    collect_metrics(args.sender_ip, args.sender_port, args.receiver_ip, args.receiver_port, args.interval, args.duration)
    
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