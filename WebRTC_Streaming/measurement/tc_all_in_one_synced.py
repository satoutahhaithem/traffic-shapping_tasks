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
DEFAULT_INTERVAL = 20.0  # seconds per preset
DEFAULT_DURATION = 120  # seconds (2 minutes)
DEFAULT_OUTPUT_DIR = "./tc_performance_graphs"
DEFAULT_INTERFACE = "wlp0s20f3" # Change this to your default interface

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
    {"name": "VERY POOR", "rate": "500kbit", "delay": "300ms", "loss": "10%", "burst": "5k", "limit": "50"},
    {"name": "POOR", "rate": "1mbit", "delay": "200ms", "loss": "5%", "burst": "10k", "limit": "75"},
    {"name": "FAIR", "rate": "2mbit", "delay": "100ms", "loss": "2%", "burst": "15k", "limit": "100"},
    {"name": "GOOD", "rate": "5mbit", "delay": "50ms", "loss": "1%", "burst": "20k", "limit": "150"},
    {"name": "EXCELLENT", "rate": "10mbit", "delay": "20ms", "loss": "0%", "burst": "30k", "limit": "200"}
]

def run_command(command):
    """Executes a shell command."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error executing command: {command}\n{e.stderr.decode()}{Colors.ENDC}")

def setup_ifb(interface):
    """Sets up the IFB device for incoming traffic shaping."""
    print(f"{Colors.CYAN}Setting up IFB device on {interface}...{Colors.ENDC}")
    run_command("sudo modprobe ifb numifbs=1")
    run_command("sudo ip link set dev ifb0 up")
    run_command(f"sudo tc qdisc add dev {interface} handle ffff: ingress")
    run_command(f"sudo tc filter add dev {interface} parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb0")

def cleanup_ifb(interface):
    """Cleans up the IFB device."""
    print(f"{Colors.CYAN}Cleaning up IFB device...{Colors.ENDC}")
    run_command(f"sudo tc qdisc del dev {interface} ingress")
    run_command("sudo ip link set dev ifb0 down")

def apply_tc_conditions(preset):
    """Applies traffic control conditions to the ifb0 interface."""
    print(f"\n{Colors.BLUE}======================================================{Colors.ENDC}")
    print(f"{Colors.BLUE}APPLYING PRESET: {preset['name']}{Colors.ENDC}")
    print(f"{Colors.BLUE}======================================================{Colors.ENDC}")
    run_command("sudo tc qdisc del dev ifb0 root 2>/dev/null")
    run_command(f"sudo tc qdisc add dev ifb0 root handle 1: netem delay {preset['delay']} loss {preset['loss']} limit {preset['limit']}")
    run_command(f"sudo tc qdisc add dev ifb0 parent 1: handle 2: tbf rate {preset['rate']} burst {preset['burst']} latency 1000ms")

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

# Function to run the synchronized test cycle
def run_test_cycle(sender_ip, sender_port, receiver_ip, receiver_port, interval, duration, interface):
    global running, data
    
    setup_ifb(interface)
    start_time = time.time()
    
    print(f"{Colors.GREEN}Starting metrics collection for {duration} seconds...{Colors.ENDC}")
    
    try:
        while running and (duration <= 0 or time.time() - start_time < duration):
            for preset in NETWORK_PRESETS:
                apply_tc_conditions(preset)
                
                cycle_start_time = time.time()
                while time.time() - cycle_start_time < interval:
                    current_time = time.time() - start_time
                    data["timestamps"].append(current_time)
                    
                    # Store commanded values
                    rate_mbps = float(preset['rate'].replace('mbit', '').replace('kbit', '')) / (1000 if 'kbit' in preset['rate'] else 1)
                    delay_ms = float(preset['delay'].replace('ms', ''))
                    loss_percent = float(preset['loss'].replace('%', ''))
                    data["commanded"]["rate"].append(rate_mbps)
                    data["commanded"]["delay"].append(delay_ms)
                    data["commanded"]["loss"].append(loss_percent)
                    
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

                    time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Metrics collection stopped by user{Colors.ENDC}")
    
    finally:
        cleanup_ifb(interface)
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
    
    output_file = os.path.join(output_dir, f"tc_performance_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"{Colors.GREEN}Saved performance graph to: {output_file}{Colors.ENDC}")
    
    data_file = os.path.join(output_dir, f"tc_data_{timestamp}.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{Colors.GREEN}Saved raw data to: {data_file}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}Displaying graph...{Colors.ENDC}")
    plt.show()

# Main function
def main():
    global running
    
    parser = argparse.ArgumentParser(description="TC All-in-One Synchronized Measurement")
    parser.add_argument("--sender-ip", default=DEFAULT_SENDER_IP, help="Sender IP address")
    parser.add_argument("--receiver-ip", default=DEFAULT_RECEIVER_IP, help="Receiver IP address")
    parser.add_argument("--sender-port", type=int, default=DEFAULT_SENDER_PORT, help="Sender metrics port")
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT, help="Receiver metrics port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Metrics collection interval per preset in seconds")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Total duration in seconds (0 for unlimited)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs")
    parser.add_argument("--interface", default=DEFAULT_INTERFACE, help="Network interface to apply traffic shaping to")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}TC All-in-One Synchronized Measurement{Colors.ENDC}")
    print(f"{Colors.HEADER}======================================{Colors.ENDC}")
    print(f"Sender: {args.sender_ip}:{args.sender_port}")
    print(f"Receiver: {args.receiver_ip}:{args.receiver_port}")
    print(f"Interval per preset: {args.interval} seconds")
    print(f"Total Duration: {args.duration} seconds (0 = unlimited)")
    print(f"Output Directory: {args.output}")
    print(f"Interface: {args.interface}")
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
    
    run_test_cycle(args.sender_ip, args.sender_port, args.receiver_ip, args.receiver_port, args.interval, args.duration, args.interface)
    
    if len(data["timestamps"]) > 0:
        generate_graphs(args.output)
        print(f"\n{Colors.GREEN}Analysis complete!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}No data collected. Cannot generate graphs.{Colors.ENDC}")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print(f"{Colors.RED}This script must be run as root to modify network settings.{Colors.ENDC}")
        exit(1)
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Program interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
