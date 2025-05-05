#!/bin/bash

# This script applies TC commands to both the network interface and loopback interface
# to ensure packet loss affects the video stream regardless of which interface is used

# Get the default network interface
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -n 1)
echo "Network interface: $INTERFACE"

# Function to apply settings to both interfaces
apply_to_both() {
    local delay=$1
    local rate=$2
    local loss=$3
    
    echo "Applying: delay=${delay}ms, rate=${rate}, loss=${loss}%"
    
    # Remove any existing TC settings
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    sudo tc qdisc del dev lo root 2>/dev/null
    
    # Apply to network interface
    echo "Applying to $INTERFACE..."
    sudo tc qdisc add dev $INTERFACE root netem delay ${delay}ms rate $rate loss ${loss}%
    
    # Apply to loopback interface
    echo "Applying to loopback (lo)..."
    sudo tc qdisc add dev lo root netem delay ${delay}ms rate $rate loss ${loss}%
    
    # Show current settings
    echo "Current settings on $INTERFACE:"
    sudo tc -s qdisc show dev $INTERFACE
    
    echo "Current settings on loopback (lo):"
    sudo tc -s qdisc show dev lo
}

# Check command line arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <delay_ms> <rate> <loss_percent>"
    echo "Example: $0 100 1mbit 50"
    exit 1
fi

# Apply settings from command line arguments
apply_to_both $1 $2 $3

echo "Done! TC settings applied to both interfaces."