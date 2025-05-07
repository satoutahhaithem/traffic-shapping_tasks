#!/bin/bash

# WebRTC Traffic Control Script
# This script applies network conditions to WebRTC traffic

# Set the network interface
INTERFACE="wlp0s20f3"  # Change this to match your network interface

# Default ports for WebRTC traffic
WEBRTC_PORTS="8080:8081"  # Sender:Receiver ports

# Function to show the current network performance stats
show_stats() {
    echo "========================================================"
    echo "Network Statistics for $INTERFACE"
    echo "========================================================"
    
    # Show the interface statistics
    echo "Interface Statistics:"
    netstat -i | grep $INTERFACE
    
    # Show tc qdisc statistics
    echo -e "\nTraffic Control Settings:"
    tc -s qdisc show dev $INTERFACE
    
    # Check if netem is configured
    if tc qdisc show dev $INTERFACE | grep -q "netem"; then
        echo -e "\nNetwork Emulation Active:"
        tc qdisc show dev $INTERFACE | grep -i "rate\|delay\|loss" | sed 's/^/    /'
        
        # Show WebRTC specific stats if available
        if command -v ss &> /dev/null; then
            echo -e "\nWebRTC Connections:"
            ss -tunap | grep -E ":(${WEBRTC_PORTS//:/|})" | grep -v LISTEN
        fi
    else
        echo -e "\nNetwork emulation is NOT active."
    fi
    
    # Show current bandwidth usage
    if command -v ifstat &> /dev/null; then
        echo -e "\nCurrent Bandwidth Usage (KB/s):"
        ifstat -i $INTERFACE -b 1 1 | tail -n 1
    fi
    
    echo "========================================================"
}

# Function to apply network conditions
apply_conditions() {
    local rate="$1"     # Bandwidth rate (e.g., "1mbit")
    local delay="$2"    # Latency delay (e.g., "100ms")
    local loss="$3"     # Packet loss (e.g., "10%")
    local jitter="$4"   # Delay jitter (e.g., "20ms")
    local corrupt="$5"  # Packet corruption (e.g., "5%")

    echo "Applying WebRTC network conditions:"
    echo "  - Rate:     $rate"
    echo "  - Delay:    $delay"
    echo "  - Jitter:   $jitter"
    echo "  - Loss:     $loss"
    echo "  - Corrupt:  $corrupt"

    # First, ensure the qdisc is added to the interface if it doesn't exist yet
    if ! tc qdisc show dev $INTERFACE | grep -q "netem"; then
        # Add the root qdisc for network emulation
        sudo tc qdisc add dev $INTERFACE root handle 1: htb default 10
        sudo tc class add dev $INTERFACE parent 1: classid 1:10 htb rate 1000mbit
        sudo tc qdisc add dev $INTERFACE parent 1:10 handle 10: netem
    fi

    # Apply the new network conditions using tc
    # The order of parameters is important for tc
    local tc_cmd="sudo tc qdisc change dev $INTERFACE parent 1:10 handle 10: netem"
    
    # Add rate limiting if specified
    if [ -n "$rate" ]; then
        tc_cmd="$tc_cmd rate $rate"
    fi
    
    # Add delay and jitter if specified
    if [ -n "$delay" ]; then
        if [ -n "$jitter" ]; then
            tc_cmd="$tc_cmd delay $delay $jitter distribution normal"
        else
            tc_cmd="$tc_cmd delay $delay"
        fi
    fi
    
    # Add packet loss if specified
    if [ -n "$loss" ]; then
        tc_cmd="$tc_cmd loss $loss"
    fi
    
    # Add corruption if specified
    if [ -n "$corrupt" ]; then
        tc_cmd="$tc_cmd corrupt $corrupt"
    fi
    
    # Execute the tc command
    eval $tc_cmd
    
    # Verify the settings were applied
    echo "Verifying settings:"
    tc qdisc show dev $INTERFACE
}

# Function to reset network conditions
reset_conditions() {
    echo "Resetting network conditions on $INTERFACE..."
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null || echo "No traffic control rules to delete."
    echo "Network conditions reset."
}

# Function to apply WebRTC-specific presets
apply_preset() {
    local preset="$1"
    
    case $preset in
        "perfect")
            # Perfect conditions - no limitations
            apply_conditions "1000mbit" "0ms" "0%" "0ms" "0%"
            ;;
        "good")
            # Good conditions - slight limitations
            apply_conditions "10mbit" "20ms" "0.1%" "5ms" "0%"
            ;;
        "average")
            # Average home internet
            apply_conditions "5mbit" "50ms" "0.5%" "10ms" "0.1%"
            ;;
        "mobile")
            # Mobile 4G connection
            apply_conditions "2mbit" "100ms" "1%" "20ms" "0.2%"
            ;;
        "poor")
            # Poor connection
            apply_conditions "1mbit" "200ms" "5%" "40ms" "1%"
            ;;
        "terrible")
            # Very bad connection
            apply_conditions "500kbit" "500ms" "15%" "100ms" "2%"
            ;;
        *)
            echo "Unknown preset: $preset"
            echo "Available presets: perfect, good, average, mobile, poor, terrible"
            return 1
            ;;
    esac
    
    echo "Applied $preset network conditions preset."
}

# Interactive menu for dynamic control
menu() {
    echo "========================================================"
    echo "WebRTC Network Control"
    echo "========================================================"
    echo "1. Apply network conditions manually"
    echo "2. Apply preset network conditions"
    echo "3. Show current network statistics"
    echo "4. Reset network conditions"
    echo "5. Exit"
    echo "========================================================"
    read -p "Select an option (1-5): " option

    case $option in
        1)
            # Apply network conditions manually
            read -p "Enter the rate (e.g., '1mbit'): " rate
            read -p "Enter the delay (e.g., '100ms'): " delay
            read -p "Enter the jitter (e.g., '20ms', leave empty for none): " jitter
            read -p "Enter the loss (e.g., '10%'): " loss
            read -p "Enter the corruption (e.g., '5%', leave empty for none): " corrupt
            apply_conditions "$rate" "$delay" "$loss" "$jitter" "$corrupt"
            ;;
        2)
            # Apply preset network conditions
            echo "Available presets:"
            echo "  - perfect:  No limitations"
            echo "  - good:     Slight limitations (10mbit, 20ms, 0.1% loss)"
            echo "  - average:  Average home internet (5mbit, 50ms, 0.5% loss)"
            echo "  - mobile:   Mobile 4G connection (2mbit, 100ms, 1% loss)"
            echo "  - poor:     Poor connection (1mbit, 200ms, 5% loss)"
            echo "  - terrible: Very bad connection (500kbit, 500ms, 15% loss)"
            read -p "Enter preset name: " preset
            apply_preset "$preset"
            ;;
        3)
            # Show current stats
            show_stats
            ;;
        4)
            # Reset network conditions
            reset_conditions
            ;;
        5)
            echo "Exiting the script."
            exit 0
            ;;
        *)
            echo "Invalid option. Please select again."
            ;;
    esac
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges to modify network settings."
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Check if tc command is available
if ! command -v tc &> /dev/null; then
    echo "Error: tc command not found. Please install the iproute2 package."
    exit 1
fi

# Check if the interface exists
if ! ip link show $INTERFACE &> /dev/null; then
    echo "Error: Network interface $INTERFACE not found."
    echo "Available interfaces:"
    ip -o link show | awk -F': ' '{print $2}'
    echo ""
    read -p "Enter the name of your network interface: " INTERFACE
    
    if ! ip link show $INTERFACE &> /dev/null; then
        echo "Error: Network interface $INTERFACE not found. Exiting."
        exit 1
    fi
fi

# Main loop for dynamic control
while true; do
    menu
    echo ""
done