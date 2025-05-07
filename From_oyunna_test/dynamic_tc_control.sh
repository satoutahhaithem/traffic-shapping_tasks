#!/bin/bash

# Set the network interface
INTERFACE="wlp0s20f3"  # WiFi interface as specified

# Function to show the current network performance stats
show_stats() {
    echo "Monitoring network statistics for $INTERFACE"
    echo "----------------------------------------------"
    # Show the interface statistics: packets, bytes, and errors
    netstat -i | grep $INTERFACE
    # Show tc qdisc statistics (e.g., packet loss, delay, etc.)
    echo "Current TC settings:"
    tc -s qdisc show dev $INTERFACE
    
    # Check if netem is configured
    if tc qdisc show dev $INTERFACE | grep -q "netem"; then
        echo ""
        echo "Network emulation is ACTIVE with the following parameters:"
        tc qdisc show dev $INTERFACE | grep -i "rate\|delay\|loss" | sed 's/^/    /'
        echo ""
        echo "These settings should affect the video quality. If you don't see any difference:"
        echo "1. Make sure you're using a high enough delay (try 1000ms or more)"
        echo "2. Make sure you're using a low enough rate (try 500kbit or less)"
        echo "3. Make sure you're using a high enough loss (try 20% or more)"
        echo ""
        echo "IMPORTANT: Always include units when entering values:"
        echo "- Rate: Use 'kbit', 'mbit', or 'gbit' (e.g., '1mbit')"
        echo "- Delay: Use 'ms' or 's' (e.g., '100ms')"
        echo "- Loss: Use '%' (e.g., '10%')"
    else
        echo ""
        echo "Network emulation is NOT ACTIVE. Use option 1 to set network conditions."
    fi
}

# Function to apply network conditions dynamically
apply_conditions() {
    local rate="$1"     # Bandwidth rate (e.g., "1mbit")
    local delay="$2"    # Latency delay (e.g., "100ms")
    local loss="$3"     # Packet loss (e.g., "10%")

    echo "Applying network conditions: Rate=$rate, Delay=$delay, Loss=$loss"

    # First, ensure the qdisc is added to the interface if it doesn't exist yet
    if ! tc qdisc show dev $INTERFACE | grep -q "netem"; then
        # Add the root qdisc for network emulation if not already added
        sudo tc qdisc add dev $INTERFACE root netem
    fi

    # Apply the new network conditions using tc
    sudo tc qdisc change dev $INTERFACE root netem rate $rate delay $delay loss $loss
}

# Function to reset network conditions (remove tc configuration)
reset_conditions() {
    echo "Resetting network conditions."
    sudo tc qdisc del dev $INTERFACE root
}

# Interactive menu for dynamic control
menu() {
    echo "----------------------------"
    echo "Dynamic Network Control (TC)"
    echo "----------------------------"
    echo "1. Set network conditions (Rate, Delay, Loss)"
    echo "2. Show current stats"
    echo "3. Reset network conditions"
    echo "4. Exit"
    echo "----------------------------"
    read -p "Select an option (1-4): " option

    case $option in
        1)
            # Set network conditions
            read -p "Enter the rate (e.g., '1mbit'): " rate
            read -p "Enter the delay (e.g., '100ms'): " delay
            read -p "Enter the loss (e.g., '10%'): " loss
            apply_conditions "$rate" "$delay" "$loss"
            ;;
        2)
            # Show current stats
            show_stats
            ;;
        3)
            # Reset network conditions
            reset_conditions
            ;;
        4)
            echo "Exiting the script."
            exit 0
            ;;
        *)
            echo "Invalid option. Please select again."
            ;;
    esac
}

# Main loop for dynamic control
while true; do
    menu
done

