#!/bin/bash

# Set the network interface
INTERFACE="wlp0s20f3"  # network interface (e.g., eth0, wlan0, etc.)

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. It's required for parsing JSON responses."
    echo "Install it with: sudo apt install jq"
    echo "Continuing without live data updates..."
    HAS_JQ=false
else
    HAS_JQ=true
fi

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
    
    # Only update live data if jq is installed
    if [ "$HAS_JQ" = true ]; then
        # Get current video quality settings from the sender
        echo "Getting current video quality settings..."
        resolution_scale=$(curl -s http://localhost:5000/get_resolution || echo "0.75")
        jpeg_quality=$(curl -s http://localhost:5000/get_quality || echo "85")
        frame_rate=$(curl -s http://localhost:5000/get_fps || echo "20")
        
        # Get current metrics from the sender and receiver
        echo "Getting current metrics..."
        bandwidth_usage=$(curl -s http://localhost:5000/get_metrics | jq -r '.bandwidth_usage // 0' || echo "0")
        frame_delivery_time=$(curl -s http://localhost:8081/get_metrics | jq -r '.frame_delivery_time // 0' || echo "0")
        frame_drop_rate=$(curl -s http://localhost:8081/get_metrics | jq -r '.frame_drop_rate // 0' || echo "0")
        visual_quality_score=$(curl -s http://localhost:5000/get_metrics | jq -r '.visual_quality_score // 0' || echo "0")
        smoothness_score=$(curl -s http://localhost:8081/get_metrics | jq -r '.smoothness_score // 0' || echo "0")
        
        # Determine network condition name based on parameters
        network_condition="Custom"
        if [[ "$rate" == "2mbit" && "$delay" == "150ms" && "$loss" == "3%" ]]; then
            network_condition="Poor"
        elif [[ "$rate" == "4mbit" && "$delay" == "80ms" && "$loss" == "1%" ]]; then
            network_condition="Fair"
        elif [[ "$rate" == "6mbit" && "$delay" == "40ms" && "$loss" == "0.5%" ]]; then
            network_condition="Good"
        elif [[ "$rate" == "10mbit" && "$delay" == "20ms" && "$loss" == "0%" ]]; then
            network_condition="Excellent"
        fi
        
        # Update live data
        echo "Updating live data..."
        python3 dynamic_quality_testing/update_live_data.py \
            "$network_condition" "$rate" "$delay" "$loss" \
            "$resolution_scale" "$jpeg_quality" "$frame_rate" \
            "$bandwidth_usage" "$frame_delivery_time" "$frame_drop_rate" \
            "$visual_quality_score" "$smoothness_score"
    else
        echo "Skipping live data update (jq not installed)."
    fi
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

