#!/bin/bash

# Setup script for the receiver PC

# Get the IP address of the WiFi interface
WIFI_INTERFACE="wlp0s20f3"  # Change this if your interface has a different name
IP_ADDRESS=$(ip -4 addr show $WIFI_INTERFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

if [ -z "$IP_ADDRESS" ]; then
    echo "Could not find IP address for interface $WIFI_INTERFACE"
    echo "Available interfaces:"
    ip -o -4 addr show | awk '{print $2, $4}'
    echo ""
    read -p "Enter the name of your network interface: " WIFI_INTERFACE
    IP_ADDRESS=$(ip -4 addr show $WIFI_INTERFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    
    if [ -z "$IP_ADDRESS" ]; then
        echo "Still could not find IP address. Please check your network configuration."
        exit 1
    fi
fi

echo "=========================================================="
echo "Receiver PC Setup for Video Streaming System"
echo "=========================================================="
echo "Your WiFi interface: $WIFI_INTERFACE"
echo "Your IP address: $IP_ADDRESS"
echo ""

# Update the interface in dynamic_tc_control.sh
if [ -f "From_oyunna_test/dynamic_tc_control.sh" ]; then
    echo "Updating network interface in dynamic_tc_control.sh to $WIFI_INTERFACE..."
    sed -i "s/INTERFACE=\"[^\"]*\"/INTERFACE=\"$WIFI_INTERFACE\"/" From_oyunna_test/dynamic_tc_control.sh
    echo "Done!"
fi

echo ""
echo "=========================================================="
echo "Receiver PC Setup Complete!"
echo "=========================================================="
echo ""
echo "To run the video receiver:"
echo "  python3 From_oyunna_test/receive_video.py"
echo ""
echo "To apply traffic control (network conditions):"
echo "  sudo bash From_oyunna_test/dynamic_tc_control.sh"
echo ""
echo "Your video receiver will be listening at:"
echo "  http://$IP_ADDRESS:8081/receive_video"
echo ""
echo "You can view the received video stream at:"
echo "  http://$IP_ADDRESS:8081/rx_video_feed"
echo ""
echo "Make sure the sender PC is configured to send frames to your IP address: $IP_ADDRESS"
echo "=========================================================="