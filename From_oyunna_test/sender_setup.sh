#!/bin/bash

# Setup script for the sender PC

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
echo "Sender PC Setup for Video Streaming System"
echo "=========================================================="
echo "Your WiFi interface: $WIFI_INTERFACE"
echo "Your IP address: $IP_ADDRESS"
echo ""

# Ask for the receiver's IP address
read -p "Enter the IP address of the receiver PC: " RECEIVER_IP

# Update the receiver_ip in video_streamer.py
if [ -f "From_oyunna_test/video_streamer.py" ]; then
    echo "Updating receiver_ip in video_streamer.py to $RECEIVER_IP..."
    sed -i "s/receiver_ip = \"[0-9.]*\"/receiver_ip = \"$RECEIVER_IP\"/" From_oyunna_test/video_streamer.py
    echo "Done!"
fi

# Update the interface in dynamic_tc_control.sh
if [ -f "From_oyunna_test/dynamic_tc_control.sh" ]; then
    echo "Updating network interface in dynamic_tc_control.sh to $WIFI_INTERFACE..."
    sed -i "s/INTERFACE=\"[^\"]*\"/INTERFACE=\"$WIFI_INTERFACE\"/" From_oyunna_test/dynamic_tc_control.sh
    echo "Done!"
fi

echo ""
echo "=========================================================="
echo "Sender PC Setup Complete!"
echo "=========================================================="
echo ""
echo "To run the video streamer:"
echo "  python3 From_oyunna_test/video_streamer.py"
echo ""
echo "To apply traffic control (network conditions):"
echo "  sudo bash From_oyunna_test/dynamic_tc_control.sh"
echo ""
echo "Your video stream will be available at:"
echo "  http://$IP_ADDRESS:5000/tx_video_feed"
echo ""
echo "The receiver should be able to view the stream at:"
echo "  http://$RECEIVER_IP:8081/rx_video_feed"
echo "=========================================================="