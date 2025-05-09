#!/bin/bash

# Setup script for the sender PC in the Dynamic Quality Testing system

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
echo "Sender PC Setup for Dynamic Quality Testing System"
echo "=========================================================="
echo "Your WiFi interface: $WIFI_INTERFACE"
echo "Your IP address: $IP_ADDRESS"
echo ""

# Ask for the receiver's IP address
read -p "Enter the IP address of the receiver PC: " RECEIVER_IP

# Update the receiver_ip in video_streamer.py
if [ -f "dynamic_quality_testing/video_streamer.py" ]; then
    echo "Updating receiver_ip in video_streamer.py to $RECEIVER_IP..."
    sed -i "s/'http:\/\/127.0.0.1:8081\/receive_video'/'http:\/\/$RECEIVER_IP:8081\/receive_video'/" dynamic_quality_testing/video_streamer.py
    echo "Done!"
fi

# Update the interface in dynamic_tc_control.sh
if [ -f "dynamic_quality_testing/dynamic_tc_control.sh" ]; then
    echo "Updating network interface in dynamic_tc_control.sh to $WIFI_INTERFACE..."
    sed -i "s/INTERFACE=\"[^\"]*\"/INTERFACE=\"$WIFI_INTERFACE\"/" dynamic_quality_testing/dynamic_tc_control.sh
    echo "Done!"
fi

# Update the interface in run_quality_tests.py
if [ -f "dynamic_quality_testing/run_quality_tests.py" ]; then
    echo "Updating network interface in run_quality_tests.py to $WIFI_INTERFACE..."
    sed -i "s/INTERFACE = \"[^\"]*\"/INTERFACE = \"$WIFI_INTERFACE\"/" dynamic_quality_testing/run_quality_tests.py
    echo "Done!"
fi

# Install required dependencies
echo "Installing required dependencies..."
pip install matplotlib numpy requests flask opencv-python

echo ""
echo "=========================================================="
echo "Sender PC Setup Complete!"
echo "=========================================================="
echo ""
echo "To run the video streamer:"
echo "  python3 dynamic_quality_testing/video_streamer.py"
echo ""
echo "To apply traffic control (network conditions):"
echo "  sudo bash dynamic_quality_testing/dynamic_tc_control.sh"
echo ""
echo "To run automated quality tests:"
echo "  python3 dynamic_quality_testing/run_quality_tests.py"
echo ""
echo "Your video stream will be available at:"
echo "  http://$IP_ADDRESS:5000/tx_video_feed"
echo ""
echo "The receiver should be able to view the stream at:"
echo "  http://$RECEIVER_IP:8081/rx_video_feed"
echo "=========================================================="