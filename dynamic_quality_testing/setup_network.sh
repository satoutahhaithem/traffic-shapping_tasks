#!/bin/bash

# Script to help set up the Dynamic Quality Testing system for network use

# Get the IP address of the WiFi interface
WIFI_INTERFACE="wlp0s20f3"
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
echo "Network Setup for Dynamic Quality Testing System"
echo "=========================================================="
echo "Your WiFi interface: $WIFI_INTERFACE"
echo "Your IP address: $IP_ADDRESS"
echo ""
echo "To run the system on a network:"
echo ""
echo "1. On the receiver machine:"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   python3 dynamic_quality_testing/receive_video.py"
echo ""
echo "2. On the streamer machine:"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   python3 dynamic_quality_testing/video_streamer.py"
echo ""
echo "3. To test network conditions (on either machine):"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   sudo bash dynamic_quality_testing/dynamic_tc_control.sh"
echo ""
echo "4. To run automated quality tests (on the streamer machine):"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   python3 dynamic_quality_testing/run_quality_tests.py"
echo ""
echo "5. Access the interfaces in a web browser:"
echo "   Receiver stream: http://$IP_ADDRESS:8081/rx_video_feed"
echo "   Receiver status: http://$IP_ADDRESS:8081/status"
echo "   Streamer local view: http://[streamer-ip]:5000/tx_video_feed"
echo "   Streamer quality controls: http://[streamer-ip]:5000/quality_controls"
echo "   Streamer status: http://[streamer-ip]:5000/status"
echo "=========================================================="

# Update the video_streamer.py to send frames to this IP address if this is the receiver machine
if [ -f "dynamic_quality_testing/video_streamer.py" ]; then
    echo "Updating video_streamer.py to send frames to $IP_ADDRESS..."
    sed -i "s/'http:\/\/127.0.0.1:8081\/receive_video'/'http:\/\/$IP_ADDRESS:8081\/receive_video'/" dynamic_quality_testing/video_streamer.py
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
echo "Setup complete! Follow the instructions above to run the system."