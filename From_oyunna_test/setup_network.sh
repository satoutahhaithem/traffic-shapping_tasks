#!/bin/bash

# Script to help set up the video streaming system for network use

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
echo "Network Setup for Video Streaming System"
echo "=========================================================="
echo "Your WiFi interface: $WIFI_INTERFACE"
echo "Your IP address: $IP_ADDRESS"
echo ""
echo "To run the system on a network:"
echo ""
echo "1. On the receiver machine:"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   python3 From_oyunna_test/receive_video.py"
echo ""
echo "2. On the streamer machine, edit video_streamer.py:"
echo "   Change 'receiver_ip = \"127.0.0.1\"' to 'receiver_ip = \"$IP_ADDRESS\"'"
echo "   (This has been done automatically if you're running this on the receiver machine)"
echo ""
echo "3. On the streamer machine:"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   python3 From_oyunna_test/video_streamer.py"
echo ""
echo "4. To test network conditions (on either machine):"
echo "   cd ~/Bureau/git_mesurement_tc"
echo "   sudo bash From_oyunna_test/dynamic_tc_control.sh"
echo ""
echo "5. Access the streams in a web browser:"
echo "   Receiver stream: http://$IP_ADDRESS:8081/rx_video_feed"
echo "   Streamer local view: http://[streamer-ip]:5000/tx_video_feed"
echo "=========================================================="

# Update the receiver_ip in video_streamer.py if this is the receiver machine
if [ -f "From_oyunna_test/video_streamer.py" ]; then
    echo "Updating receiver_ip in video_streamer.py to $IP_ADDRESS..."
    sed -i "s/receiver_ip = \"127.0.0.1\"/receiver_ip = \"$IP_ADDRESS\"/" From_oyunna_test/video_streamer.py
    echo "Done!"
fi

echo ""
echo "Setup complete! Follow the instructions above to run the system."