#!/bin/bash

echo "Installing WebRTC Streaming Dependencies"
echo "========================================"

# Check if running as root for system dependencies
if [ "$EUID" -ne 0 ]; then
    echo "Note: For system dependencies, this script should be run with sudo."
    echo "We'll install Python dependencies now, but you may need to run with sudo later."
    SUDO_AVAILABLE=false
else
    SUDO_AVAILABLE=true
fi

# Install Python dependencies
echo -e "\n1. Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing Python dependencies. Trying alternative method..."
    pip install aiortc>=1.3.2 aiohttp>=3.8.1 opencv-python>=4.5.5 numpy>=1.22.3 av>=9.2.0
    
    if [ $? -ne 0 ]; then
        echo "Failed to install Python dependencies. Please install them manually:"
        echo "pip install aiortc>=1.3.2 aiohttp>=3.8.1 opencv-python>=4.5.5 numpy>=1.22.3 av>=9.2.0"
        exit 1
    fi
fi

# Install system dependencies if running as root
if [ "$SUDO_AVAILABLE" = true ]; then
    echo -e "\n2. Installing system dependencies..."
    apt-get update
    apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev iproute2
    
    if [ $? -ne 0 ]; then
        echo "Failed to install system dependencies. Please install them manually:"
        echo "sudo apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev iproute2"
        exit 1
    fi
else
    echo -e "\n2. Skipping system dependencies (not running as root)"
    echo "Please run the following command to install system dependencies:"
    echo "sudo apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev iproute2"
fi

echo -e "\nDependencies installation completed successfully!"
echo "You can now run the WebRTC streaming system:"
echo "- Sender: python3 webrtc_sender.py"
echo "- Receiver: python3 webrtc_receiver.py"