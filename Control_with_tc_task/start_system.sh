#!/bin/bash

# This script starts all components of the video streaming system with network simulation

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install flask flask-cors opencv-python numpy requests
else
    print_message "Virtual environment found."
fi

# Activate virtual environment
source .venv/bin/activate
print_message "Virtual environment activated."

# Function to check if a port is in use
is_port_in_use() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Check if ports are already in use and kill processes if needed
if is_port_in_use 5000; then
    print_warning "Port 5000 is already in use. Attempting to kill the process..."
    sudo kill $(lsof -t -i:5000) 2>/dev/null || print_warning "Failed to kill process on port 5000"
    sleep 1
fi

if is_port_in_use 5001; then
    print_warning "Port 5001 is already in use. Attempting to kill the process..."
    sudo kill $(lsof -t -i:5001) 2>/dev/null || print_warning "Failed to kill process on port 5001"
    sleep 1
fi

if is_port_in_use 5002; then
    print_warning "Port 5002 is already in use. Attempting to kill the process..."
    sudo kill $(lsof -t -i:5002) 2>/dev/null || print_warning "Failed to kill process on port 5002"
    sleep 1
fi

# Control panel server will try multiple ports (8000, 8001, etc.) so we don't need to kill it

# Start the receiver in a new terminal
print_message "Starting video receiver on port 5001..."
gnome-terminal --title="Video Receiver" -- bash -c "source .venv/bin/activate; python3 receive_video.py; exec bash"
sleep 2

# Start the streamer in a new terminal
print_message "Starting video streamer on port 5000..."
gnome-terminal --title="Video Streamer" -- bash -c "source .venv/bin/activate; python3 video_streamer.py; exec bash"
sleep 2

# Start the TC API server in a new terminal
print_message "Starting TC API server on port 5002 (requires sudo)..."
gnome-terminal --title="TC API Server" -- bash -c "./run_tc_api.sh; exec bash"
sleep 2

# Start the control panel in a new terminal
print_message "Starting control panel server on port 8000..."
gnome-terminal --title="Control Panel" -- bash -c "source .venv/bin/activate; python3 serve_control_panel_updated.py; exec bash"

print_message "All components started. The control panel should open in your browser."
print_message "If it doesn't open automatically, visit: http://localhost:8000/control_updated.html"
print_message "To stop all components, close the terminal windows or press Ctrl+C in each one."