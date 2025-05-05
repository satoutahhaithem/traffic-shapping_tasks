#!/bin/bash
# Start script for PC-to-PC video streaming system

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

if is_port_in_use 8000; then
    print_warning "Port 8000 is already in use. Attempting to kill the process..."
    sudo kill $(lsof -t -i:8000) 2>/dev/null || print_warning "Failed to kill process on port 8000"
    sleep 1
fi

# Get the IP address of this machine
get_ip() {
    # Try to get the IP address using the network interface
    ip_addr=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v "127.0.0.1" | head -n 1)
    if [ -z "$ip_addr" ]; then
        # Fallback to hostname -I
        ip_addr=$(hostname -I | awk '{print $1}')
    fi
    echo $ip_addr
}

LOCAL_IP=$(get_ip)
print_message "Local IP address: $LOCAL_IP"

# Function to start a component in a new terminal
start_component() {
    local title=$1
    local command=$2
    
    if command -v gnome-terminal &> /dev/null; then
        # Use gnome-terminal if available
        gnome-terminal --title="$title" -- bash -c "$command; exec bash"
    elif command -v xterm &> /dev/null; then
        # Use xterm if available
        xterm -T "$title" -e "$command; exec bash" &
    else
        # Fallback to running in background
        print_warning "No terminal emulator found. Running $title in background."
        bash -c "$command" &
    fi
    
    sleep 2
}

# Start the receiver
print_message "Starting receiver on port 5001..."
start_component "Video Receiver" "source .venv/bin/activate; python3 receiver.py"

# Start the sender
print_message "Starting sender on port 5000..."
start_component "Video Sender" "source .venv/bin/activate; python3 sender.py --receiver-ip $LOCAL_IP"

# Start the control panel server
print_message "Starting control panel server on port 8000..."
start_component "Control Panel" "source .venv/bin/activate; python3 serve_control_panel.py --sender-ip $LOCAL_IP --receiver-ip $LOCAL_IP"

print_message "All components started. The control panel should open in your browser."
print_message "If it doesn't open automatically, visit: http://localhost:8000/control_panel.html"
print_message "To stop all components, close the terminal windows or press Ctrl+C in each one."

# Print connection instructions for the other PC
echo ""
print_message "=== INSTRUCTIONS FOR THE OTHER PC ==="
print_message "On the other PC, run the following commands:"
print_message "1. Start the receiver: python3 receiver.py"
print_message "2. On this PC, update the sender: python3 sender.py --receiver-ip <other-pc-ip>"
print_message "3. Access the control panel at: http://$LOCAL_IP:8000/control_panel.html"