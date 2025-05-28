#!/bin/bash
#
# Receiver Script for Video Streaming with Traffic Control
#
# This script starts the receiver components for video streaming with traffic control.
# It starts the video receiver, settings receiver, and performance measurement.
#
# Usage: ./start_receiver.sh SENDER_IP
#
# Author: Roo AI Assistant
# Date: May 2025

# Check if sender IP is provided
if [ -z "$1" ]; then
    echo -e "\033[0;31mError: Sender IP address is required.\033[0m"
    echo -e "Usage: $0 SENDER_IP"
    exit 1
fi

SENDER_IP="$1"

# Define color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check if required commands exist
if ! command_exists python || ! command_exists python3; then
    echo -e "${RED}Error: python is not installed.${NC}"
    echo -e "${RED}Please install it with: sudo apt install python3${NC}"
    exit 1
fi

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to kill a process
kill_process() {
    pkill -f "$1" 2> /dev/null
}

# Function to clean up on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # Kill receiver
    kill_process "direct_receiver.py"
    
    # Kill settings receiver
    kill_process "tc_settings_receiver.py"
    
    # Kill performance measurement
    kill_process "tc_performance_sync.py"
    
    echo -e "${GREEN}Cleanup complete.${NC}"
    exit 0
}

# Trap Ctrl+C to clean up
trap cleanup INT

# Kill any existing processes
echo -e "${BLUE}Stopping any existing processes...${NC}"
kill_process "direct_receiver.py"
kill_process "tc_settings_receiver.py"
kill_process "tc_performance_sync.py"

# Create a directory for logs
mkdir -p logs

# Start the receiver
echo -e "\n${BLUE}======================================================${NC}"
echo -e "${BLUE}STARTING VIDEO RECEIVER${NC}"
echo -e "${BLUE}======================================================${NC}"

# Start the receiver in the background
python direct_receiver.py --display --metrics-port 8001 > logs/receiver.log 2>&1 &
RECEIVER_PID=$!

# Wait a moment for receiver to start
sleep 2

# Check if receiver is running
if ! ps -p $RECEIVER_PID > /dev/null; then
    echo -e "${RED}Error: Receiver failed to start.${NC}"
    echo -e "${RED}Check logs/receiver.log for details.${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}Receiver started successfully.${NC}"

# Start the settings receiver
echo -e "\n${BLUE}======================================================${NC}"
echo -e "${BLUE}STARTING SETTINGS RECEIVER${NC}"
echo -e "${BLUE}======================================================${NC}"

# Start the settings receiver in the background
python tc_settings_receiver.py > logs/settings_receiver.log 2>&1 &
SETTINGS_PID=$!

# Wait a moment for settings receiver to start
sleep 2

# Check if settings receiver is running
if ! ps -p $SETTINGS_PID > /dev/null; then
    echo -e "${RED}Error: Settings receiver failed to start.${NC}"
    echo -e "${RED}Check logs/settings_receiver.log for details.${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}Settings receiver started successfully.${NC}"

# Start the performance measurement
echo -e "\n${BLUE}======================================================${NC}"
echo -e "${BLUE}STARTING PERFORMANCE MEASUREMENT${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "Sender IP: ${CYAN}$SENDER_IP${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check if sudo is available
if command_exists sudo; then
    # Start the performance measurement with sudo
    sudo python tc_performance_sync.py --sender-ip "$SENDER_IP" --receiver-ip localhost &
    PERF_PID=$!
else
    echo -e "${YELLOW}Warning: sudo not available. Trying to run without sudo.${NC}"
    # Start the performance measurement without sudo
    python tc_performance_sync.py --sender-ip "$SENDER_IP" --receiver-ip localhost &
    PERF_PID=$!
fi

# Wait a moment for performance measurement to start
sleep 2

# Check if performance measurement is running
if ! ps -p $PERF_PID > /dev/null; then
    echo -e "${RED}Error: Performance measurement failed to start.${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}Performance measurement started successfully.${NC}"

# Print status
echo -e "\n${GREEN}All receiver components are running.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all components.${NC}"

# Wait for user to press Ctrl+C
while true; do
    sleep 1
done