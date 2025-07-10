#!/bin/bash
#
# Setup Measurement Script for WebRTC Streaming
#
# This script sets up the measurement components on the receiver side
# to measure and graph the performance of traffic control.
#
# Usage: ./setup_measurement.sh SENDER_IP
#
# Author: Roo AI Assistant
# Date: July 2025

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

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}SETTING UP MEASUREMENT COMPONENTS${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "Sender IP: ${CYAN}$SENDER_IP${NC}"
echo -e "This script will set up the measurement components on the receiver side."
echo -e "${BLUE}======================================================${NC}"

# Check if python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: python is not installed.${NC}"
    echo -e "${RED}Please install it with: sudo apt install python3${NC}"
    exit 1
fi

# Create a directory for graphs if it doesn't exist
mkdir -p tc_performance_graphs

# Start the settings receiver in the background
echo -e "\n${GREEN}Starting tc_settings_receiver.py...${NC}"
python tc_settings_receiver.py > tc_settings_receiver.log 2>&1 &
SETTINGS_RECEIVER_PID=$!
echo -e "${GREEN}tc_settings_receiver.py started with PID $SETTINGS_RECEIVER_PID${NC}"

# Wait for the settings receiver to start
sleep 2

# Start the performance measurement
echo -e "\n${GREEN}Starting tc_performance_sync.py...${NC}"
echo -e "${YELLOW}This will measure and graph the performance.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop and generate graphs.${NC}"
sudo python tc_performance_sync.py --sender-ip $SENDER_IP --receiver-ip localhost

# Clean up when the script exits
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    kill $SETTINGS_RECEIVER_PID 2>/dev/null
    echo -e "${GREEN}Done!${NC}"
}

# Set up trap to clean up when the script exits
trap cleanup EXIT

# Wait for the user to press Ctrl+C
wait