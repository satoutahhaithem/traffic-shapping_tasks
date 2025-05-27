# Video Streaming System

A lightweight solution for streaming video between computers with real-time traffic monitoring and network control.

## What This Solves

This system addresses several challenges in video streaming:

- **Network Variability**: Handles changing network conditions with adaptive buffering
- **Performance Monitoring**: Shows real-time statistics to understand network behavior
- **Quality Control**: Allows adjusting video quality to match available bandwidth
- **Frame Rate Issues**: Ensures consistent playback speed with precise timing control
- **Testing Capabilities**: Simulates different network conditions for development and testing

## Installation

### Using requirements.txt (Recommended)
```bash
# Install all dependencies at once
pip install -r requirements.txt
```

### Manual Installation
```bash
# Install basic requirements for video streaming
pip install opencv-python numpy

# Install requirements for the traffic control performance comparison tool
pip install matplotlib requests
```

## Quick Start Guide

### Basic Video Streaming

1. **On the Receiver PC**:
   ```bash
   python direct_receiver.py --display
   ```

2. **Find the Receiver's IP Address**:
   ```bash
   ip addr show | grep inet
   ```
   Look for an IP address like 192.168.x.x

3. **On the Sender PC**:
   ```bash
   python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4
   ```
   Replace RECEIVER_IP with the actual IP address of the receiver

### With Traffic Control (Network Simulation)

1. **On the Receiver PC**:
   ```bash
   python direct_receiver.py --display --metrics-port 8001
   ```

2. **On the Sender PC** (run these in separate terminals):
   ```bash
   # Terminal 1: Start traffic control
   sudo ./auto_tc_control.sh
   
   # Terminal 2: Start the sender
   python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000
   ```

## Traffic Control Performance Comparison

This tool compares the commanded network conditions (set by traffic control) with the actual performance measured at the receiver.

### Running the Comparison Tool

1. **On the Receiver PC**:
   ```bash
   python direct_receiver.py --display --metrics-port 8001
   ```

2. **On the Sender PC** (run these in separate terminals):
   ```bash
   # Terminal 1: Start traffic control
   sudo ./auto_tc_control.sh
   
   # Terminal 2: Start the sender
   python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000
   
   # Terminal 3: Run the performance comparison tool
   sudo python tc_performance_comparison.py --sender-ip localhost --receiver-ip RECEIVER_IP
   ```
   Replace RECEIVER_IP with the actual IP address of the receiver

3. **View the Results** (on the Sender PC):
   ```bash
   firefox tc_performance_viewer.html
   ```

### Performance Comparison Options

- `--sender-ip`: Sender IP address (default: localhost)
- `--receiver-ip`: Receiver IP address (default: 192.168.2.169)
- `--interval`: Metrics collection interval in seconds (default: 1.0)
- `--duration`: Total duration in seconds, 0 for unlimited (default: 300)
- `--output`: Output directory for graphs (default: ./tc_performance_graphs)
- `--live`: Show live graphs during collection

### Understanding the Graphs

The tool generates several types of graphs:

1. **Complete Performance Comparison**: Shows bandwidth, latency, and packet loss over time
2. **Individual Comparison Graphs**: Focus on a single metric (bandwidth, latency, or packet loss)
3. **Correlation Graphs**: Show how well the commanded values match the measured values

The blue lines represent what was commanded by the traffic control system, while the red lines show what was actually measured during video streaming.

## Command Options

### Sender Options:
- `--ip`: Receiver IP address
- `--port`: Receiver port (default: 9999)
- `--video`: Video file path
- `--quality`: JPEG quality 1-100 (default: 90)
- `--scale`: Resolution scale factor (default: 1.0)
- `--fps`: Target FPS (default: use video's FPS)
- `--buffer`: Frame buffer size (default: 5)
- `--display`: Display video locally (default: True)
- `--metrics-port`: Port for metrics API server (default: 8000)

### Receiver Options:
- `--display`: Display video (REQUIRED to see the video)
- `--fps`: Override playback FPS (default: use sender's FPS)
- `--buffer`: Frame buffer size (default: 5)
- `--low-latency`: Enable low latency mode (default: True)
- `--metrics-port`: Port for metrics API server (default: 8001)

## Traffic Control

The system includes two traffic control scripts:

1. **Manual Traffic Control** (`dynamic_tc_control.sh`):
   ```bash
   sudo ./dynamic_tc_control.sh
   ```
   Allows you to manually select network conditions from a menu.

2. **Automatic Traffic Control** (`auto_tc_control.sh`):
   ```bash
   sudo ./auto_tc_control.sh
   ```
   Automatically cycles through different network conditions every 20 seconds:
   - VERY POOR: 1Mbit, 300ms delay, 5% loss
   - POOR: 2Mbit, 150ms delay, 3% loss
   - FAIR: 4Mbit, 80ms delay, 1% loss
   - GOOD: 6Mbit, 40ms delay, 0.5% loss
   - EXCELLENT: 10Mbit, 20ms delay, 0% loss
   - ULTRA: Special ultra-low-latency configuration

## Stopping the System

1. **Stop the Sender and Receiver**:
   Press Ctrl+C in each terminal

2. **Reset Traffic Control** (on the Sender PC):
   ```bash
   sudo tc qdisc del dev INTERFACE root
   ```
   Replace INTERFACE with your network interface (e.g., eth0, wlan0)

## Troubleshooting

### Video plays too fast or too slow:
```bash
# Set specific frame rate on both sides
python direct_receiver.py --display --fps 30
python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --fps 30
```

### Video stutters or blocks:
```bash
# Increase buffer sizes
python direct_receiver.py --display --buffer 30 --low-latency=False
python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --buffer 30
```

### Cannot connect to metrics API:
- Make sure both sender and receiver are running with metrics enabled
- Verify the correct IP addresses and ports
- Check for any firewall issues

### No traffic control settings detected:
- Make sure auto_tc_control.sh is running with sudo privileges
- Check if the traffic control commands are working properly
- Verify that the network interface is correctly detected

### Missing module errors when running tc_performance_comparison.py:
If you see errors about missing modules like matplotlib, install the required dependencies:
```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install just the missing module
pip install matplotlib