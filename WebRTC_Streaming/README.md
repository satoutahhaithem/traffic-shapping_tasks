# Video Streaming with Traffic Control

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

### On Receiver PC:
```bash
python direct_receiver.py --display --metrics-port 8001
```

### On Sender PC:
Run these in separate terminals:

```bash
# Terminal 1: Start traffic control
sudo ./auto_tc_control.sh

# Terminal 2: Start the sender
python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000
```
Replace RECEIVER_IP with the receiver's IP address.

## Plotting Performance Graphs

On the Sender PC, run:
```bash
# Terminal 3: Generate and display performance graphs
sudo python tc_performance_comparison.py --sender-ip localhost --receiver-ip RECEIVER_IP
```

### What Happens When You Run This Command:

1. **Data Collection Phase (2 minutes):**
   - The script connects to the sender and receiver metrics APIs
   - It checks the current traffic control settings
   - Every 10 seconds, it collects:
     * Commanded values from traffic control (bandwidth, delay, loss)
     * Measured values from the sender and receiver
   - You'll see real-time statistics in the terminal showing both commanded and measured values
   - The auto_tc_control.sh script cycles through different network conditions every 20 seconds

2. **Graph Generation Phase:**
   - After 2 minutes (or when you press Ctrl+C), data collection stops
   - The script processes the collected data
   - It generates a graph with three plots comparing commanded vs. measured values
   - The graph is displayed directly on your screen
   - The graph is also saved as a PNG file in the tc_performance_graphs directory
   - The raw data is saved as a JSON file for later analysis

3. **What the Graph Shows:**
   - **Top plot**: Bandwidth comparison (blue = commanded rate, red = measured bandwidth)
   - **Middle plot**: Latency comparison (blue = commanded delay, red = measured latency)
   - **Bottom plot**: Packet loss comparison (blue = commanded loss, red = measured loss rate)
   - The x-axis shows time in seconds
   - The blue lines show what was commanded by the traffic control system
   - The red lines show what was actually measured during streaming

### Options

To change the data collection duration:
```bash
# Collect data for 30 seconds
sudo python tc_performance_comparison.py --duration 30 --receiver-ip RECEIVER_IP

# Collect data until manually stopped with Ctrl+C
sudo python tc_performance_comparison.py --duration 0 --receiver-ip RECEIVER_IP
```

## Stopping Everything

Press Ctrl+C in each terminal, then reset traffic control:
```bash
sudo tc qdisc del dev INTERFACE root
```
Replace INTERFACE with your network interface (e.g., eth0, wlan0).