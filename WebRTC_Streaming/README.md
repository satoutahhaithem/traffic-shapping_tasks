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
# Terminal 3: Generate performance graphs
sudo python tc_performance_comparison.py --sender-ip localhost --receiver-ip RECEIVER_IP
```

The graphs will be saved in the `tc_performance_graphs` directory, showing:
- Bandwidth comparison (commanded vs. measured)
- Latency comparison (commanded vs. measured)
- Packet loss comparison (commanded vs. measured)

## Stopping Everything

Press Ctrl+C in each terminal, then reset traffic control:
```bash
sudo tc qdisc del dev INTERFACE root
```
Replace INTERFACE with your network interface (e.g., eth0, wlan0).