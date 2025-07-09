# Real-Time Network Performance Measurement

This branch (`mesurement_real_preformance`) contains tools for measuring and visualizing the effects of traffic control on video streaming performance.

## Overview

The system applies different network conditions (bandwidth, latency, packet loss) on the sender side and measures their effects on the receiver side. It generates graphs showing:

1. **Commanded vs. Measured Bandwidth**: How the actual bandwidth compares to what was set by traffic control
2. **Commanded vs. Measured Latency**: How the actual latency compares to what was set by traffic control
3. **Commanded vs. Measured Packet Loss**: How the actual packet loss compares to what was set by traffic control

## Setup Instructions

### On the Sender Machine (192.168.2.120)

1. Start the video sender:
   ```bash
   python direct_sender.py --ip 192.168.2.169 --video ../video/zidane.mp4 --metrics-port 8000
   ```

2. Apply traffic control with synchronization:
   ```bash
   sudo bash auto_tc_control_sync.sh 192.168.2.169
   ```

### On the Receiver Machine (192.168.2.169)

1. Start the video receiver:
   ```bash
   python direct_receiver.py --display --metrics-port 8001
   ```

2. Run the measurement setup script:
   ```bash
   ./setup_measurement.sh 192.168.2.120
   ```

   This script will:
   - Start the tc_settings_receiver.py script to receive traffic control settings
   - Run the tc_performance_sync.py script to measure and graph the performance

## How It Works

1. The `auto_tc_control_sync.sh` script on the sender applies different network conditions in a cycle:
   - VERY POOR: 1Mbps bandwidth, 300ms latency, 5% packet loss
   - POOR: 2Mbps bandwidth, 150ms latency, 3% packet loss
   - FAIR: 4Mbps bandwidth, 80ms latency, 1% packet loss
   - GOOD: 6Mbps bandwidth, 40ms latency, 0.5% packet loss
   - EXCELLENT: 10Mbps bandwidth, 20ms latency, 0% packet loss
   - ULTRA: 50Mbps bandwidth, 1ms latency, 0% packet loss

2. The script sends these settings to the receiver via HTTP.

3. The `tc_settings_receiver.py` script on the receiver receives and stores these settings.

4. The `tc_performance_sync.py` script measures the actual performance and compares it with the commanded values.

5. After running for about 2 minutes (or when you press Ctrl+C), it generates graphs showing both commanded and measured values.

## Generated Graphs

The graphs will be saved in the `tc_performance_graphs` directory on the receiver machine. They show:

1. **Bandwidth Comparison**: Commanded bandwidth (blue) vs. Measured bandwidth (red)
2. **Latency Comparison**: Commanded latency (blue) vs. Measured latency (red)
3. **Packet Loss Comparison**: Commanded packet loss (blue) vs. Measured packet loss (red)

These graphs help visualize how well the traffic control settings are being applied and how they affect the actual performance of the video streaming.

## Troubleshooting

If you see warnings like:
```
Warning: Could not send settings to receiver. Continuing anyway.
Make sure tc_settings_receiver.py is running on the receiver.
```

Make sure the `tc_settings_receiver.py` script is running on the receiver machine before starting the traffic control on the sender.

If you don't see any graphs after running the measurement, check that both the sender and receiver metrics APIs are accessible (ports 8000 and 8001).