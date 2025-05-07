# Dynamic Quality of Service Video Streaming

This system allows you to stream video between two computers with dynamic quality of service (QoS) control. The system automatically adjusts video quality parameters based on real-time network conditions, ensuring optimal video quality under varying network constraints.

## System Components

1. **Video Streamer** (`video_streamer.py`): Reads a video file and streams it to a receiver with dynamic quality adjustment
2. **Video Receiver** (`receive_video.py`): Receives video frames and displays them in a browser
3. **Traffic Control** (`dynamic_tc_control.sh`): Tool for simulating different network conditions
4. **Setup Scripts**: Helper scripts for configuring the system on different PCs

## Key Features

- **Dynamic Resolution Scaling**: Automatically adjusts video resolution based on network bandwidth
- **Adaptive JPEG Quality**: Changes compression level in response to network conditions
- **Frame Rate Control**: Adjusts FPS to maintain smooth playback under constraints
- **Network Condition Monitoring**: Continuously measures bandwidth, latency, and packet loss
- **Independent Local View**: Sender's local view remains high quality regardless of network conditions

## Setup Instructions

### Preparing Both PCs

1. Copy the `dynamic_qos_streaming` folder to both the sender and receiver PCs
2. Make sure both PCs have the required Python packages installed:
   ```bash
   pip install flask opencv-python numpy requests
   ```
3. Ensure both PCs have the `tc` command available (part of the `iproute2` package)

### Sender PC Setup

1. Run the sender setup script:
   ```bash
   cd /path/to/git_mesurement_tc
   ./dynamic_qos_streaming/sender_setup.sh
   ```
2. When prompted, enter the IP address of the receiver PC
3. The script will configure the system for sending video frames

### Receiver PC Setup

1. Run the receiver setup script:
   ```bash
   cd /path/to/git_mesurement_tc
   ./dynamic_qos_streaming/receiver_setup.sh
   ```
2. The script will configure the system for receiving video frames

## Running the System

### On the Receiver PC

1. Start the video receiver:
   ```bash
   python3 dynamic_qos_streaming/receive_video.py
   ```
2. The receiver will listen for frames on port 8081

### On the Sender PC

1. Start the video streamer:
   ```bash
   python3 dynamic_qos_streaming/video_streamer.py
   ```
2. The streamer will read the video file and send frames to the receiver

### Viewing the Video

1. On the receiver PC, open a web browser and go to:
   ```
   http://localhost:8081/rx_video_feed
   ```
2. On the sender PC, you can view the local stream at:
   ```
   http://localhost:5000/tx_video_feed
   ```

## Dynamic QoS in Action

The system automatically detects network conditions and adjusts video quality parameters in real-time:

1. **Good Network Conditions**:
   - Higher resolution (75-100% of original)
   - Higher JPEG quality (80-95%)
   - Higher frame rate (up to 30 FPS)

2. **Moderate Network Conditions**:
   - Medium resolution (50-75% of original)
   - Medium JPEG quality (70-80%)
   - Medium frame rate (15-30 FPS)

3. **Poor Network Conditions**:
   - Lower resolution (25-50% of original)
   - Lower JPEG quality (50-70%)
   - Lower frame rate (5-15 FPS)

## Testing with Traffic Control

You can use the included traffic control script to simulate different network conditions:

```bash
sudo bash dynamic_qos_streaming/dynamic_tc_control.sh
```

Try different settings to see how the system dynamically adapts:

1. **Start with good conditions**:
   - Rate: `10mbit`
   - Delay: `20ms`
   - Loss: `0%`

2. **Gradually worsen conditions**:
   - Rate: `3mbit`
   - Delay: `100ms`
   - Loss: `2%`

3. **Apply severe conditions**:
   - Rate: `1mbit`
   - Delay: `300ms`
   - Loss: `10%`

4. **Return to good conditions** to see the system recover

## Monitoring QoS Adaptation

The system provides real-time information about quality adjustments:

1. **Status Page**: Visit `http://localhost:5000/status` on the sender PC
2. **Console Output**: Both sender and receiver print quality adjustment information
3. **Visual Comparison**: Compare the sender's local view with the receiver's view

## Troubleshooting

- If the video doesn't appear, check that both applications are running and that the IP addresses are configured correctly
- If quality adaptation doesn't seem to work, check the console output for error messages
- Make sure you're using the correct network interface in the dynamic_tc_control.sh script
- If the system seems too aggressive or not aggressive enough in adapting quality, adjust the thresholds in the code