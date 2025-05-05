# PC-to-PC Video Streaming with QoS Control

This system allows you to stream video from one PC to another over a network, with the ability to control Quality of Service (QoS) parameters.

## System Architecture

The system consists of the following components:

1. **Sender** (runs on the source PC):
   - Captures video from a file or camera
   - Streams video frames to the receiver
   - Applies QoS parameters as directed

2. **Receiver** (runs on the destination PC):
   - Receives video frames from the sender
   - Displays the video
   - Provides statistics on streaming quality

3. **QoS Controller** (can run on either PC):
   - Controls network parameters (bandwidth, delay, packet loss)
   - Provides a web interface for adjusting parameters
   - Monitors streaming performance

## Requirements

- Python 3.6+ on both PCs
- Flask, OpenCV, NumPy, and Requests libraries
- Network connectivity between the two PCs
- Linux with tc (Traffic Control) utility for QoS control (optional)

## Setup Instructions

### 1. Install Dependencies

On both PCs, install the required Python libraries:

```bash
pip install flask flask-cors opencv-python numpy requests
```

### 2. Configure Network Settings

1. Ensure both PCs are on the same network or can reach each other
2. Note the IP address of the sender PC (use `ifconfig` or `ip addr` to find it)
3. Update the `SENDER_IP` in `receiver.py` to match the sender's IP address

### 3. Start the System

#### On the Receiver PC:

```bash
python receiver.py
```

This will start the receiver on port 5001.

#### On the Sender PC:

```bash
python sender.py
```

This will start the sender on port 5000 and begin streaming to the receiver.

#### For QoS Control:

```bash
python qos_controller.py
```

This will start the QoS controller on port 5002 and open the control panel in your browser.

## Using the System

1. Access the control panel at http://localhost:5002/control_panel.html
2. Use the sliders to adjust bandwidth, delay, and packet loss
3. Monitor the streaming quality metrics
4. View the video stream at http://receiver-ip:5001/video_feed

## Troubleshooting

- If the video doesn't appear, check network connectivity between the PCs
- Ensure all required ports (5000, 5001, 5002) are open in any firewalls
- Check the console output for error messages
- Verify that the sender IP address is correctly configured in the receiver

## Advanced Configuration

See the comments in each Python file for additional configuration options:
- Change video source in `sender.py`
- Adjust buffer sizes and timeouts in `receiver.py`
- Modify QoS parameter ranges in `qos_controller.py`