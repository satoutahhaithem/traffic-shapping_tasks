# Dynamic Quality Testing System

This system allows you to stream video between two computers with dynamic quality testing. The system automatically varies video quality parameters and measures their impact on performance under different network conditions.

## System Components

1. **Video Streamer** (`video_streamer.py`): Reads a video file and streams it to a receiver with dynamic quality adjustment
2. **Video Receiver** (`receive_video.py`): Receives video frames and displays them in a browser
3. **Traffic Control** (`dynamic_tc_control.sh`): Tool for simulating different network conditions

## Key Features

- **Automatic Quality Testing**: Systematically tests different quality settings
- **Network Condition Simulation**: Applies various bandwidth, delay, and packet loss settings
- **Performance Measurement**: Records metrics for each quality/network combination
- **Comparative Analysis**: Helps identify optimal quality settings for different network conditions

## Setup Instructions

### Preparing Both PCs

1. Copy the `dynamic_quality_testing` folder to both the sender and receiver PCs
2. Make sure both PCs have the required Python packages installed:
   ```bash
   pip install flask opencv-python numpy requests matplotlib
   ```
3. Ensure both PCs have the `tc` command available (part of the `iproute2` package)

### Sender PC Setup

1. Edit the `video_streamer.py` file to set the receiver's IP address:
   ```python
   receiver_ip = "192.168.x.x"  # Change to the IP address of the receiver PC
   ```

2. Make sure the video file path is correct:
   ```python
   video_path = '/path/to/your/video/file.mp4'
   ```

### Receiver PC Setup

No special configuration is needed for the receiver PC.

## Running the System

### On the Receiver PC

1. Start the video receiver:
   ```bash
   python3 dynamic_quality_testing/receive_video.py
   ```
2. The receiver will listen for frames on port 8081

### On the Sender PC

1. Start the video streamer:
   ```bash
   python3 dynamic_quality_testing/video_streamer.py
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

## Dynamic Quality Testing

The system includes functionality to automatically test different quality settings:

1. **Resolution Scaling**: Tests different scaling factors (25%, 50%, 75%, 100%)
2. **JPEG Quality**: Tests different compression levels (50%, 65%, 80%, 95%)
3. **Frame Rate**: Tests different FPS settings (10, 15, 20, 30)

### Running Automated Tests

To run the automated quality tests:

```bash
python3 dynamic_quality_testing/run_quality_tests.py
```

This script will:
1. Systematically vary quality parameters
2. Apply different network conditions using traffic control
3. Measure performance metrics for each combination
4. Generate a report with the results

### Test Matrix

The testing system uses the following matrix of settings:

| Resolution Scale | JPEG Quality | Frame Rate | Network Condition |
|------------------|-------------|------------|-------------------|
| 25%              | 50%         | 10 FPS     | Poor (1mbit, 200ms, 5% loss) |
| 50%              | 65%         | 15 FPS     | Fair (3mbit, 100ms, 2% loss) |
| 75%              | 80%         | 20 FPS     | Good (5mbit, 50ms, 1% loss) |
| 100%             | 95%         | 30 FPS     | Excellent (10mbit, 20ms, 0% loss) |

### Metrics Collected

For each combination, the system collects:

- **Bandwidth Usage**: How much network bandwidth is consumed
- **Frame Delivery Time**: How long it takes to deliver each frame
- **Frame Drop Rate**: Percentage of frames that fail to deliver
- **Visual Quality Score**: Estimated quality based on resolution and compression
- **Smoothness Score**: Estimated smoothness based on frame rate and delivery time

## Applying Network Conditions

You can manually apply different network conditions using the included traffic control script:

```bash
sudo bash dynamic_quality_testing/dynamic_tc_control.sh
```

This interactive script allows you to:
1. Set custom network conditions (bandwidth, delay, packet loss)
2. Monitor the current network statistics
3. Reset network conditions to normal

## Troubleshooting

- If the video doesn't appear, check that both applications are running and that the IP addresses are configured correctly
- If traffic control doesn't seem to have an effect, try more extreme settings
- Make sure you're using the correct network interface in the dynamic_tc_control.sh script
- Check the terminal output for any error messages