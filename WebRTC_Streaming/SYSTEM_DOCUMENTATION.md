# Video Streaming System Documentation

## Introduction: What We're Trying to Solve

### The Problem

Video streaming over networks presents several challenges:

1. **Network Variability**: Network conditions can change rapidly, affecting video quality and playback smoothness.
2. **Bandwidth Limitations**: Limited bandwidth can cause video stuttering, buffering, or quality degradation.
3. **Latency Issues**: High latency can disrupt real-time video applications and cause delays.
4. **Frame Rate Inconsistency**: Inconsistent frame rates can lead to video playing too fast or too slow.
5. **Monitoring Difficulties**: It's challenging to understand what's happening with network traffic during streaming.

### Our Solution

This video streaming system addresses these challenges by:

1. **Adaptive Buffering**: Using frame buffers on both sender and receiver to smooth out network jitter and inconsistencies.
2. **Quality Control**: Providing options to adjust video quality, resolution, and frame rate to adapt to network conditions.
3. **Traffic Monitoring**: Displaying real-time statistics about network usage, frame rates, and performance.
4. **Network Simulation**: Including a traffic control tool to simulate different network conditions for testing and optimization.
5. **Multi-threaded Architecture**: Using separate threads for reading/sending and receiving/displaying frames to ensure smooth operation.

The system is designed to be:
- **Simple**: Easy to set up and use with minimal configuration
- **Flexible**: Adaptable to different network conditions and requirements
- **Informative**: Providing detailed statistics for monitoring and troubleshooting
- **Educational**: Demonstrating principles of video streaming, buffering, and network traffic management

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Implementation Details](#implementation-details)
4. [Buffer Management](#buffer-management)
5. [Traffic Control in Depth](#traffic-control-in-depth)
6. [Deployment Instructions](#deployment-instructions)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## System Overview

The video streaming system enables real-time video transmission between computers with traffic monitoring capabilities. It consists of two main components:

1. **Sender**: Captures video from a file and sends it to the receiver
2. **Receiver**: Receives video frames and displays them

The system also includes a traffic control component that can simulate different network conditions for testing.

### Key Features

- **Real-time Video Streaming**: Stream video between computers with minimal delay
- **Traffic Monitoring**: View detailed statistics about network usage and performance
- **Adjustable Quality**: Control video quality, resolution, and frame rate
- **Frame Buffering**: Smooth out network jitter and inconsistencies
- **Network Simulation**: Test performance under different network conditions
- **Cross-platform**: Works on Linux, macOS, and Windows

### Use Cases

1. **Educational Demonstrations**: Show how video streaming works and how network conditions affect performance
2. **Network Testing**: Test video performance under different network conditions
3. **Development Testing**: Test video applications with controlled network parameters
4. **Remote Viewing**: Stream video content from one computer to another

## Architecture

### Basic System Architecture

```
+-------------+                      +---------------+
|   SENDER    |                      |   RECEIVER    |
|             |                      |               |
| +---------+ |                      | +-----------+ |
| |  Video  | |                      | |   Frame   | |
| | Reader  | |                      | | Receiver  | |
| +---------+ |                      | +-----------+ |
|      |      |                      |       |       |
| +---------+ |                      | +-----------+ |
| |  Frame  | |      Network         | |   Frame   | |
| | Buffer  | | <---------------->   | |  Buffer   | |
| +---------+ |                      | +-----------+ |
|      |      |                      |       |       |
| +---------+ |                      | +-----------+ |
| |  Frame  | |                      | |   Video   | |
| | Sender  | |                      | |  Display  | |
| +---------+ |                      | +-----------+ |
+-------------+                      +---------------+
        ^
        |
+---------------+
| Traffic       |
| Control (TC)  |
+---------------+
```

**Explanation:**
- The **Sender** reads frames from a video file, stores them in a buffer, and sends them over the network
- The **Receiver** receives frames from the network, stores them in a buffer, and displays them
- **Traffic Control** can modify network conditions to simulate different scenarios

### Data Flow

```
+-------+    +-------+    +--------+    +--------+    +--------+    +-------+
| Video | -> | Frame | -> | Encode | -> | Send   | -> | Receive| -> | Frame |
| File  |    | Read  |    | (JPEG) |    | Frame  |    | Frame  |    | Decode|
+-------+    +-------+    +--------+    +--------+    +--------+    +-------+
                                           |  ^
                                           v  |
                                        +--------+
                                        | Network|
                                        +--------+
```

**Explanation:**
- Video frames are read from a file
- Each frame is encoded using JPEG compression
- Frames are sent over the network
- The receiver gets the frames and decodes them
- Decoded frames are displayed on the receiver's screen

### Threading Model

```
SENDER                                RECEIVER
+-------------------+                +-------------------+
| Main Thread       |                | Main Thread       |
| - Read frames     |                | - Display frames  |
| - Fill buffer     |                | - Process input   |
| - Monitor stats   |                | - Monitor stats   |
+-------------------+                +-------------------+
         |                                    |
         v                                    v
+-------------------+                +-------------------+
| Sender Thread     |                | Receiver Thread   |
| - Get from buffer |                | - Receive frames  |
| - Send frames     |                | - Fill buffer     |
+-------------------+                +-------------------+
```

**Explanation:**
- Both sender and receiver use two threads to work efficiently
- On the sender:
  - Main thread reads frames from the video and adds them to a buffer
  - Sender thread takes frames from the buffer and sends them at the correct rate
- On the receiver:
  - Receiver thread gets frames from the network and adds them to a buffer
  - Main thread displays frames from the buffer at the correct rate

## Implementation Details

### Code Organization

The system consists of three main Python scripts:

1. **direct_sender.py**: Handles video capture and transmission
2. **direct_receiver.py**: Handles frame reception and display
3. **webrtc_tc_control.sh**: Bash script for network traffic control

### Key Data Structures

#### Frame Buffers

Both sender and receiver use `collections.deque` as frame buffers with a fixed maximum length:

```python
# Sender
frame_buffer = deque(maxlen=args.buffer)  # Default: 30 frames

# Receiver
frame_buffer = deque(maxlen=args.buffer)  # Default: 60 frames
```

The deque (double-ended queue) is perfect for this use case because:
- It automatically removes the oldest frames when full
- It has fast append and pop operations from both ends
- It has a fixed maximum size to prevent memory issues

#### Statistics Tracking

Both sender and receiver track various statistics using deques with fixed lengths for moving averages:

```python
# Last 30 frame sizes for averaging
frame_sizes = deque(maxlen=30)

# Last 30 frame processing times
frame_times = deque(maxlen=30)
```

These statistics help monitor performance and diagnose issues.

### Threading Implementation

#### Sender Threading Model

The sender uses two threads:

1. **Main Thread**: Reads frames from the video file and adds them to the buffer
2. **Send Thread**: Takes frames from the buffer and sends them over the network

```python
# Start sending thread
send_thread = threading.Thread(target=send_frames_thread, args=(client_socket, target_fps))
send_thread.daemon = True
send_thread.start()
```

The daemon flag ensures the thread will be terminated when the main program exits.

#### Receiver Threading Model

The receiver also uses two threads:

1. **Main Thread**: Displays frames from the buffer at the correct rate
2. **Buffer Thread**: Receives frames from the network and adds them to the buffer

```python
# Start frame buffering thread
buffer_thread = threading.Thread(target=buffer_frames, args=(client_socket,))
buffer_thread.daemon = True
buffer_thread.start()
```

## Buffer Management

The buffer system is a critical component that helps smooth out network jitter and inconsistencies. However, it requires careful management to prevent issues.

### Buffer Mechanics

```
+-------------+     +-------------+     +-------------+
| Frame       |     | Frame       |     | Network     |
| Production  | --> | Buffer      | --> | Transmission|
| (Main Thread)|    | (Queue)     |     | (Send Thread)|
+-------------+     +-------------+     +-------------+
```

1. **Frame Production Rate**: How quickly frames are read from the video file
2. **Frame Consumption Rate**: How quickly frames are sent over the network
3. **Buffer Size**: How many frames can be stored in the buffer

### Common Buffer Issues

#### 1. Buffer Overflow

When frames are produced faster than they can be consumed, the buffer fills up:

```
STATISTICS:
Buffer fullness: 30/30 (100.0%)
```

This is often accompanied by an extremely high "Actual FPS" value, which indicates the frame production rate is much higher than the consumption rate.

**Causes:**
- Network bandwidth is limited
- Receiver is not connected or is slow
- Video file is being read too quickly

**Solutions:**
- Reduce the frame production rate: `--fps 15`
- Increase the buffer size: `--buffer 60`
- Ensure the receiver is running before starting the sender
- Apply traffic control to simulate realistic network conditions

#### 2. Buffer Underflow

When frames are consumed faster than they can be produced, the buffer empties:

```
STATISTICS:
Buffer fullness: 0/30 (0.0%)
```

**Causes:**
- Video file reading is slow (e.g., from a slow disk)
- Video processing (resizing, encoding) is CPU-intensive
- Frame production rate is set too low

**Solutions:**
- Reduce the frame consumption rate: `--fps 15`
- Reduce the processing load: `--scale 0.5`
- Increase the buffer size: `--buffer 60`

### Adaptive Buffer Management

The system implements adaptive buffer management to handle varying network conditions:

```python
# Adaptive timing based on buffer state
if use_buffering:
    if len(frame_buffer) < buffer_size / 3:
        # Buffer is getting low, slow down slightly
        sleep_time += 0.003
    elif len(frame_buffer) > buffer_size * 0.8:
        # Buffer is well-filled, can be more aggressive
        sleep_time = max(0, sleep_time - 0.001)
```

This code adjusts the timing based on buffer fullness:
- If the buffer is getting low, it slows down frame consumption
- If the buffer is well-filled, it speeds up frame consumption

### Handling Extremely High FPS

If you see an extremely high "Actual FPS" value (e.g., 1000+), it indicates a timing issue:

**Causes:**
- The frame processing time is being measured incorrectly
- The system is reading frames as fast as possible without proper rate limiting
- The receiver is not connected, so frames are being read but not sent

**Solutions:**
1. **Explicitly set the FPS**:
   ```bash
   python direct_sender.py --ip RECEIVER_IP --fps 30
   ```

2. **Add a delay in the main loop**:
   ```python
   # Add a small delay to prevent CPU spinning
   time.sleep(0.001)
   ```

3. **Ensure the receiver is connected** before starting the sender

### Graceful Shutdown

To properly stop the sender or receiver:

1. Press `Ctrl+C` once and wait for the program to clean up resources
2. If it doesn't exit within a few seconds, press `Ctrl+C` again

The system implements graceful shutdown with thread joining:

```python
# Clean up resources
running = False
if send_thread and send_thread.is_alive():
    send_thread.join(timeout=1.0)

if 'cap' in locals() and cap.isOpened():
    cap.release()

client_socket.close()
```

This code:
- Sets the `running` flag to False to signal threads to stop
- Waits for threads to finish (with a timeout)
- Releases video capture resources
- Closes network sockets

## Traffic Control in Depth

The traffic control component (`webrtc_tc_control.sh`) is a crucial part of the system that allows you to simulate different network conditions. This helps test how the video streaming performs under various scenarios, such as limited bandwidth, high latency, or packet loss.

### How Traffic Control Works

The script uses Linux's Traffic Control (`tc`) utility, which is part of the iproute2 package. TC allows you to control the network traffic by applying rules to network interfaces.

```
+-------------+     +---------------+     +---------------+     +-------------+
| Application | --> | TC Queueing   | --> | Network       | --> | Destination |
| (Sender)    |     | Discipline    |     | Interface     |     | (Receiver)  |
+-------------+     +---------------+     +---------------+     +-------------+
```

### Key TC Concepts

1. **Queueing Discipline (qdisc)**: Algorithms that control how packets are queued and sent
2. **Network Emulation (netem)**: A qdisc that can emulate properties of wide area networks
3. **Rate Limiting**: Controls bandwidth usage
4. **Delay**: Adds latency to packets
5. **Packet Loss**: Randomly drops packets at a specified rate

### Traffic Control Implementation

The script implements these concepts using the following TC commands:

#### Adding Network Emulation

```bash
# Add the root qdisc for network emulation
sudo tc qdisc add dev $INTERFACE root netem
```

#### Applying Network Conditions

```bash
# Apply network conditions (bandwidth, delay, packet loss)
sudo tc qdisc change dev $INTERFACE root netem rate $rate delay $delay loss $loss
```

#### Resetting Network Conditions

```bash
# Remove all traffic control settings
sudo tc qdisc del dev $INTERFACE root
```

### Preset Network Conditions

The script includes several presets to simulate common network scenarios:

1. **Excellent**: 10mbit, 20ms delay, 0% loss
   - Simulates a high-quality home broadband connection
   - Video should play smoothly with high quality

2. **Good**: 6mbit, 40ms delay, 0.5% loss
   - Simulates a decent mobile or home connection
   - Video should play well with occasional minor issues

3. **Fair**: 4mbit, 80ms delay, 1% loss
   - Simulates a basic mobile connection or congested network
   - Video may show some quality reduction or occasional stuttering

4. **Poor**: 2mbit, 150ms delay, 3% loss
   - Simulates a poor connection or heavily congested network
   - Video will likely show significant quality reduction and stuttering

5. **Very Poor**: 1mbit, 300ms delay, 5% loss
   - Simulates a very poor connection
   - Video will show severe quality issues and frequent stuttering

### Using Traffic Control for Testing

The traffic control script is particularly useful for:

1. **Development Testing**: Test how your video application performs under different network conditions
2. **Quality Optimization**: Find the optimal quality settings for different network scenarios
3. **Buffer Tuning**: Determine the ideal buffer sizes for different latency conditions
4. **Educational Demonstrations**: Show how network conditions affect video streaming

### Traffic Control Limitations

It's important to note some limitations:

1. **Linux Only**: The TC utility is only available on Linux systems
2. **Root Required**: You need root privileges (sudo) to use TC
3. **Local Effects**: TC affects all traffic on the specified interface, not just your application
4. **Simulation Only**: While TC provides a good approximation of network conditions, it may not perfectly match real-world scenarios

## Deployment Instructions

### Prerequisites

- **Python**: Version 3.6 or higher
- **Required Packages**: OpenCV, NumPy
- **Network**: Both computers must be on the same network or have direct connectivity

### Basic Setup

#### Sender Setup

1. Identify the receiver's IP address
2. Ensure you have a video file to stream
3. Run the sender script:

```bash
python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO
```

#### Receiver Setup

1. Ensure you have a display available (for `--display` option)
2. Run the receiver script:

```bash
python direct_receiver.py --display
```

### Advanced Setup

#### Testing on a Single Computer

You can run both sender and receiver on the same computer for testing:

1. Start the receiver:
```bash
python direct_receiver.py --display
```

2. In another terminal, start the sender:
```bash
python direct_sender.py --ip localhost
```

#### Using Traffic Control (Linux only)

To simulate different network conditions:

1. Run the traffic control script on the sender:
```bash
sudo ./webrtc_tc_control.sh
```

2. Select the desired network conditions from the menu
3. Run the sender and receiver as normal

## Performance Tuning

### Optimizing for Low-Bandwidth Networks

1. Reduce the resolution:
```bash
python direct_sender.py --ip RECEIVER_IP --scale 0.5
```

2. Reduce the quality:
```bash
python direct_sender.py --ip RECEIVER_IP --quality 70
```

3. Reduce the frame rate:
```bash
python direct_sender.py --ip RECEIVER_IP --fps 15
```

### Optimizing for High-Latency Networks

1. Increase buffer sizes:
```bash
python direct_sender.py --ip RECEIVER_IP --buffer 60
python direct_receiver.py --display --buffer 120
```

### Fixing Video Speed Issues

If the video plays too fast or too slow:

```bash
# On the receiver - force a specific playback FPS
python direct_receiver.py --display --fps 30

# On the sender - control the sending rate
python direct_sender.py --ip RECEIVER_IP --fps 30
```

## Troubleshooting

### Connection Issues

**Problem**: Sender cannot connect to receiver

**Solutions**:
- Verify both computers are on the same network
- Check if the receiver is running and listening
- Verify the correct IP address is being used
- Check if any firewalls are blocking the connection

### Video Display Issues

**Problem**: No video appears on the receiver

**Solutions**:
- Ensure the `--display` flag is used on the receiver
- Check if the video file exists and is readable
- Verify OpenCV is installed correctly with GUI support
- Try a different video file format

### Performance Issues

**Problem**: Video is choppy or stutters

**Solutions**:
- Increase buffer sizes on both sender and receiver
- Reduce video quality or resolution
- Limit the frame rate
- Check network conditions using the traffic control script

### Video Speed Issues

**Problem**: Video plays too fast or too slow

**Solutions**:
- Use the `--fps` option on both sender and receiver to set the same frame rate
- Try matching the original video's frame rate
- Check if the video file has correct FPS metadata

### Buffer Issues

**Problem**: Buffer is always full (100%)

**Solutions**:
- Ensure the receiver is running and connected
- Reduce the frame production rate: `--fps 15`
- Increase the buffer size: `--buffer 60`
- Check network conditions and bandwidth

**Problem**: Extremely high "Actual FPS" reported (e.g., 1000+)

**Solutions**:
- Explicitly set the FPS: `--fps 30`
- Ensure the receiver is connected before starting the sender
- Check if there's a network bottleneck using the traffic control script

### Traffic Control Issues

**Problem**: Traffic control script doesn't work

**Solutions**:
- Make sure you're running with sudo
- Verify iproute2 is installed
- Check if the network interface is correctly identified
- Try manually specifying the network interface in the script