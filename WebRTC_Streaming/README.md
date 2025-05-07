# WebRTC Video Streaming with Traffic Control

This folder contains a complete WebRTC-based video streaming solution with network traffic control capabilities. WebRTC is a modern, low-latency protocol designed specifically for real-time communication, making it ideal for video streaming applications.

## Advantages of WebRTC Over HTTP-Based Streaming

- **Lower Latency**: WebRTC is designed for real-time communication with minimal delay
- **Better Quality**: Adaptive bitrate and resolution based on network conditions
- **More Efficient**: Uses UDP instead of TCP for better performance over unreliable networks
- **Peer-to-Peer Capable**: Can establish direct connections between peers (though we use a server model here)
- **Industry Standard**: Used by major video conferencing and streaming platforms

## System Components

1. **WebRTC Sender** (`webrtc_sender.py`): Reads a video file and streams it using WebRTC
2. **WebRTC Receiver** (`webrtc_receiver.py`): Receives the WebRTC stream and displays it
3. **Traffic Control** (`webrtc_tc_control.sh`): Simulates different network conditions

## Requirements

Before running the system, make sure you have the following installed:

### Option 1: Using the Installation Script (Recommended)

```bash
cd WebRTC_Streaming
# For Python dependencies only
./install_dependencies.sh

# For all dependencies (including system packages)
sudo ./install_dependencies.sh
```

### Option 2: Manual Installation

```bash
# Install Python dependencies using requirements.txt
cd WebRTC_Streaming
pip install -r requirements.txt

# Install system dependencies (for Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev

# For traffic control
sudo apt-get install -y iproute2
```

If you encounter import errors, make sure all dependencies are properly installed:

```bash
# Alternative manual installation of Python dependencies
pip install aiortc>=1.3.2 aiohttp>=3.8.1 opencv-python>=4.5.5 numpy>=1.22.3 av>=9.2.0
```

## Running the System

### Step 1: Start the Sender

```bash
cd WebRTC_Streaming
python3 webrtc_sender.py
```

This will start the WebRTC sender server on port 8090 (default). The server will read the video file and make it available for streaming.

### Step 2: Start the Receiver

#### Option A: Standard Receiver (with OpenCV display)

```bash
cd WebRTC_Streaming
python3 webrtc_receiver.py
```

#### Option B: Headless Receiver (for systems with Wayland or display issues)

If you encounter QT/Wayland errors or other display-related issues, use the headless receiver:

```bash
cd WebRTC_Streaming
python3 headless_receiver.py
```

This version doesn't use OpenCV for display, avoiding QT/Wayland issues. The video will only be visible in the browser, not in a separate window.

This will start the WebRTC receiver server on port 8081. The receiver will display the video in an OpenCV window and also make it available in a web browser.

### Step 3: Connect to the Streams

1. **Sender Web Interface**: Open a browser and go to `http://localhost:8090`
2. **Receiver Web Interface**: Open a browser and go to `http://localhost:8091`

On the receiver web interface, enter the IP address of the sender machine and click "Connect".

### Step 4: Apply Traffic Control (Optional)

To simulate different network conditions, run the traffic control script:

```bash
cd WebRTC_Streaming
sudo bash webrtc_tc_control.sh
```

This will open an interactive menu where you can:
- Apply custom network conditions (bandwidth, delay, loss, jitter)
- Use predefined presets (perfect, good, average, mobile, poor, terrible)
- View current network statistics
- Reset network conditions

## Network Testing Scenarios

For testing how WebRTC handles different network conditions, try these presets:

1. **Perfect**: No limitations - baseline performance
2. **Good**: Slight limitations (10mbit, 20ms, 0.1% loss)
3. **Average**: Typical home internet (5mbit, 50ms, 0.5% loss)
4. **Mobile**: 4G mobile connection (2mbit, 100ms, 1% loss)
5. **Poor**: Poor connection (1mbit, 200ms, 5% loss)
6. **Terrible**: Very bad connection (500kbit, 500ms, 15% loss)

## Comparing with HTTP-Based Streaming

WebRTC provides several advantages over the HTTP-based streaming in the From_oyunna_test folder:

1. **Smoother Video**: WebRTC maintains smoother playback even under poor network conditions
2. **Lower Latency**: Typically 200-500ms end-to-end latency vs. several seconds for HTTP
3. **Better Adaptation**: Automatically adjusts to network conditions
4. **More Metrics**: Provides detailed statistics about the connection quality

## Troubleshooting

### Debugging Tools

The package includes several debugging tools to help identify and fix issues:

1. **Connection Tester**:
   ```bash
   ./test_connection.py --host <sender-ip> --port <sender-port>
   ```
   This tool checks if the receiver can connect to the sender.

2. **Receiver Debugger**:
   ```bash
   ./debug_receiver.py
   ```
   This tool checks for missing dependencies, required files, and tests the display.

### Common Issues

- **Port Already in Use Error**: If you see "address already in use" error, try using a different port:
  ```bash
  python3 webrtc_sender.py --port 8092
  python3 webrtc_receiver.py --port 8093
  ```

- **Missing Dependencies**: Make sure all required dependencies are installed:
  ```bash
  ./install_dependencies.sh
  ```

- **Connection Issues**: Make sure both machines can reach each other (check firewalls)
  
- **Video Not Appearing**:
  - Check the console output for errors
  - Run with verbose logging: `python3 webrtc_receiver.py --verbose`
  - Make sure the sender is running and accessible

- **QT/Wayland Errors**:
  - If you see errors like `Could not find the Qt platform plugin "wayland"`, use the headless receiver:
    ```bash
    python3 headless_receiver.py
    ```
  - This version doesn't use OpenCV's GUI functionality, avoiding display-related issues
  - The video will only be visible in the browser, not in a separate window

- **Poor Performance**: Try running on machines with better hardware

- **Traffic Control Not Working**: Make sure you're using the correct network interface

## Advanced Configuration

- Edit the `webrtc_sender.py` file to change video source or encoding parameters
- Edit the `webrtc_receiver.py` file to change how the video is displayed
- Edit the `webrtc_tc_control.sh` file to modify the network condition presets