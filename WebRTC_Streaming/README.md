# Video Streaming System

A lightweight solution for streaming video between computers with real-time traffic monitoring and network control.

## What This Solves

This system addresses several challenges in video streaming:

- **Network Variability**: Handles changing network conditions with adaptive buffering
- **Performance Monitoring**: Shows real-time statistics to understand network behavior
- **Quality Control**: Allows adjusting video quality to match available bandwidth
- **Frame Rate Issues**: Ensures consistent playback speed with precise timing control
- **Testing Capabilities**: Simulates different network conditions for development and testing

## Complete System Workflow

For the best experience with the full system including performance monitoring and visualization:

1. **Start the Receiver with Metrics Enabled**
   ```bash
   python direct_receiver.py --display --metrics-port 8001
   ```

2. **Start the Automatic Traffic Control**
   ```bash
   sudo ./auto_tc_control.sh
   ```
   This will automatically cycle through different network conditions every 20 seconds.

3. **Start the Sender with Metrics Enabled**
   ```bash
   python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO --metrics-port 8000
   ```
   Replace RECEIVER_IP with the actual IP address and PATH_TO_VIDEO with your video file path.

4. **Start the Performance Monitor**
   ```bash
   python performance_monitor.py --live
   ```
   This will collect metrics, display live graphs, and save performance data.

5. **View the Results in the Performance Viewer**
   ```bash
   firefox performance_viewer.html
   ```
   This will open the web-based interface for analyzing the performance graphs.

6. **When Finished**
   - Press Ctrl+C in each terminal to stop the respective processes
   - Make sure to reset network conditions: `sudo tc qdisc del dev INTERFACE root`

## Step-by-Step Execution Guide

### Basic Setup (Without Traffic Control)

1. **Prepare Both Computers**
   - Ensure both computers are on the same network
   - Install required packages on both computers:
     ```bash
     pip install opencv-python numpy
     ```

2. **Start the Receiver First**
   - On the receiver computer, open a terminal
   - Navigate to the WebRTC_Streaming directory
   - Run the receiver script with the display flag:
     ```bash
     python direct_receiver.py --display
     ```
   - You should see a message: "Listening on 0.0.0.0:9999..."

3. **Find the Receiver's IP Address**
   - On the receiver computer, open a new terminal
   - Run the following command to find the IP address:
     ```bash
     ip addr show | grep inet
     ```
   - Look for an IP address like 192.168.x.x (local network)
   - Note this IP address for the next step

4. **Start the Sender**
   - On the sender computer, open a terminal
   - Navigate to the WebRTC_Streaming directory
   - Run the sender script with the receiver's IP:
     ```bash
     python direct_sender.py --ip 192.168.2.169 --video ../video/zidane.mp4
     ```
     (Replace RECEIVER_IP with the actual IP address and PATH_TO_VIDEO with your video file path)
   - You should see "Connected to receiver" and statistics in the terminal
   - You should also see a window showing the video on the sender's PC
   - The video on the sender and receiver should be synchronized
   - If you don't want to see the video on the sender's PC, use the --display=False flag:
     ```bash
     python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO --display=False
     ```
   - If the videos are not synchronized, adjust the sync delay:
     ```bash
     python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO --sync-delay 0.5
     ```

5. **Observe the Video Stream**
   - On the receiver computer, a window should appear showing the video
   - Both terminals will display real-time traffic statistics

### Advanced Setup (With Dynamic Traffic Control)

1. **Follow Steps 1-2 from Basic Setup**
   - Prepare both computers
   - Start the receiver

2. **Set Up Dynamic Traffic Control on Sender**
   - On the sender computer, open a terminal
   - Navigate to the WebRTC_Streaming directory
   - You have two options for traffic control:
     
     **Option 1: Manual Traffic Control**
     - Make the traffic control script executable:
       ```bash
       chmod +x dynamic_tc_control.sh
       ```
     - Run the traffic control script with sudo:
       ```bash
       sudo ./dynamic_tc_control.sh
       ```
     - The script will detect your network interface or ask you to select one
     - You'll see a menu with options
     
     **Option 2: Automatic Traffic Control (Recommended)**
     - Make the automatic traffic control script executable:
       ```bash
       chmod +x auto_tc_control.py
       ```
     - Run the automatic traffic control script with sudo:
       ```bash
       sudo python3 auto_tc_control.py
       ```
     - The script will automatically detect your network interface
     - It will continuously monitor network conditions and adjust them dynamically

3. **Apply Network Conditions**
   - From the menu, select option 2 (Apply preset network conditions)
   - Choose a preset (1-5) to simulate different network conditions:
     - 1: Excellent (10mbit, 20ms delay, 0% loss)
     - 2: Good (6mbit, 40ms delay, 0.5% loss)
     - 3: Fair (4mbit, 80ms delay, 1% loss)
     - 4: Poor (2mbit, 150ms delay, 3% loss)
     - 5: Very Poor (1mbit, 300ms delay, 5% loss)
   - The script will apply the selected network conditions
   - You can also select option 5 to apply ultra-low-latency conditions for the best possible streaming experience

4. **Start the Sender**
   - In a new terminal on the sender computer, run:
     ```bash
     python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO
     ```
     (Replace RECEIVER_IP with the actual IP address and PATH_TO_VIDEO with your video file path)

5. **Observe the Effects**
   - Watch how the video quality and performance change under different network conditions
   - Monitor the statistics in both terminals

6. **Try Different Conditions**
   - While the video is streaming, you can change network conditions
   - Go back to the traffic control terminal
   - Select option 2 again and choose a different preset
   - Observe how the video adapts to the new conditions

7. **Reset Network Conditions**
   - When finished, select option 4 from the traffic control menu to reset network conditions
   - Select option 6 to exit the traffic control script

### Testing on a Single Computer

1. **Open Two Terminals**
   - In the first terminal, start the receiver:
     ```bash
     python direct_receiver.py --display
     ```
   - In the second terminal, start the sender using localhost:
     ```bash
     python direct_sender.py --ip localhost --video PATH_TO_VIDEO
     ```

2. **Optional: Apply Dynamic Traffic Control**
   - Open a third terminal
   - You have two options:
     
     **Option 1: Manual Traffic Control**
     ```bash
     sudo ./dynamic_tc_control.sh
     ```
     - Follow steps 3-7 from the Advanced Setup
     
     **Option 2: Automatic Traffic Control (Recommended)**
     ```bash
     sudo python3 auto_tc_control.py
     ```
     - The script will automatically adjust network conditions based on real-time metrics

### Stopping the System

1. **Stop the Sender**
   - In the sender terminal, press Ctrl+C once
   - Wait for the program to clean up resources
   - If it doesn't exit within a few seconds, press Ctrl+C again

2. **Stop the Receiver**
   - In the receiver terminal, press Ctrl+C once
   - Wait for the program to clean up resources
   - If it doesn't exit within a few seconds, press Ctrl+C again

3. **Reset Traffic Control (if used)**
   - In the traffic control terminal, select option 4 to reset network conditions
   - Select option 6 to exit

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

## Buffer Management

If you see "Buffer fullness: 30/30 (100.0%)" or an extremely high "Actual FPS" value in the statistics, it indicates that the sender is producing frames faster than they can be sent. This can happen when:

1. The network connection is slow
2. The receiver is not connected
3. The video file is being read too quickly

### Clearing a Full Buffer

To address a full buffer:

1. **Reduce the reading rate**:
   ```bash
   python direct_sender.py --ip RECEIVER_IP --fps 15
   ```

2. **Increase the buffer size**:
   ```bash
   python direct_sender.py --ip RECEIVER_IP --buffer 60
   ```

3. **Ensure the receiver is running** before starting the sender

4. **Restart both sender and receiver** if the buffer remains full

### Properly Stopping the Program

To properly stop the sender or receiver:

1. Press `Ctrl+C` once and wait for the program to clean up resources
2. If it doesn't exit within a few seconds, press `Ctrl+C` again
3. If you see a KeyboardInterrupt error, the program has been forcibly terminated but should still have released most resources

## Low Latency Mode

The system is configured for low latency by default, which minimizes the delay between the sender and receiver displays. This is achieved by:

1. **Minimal buffer sizes** (only 5 frames) on both sender and receiver
2. **Ultra-minimal initial buffering** (only 2-3 frames) before playback starts
3. **Faster response times** with shorter sleep intervals
4. **Immediate frame display** on the sender side

If you experience stuttering or frame drops with low latency mode, you can disable it:

```bash
python direct_receiver.py --display --low-latency=False --buffer 15
python direct_sender.py --ip RECEIVER_IP --video PATH_TO_VIDEO --buffer 15
```

This will increase the buffer sizes and initial buffering, which can help with smoother playback at the cost of higher latency.

## Dynamic Traffic Control (TC)

### Manual Traffic Control

The dynamic traffic control script (`dynamic_tc_control.sh`) allows you to manually simulate different network conditions to test how the video streaming performs under various scenarios. It's particularly useful for testing how the low-latency optimizations perform under different network conditions.

### Automatic Traffic Control

The automatic traffic control script (`auto_tc_control.sh`) provides an easy way to test your video streaming under different network conditions. It automatically cycles through a series of network presets, from very poor to ultra-smooth, at regular intervals.

Key features of the automatic traffic control:

1. **Automatic cycling** - Automatically changes network conditions every 20 seconds
2. **Progressive testing** - Starts with poor conditions and gradually improves to excellent
3. **Colorful output** - Uses color-coded terminal output for better readability
4. **Real-time statistics** - Shows current network conditions and statistics
5. **Ultra-smooth mode** - Includes an optimized ultra-low-latency configuration

This script is perfect for testing how your video streaming performs under different network conditions without having to manually change settings.

### Performance Monitoring and Visualization

The performance monitoring script (`performance_monitor.py`) collects metrics from the sender and receiver, and generates detailed graphs showing the relationship between network conditions and actual performance.

Key features of the performance monitor:

1. **Real-time metrics collection** - Gathers data from both sender and receiver
2. **Network condition tracking** - Records the current traffic control settings
3. **Comprehensive graphs** - Generates multiple visualizations showing different aspects of performance
4. **Performance impact analysis** - Shows how network conditions affect streaming quality
5. **Live graph option** - Can display graphs in real-time during data collection
6. **Data export** - Saves raw data for further analysis

This tool is essential for understanding how different network conditions affect your streaming performance and for identifying bottlenecks in your system.

### Performance Viewer

The performance viewer (`performance_viewer.html`) is a web-based interface for viewing and analyzing the performance graphs generated by the monitoring script. It provides an easy way to compare different test runs and understand the impact of network conditions on streaming quality.

Key features of the performance viewer:

1. **Interactive interface** - Browse and select different graph files
2. **Filtering options** - View specific types of graphs based on your needs
3. **Key metrics display** - See important performance indicators at a glance
4. **Responsive design** - Works on desktop and mobile browsers
5. **Search functionality** - Easily find specific test results

To use the performance viewer, simply open the HTML file in any web browser after running the performance monitor to generate graphs.

#### Metrics API

Both the sender and receiver include a built-in metrics API that provides real-time performance data:

- **Sender Metrics API**: http://SENDER_IP:8000/metrics
- **Receiver Metrics API**: http://RECEIVER_IP:8001/metrics

These APIs return JSON data with detailed performance metrics including:
- Bandwidth usage
- Frame rates
- Buffer fullness
- Frame delivery times
- Drop rates
- Resolution and quality settings

You can access these metrics directly in a web browser or use them with monitoring tools.

### Prerequisites for Traffic Control

- Linux operating system (TC is a Linux-specific tool)
- Root privileges (sudo access)
- iproute2 package installed (provides the `tc` command)

```bash
# Install iproute2 if not already installed
sudo apt update
sudo apt install iproute2
```

### Running Traffic Control

#### Manual Traffic Control

1. Make the script executable (if not already):
   ```bash
   chmod +x dynamic_tc_control.sh
   ```

2. Run the script with sudo:
   ```bash
   sudo ./dynamic_tc_control.sh
   ```

#### Automatic Traffic Control

1. Make the script executable (if not already):
   ```bash
   chmod +x auto_tc_control.sh
   ```

2. Run the script with sudo:
   ```bash
   sudo ./auto_tc_control.sh
   ```

3. The script will:
   - Detect your network interface or ask you to select one
   - Start cycling through network conditions automatically
   - Apply each condition for 20 seconds before moving to the next
   - Show real-time statistics for each condition
   - Continue cycling until you press Ctrl+C

3. When you first run the script, it will detect your network interface or ask you to select one.

4. For manual control, from the menu, you can:
   - Option 1: Set custom network conditions (bandwidth, delay, packet loss)
   - Option 2: Apply preset network conditions (Excellent, Good, Fair, Poor, Very Poor)
   - Option 3: Show current network statistics
   - Option 4: Reset network conditions
   - Option 5: Apply ultra-low-latency conditions
   - Option 6: Exit

5. The script cycles through these network conditions:
   - **VERY POOR**: 1Mbit, 300ms delay, 5% loss
   - **POOR**: 2Mbit, 150ms delay, 3% loss
   - **FAIR**: 4Mbit, 80ms delay, 1% loss
   - **GOOD**: 6Mbit, 40ms delay, 0.5% loss
   - **EXCELLENT**: 10Mbit, 20ms delay, 0% loss
   - **ULTRA**: Special ultra-low-latency configuration

### Troubleshooting TC

- If you get "Command not found" errors, ensure iproute2 is installed
- If you get permission errors, make sure you're running with sudo
- If network interface detection fails, manually edit the INTERFACE variable in the script

## Common Issues and Solutions

### Video plays too fast or too slow:
```bash
# Set specific frame rate on both sides
python direct_receiver.py --display --fps 30
python direct_sender.py --ip RECEIVER_IP --fps 30
```

### Video stutters or blocks:
```bash
# Increase buffer sizes
python direct_receiver.py --display --buffer 30 --low-latency=False
python direct_sender.py --ip RECEIVER_IP --buffer 30

### Network conditions affect video quality:
If you're experiencing poor video quality or high latency, try applying different network conditions:

```bash
# For ultra-low-latency (best performance)
sudo ./dynamic_tc_control.sh
# Then select option 5 to apply ultra-low-latency conditions

# For testing under poor network conditions
sudo ./dynamic_tc_control.sh
# Then select option 2 and choose preset 4 or 5

# For automatic cycling through different network conditions
sudo ./auto_tc_control.sh

# For monitoring performance and generating graphs
python performance_monitor.py --live
```

### Accessing metrics directly:
```bash
# View sender metrics in browser
firefox http://localhost:8000/metrics

# View receiver metrics in browser
firefox http://RECEIVER_IP:8001/metrics

# Get metrics via curl
curl http://localhost:8000/metrics | jq
```

### Performance Monitoring Options:
```bash
# Basic monitoring with default settings
python performance_monitor.py

# Live graph display during monitoring
python performance_monitor.py --live

# Custom monitoring configuration
python performance_monitor.py --sender-ip localhost --receiver-ip 192.168.2.169 --interval 0.5 --duration 600

# Save graphs to a specific directory
python performance_monitor.py --output ./my_performance_graphs
```

### Viewing Performance Graphs:
```bash
# Open the performance viewer in a web browser
firefox performance_viewer.html

# Or use any other browser
google-chrome performance_viewer.html
```

For the best experience, follow this workflow:
1. Start the receiver and sender with metrics enabled
2. Run the automatic traffic control script to cycle through different network conditions
3. Run the performance monitor to collect data and generate graphs
4. View the results in the performance viewer to analyze the impact of network conditions
```

### High bandwidth usage:
```bash
# Reduce quality and resolution
python direct_sender.py --ip RECEIVER_IP --quality 70 --scale 0.75
```

### Extremely high "Actual FPS" reported:
This usually indicates the sender is reading frames faster than it can send them. Try:
```bash
# Limit the frame rate
python direct_sender.py --ip RECEIVER_IP --fps 30
```

## Documentation

For detailed documentation including architecture diagrams, implementation details, and advanced usage, see:

- [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md) - Complete system documentation with diagrams and technical details

## How It Works

The system uses a multi-threaded approach with frame buffering:

1. The sender reads video frames and stores them in a buffer
2. A separate thread sends frames at the target frame rate
3. The receiver stores incoming frames in a buffer
4. A separate thread displays frames at the correct playback rate
5. Both sides monitor and display traffic statistics

This architecture ensures smooth video playback even with network jitter and provides real-time performance metrics.