# Video Streaming with Network Condition Simulation

This project allows you to stream video while simulating different network conditions using Linux Traffic Control (tc).

## Components

1. **dynamic_tc_control.sh**: Controls network conditions (bandwidth, delay, packet loss) via command line
2. **video_streamer.py**: Streams video frames to a receiver
3. **receive_video.py**: Receives and displays video frames
4. **tc_api.py**: API server for controlling network conditions via web interface
5. **control_updated.html**: Enhanced web interface for controlling all network parameters

## Setup

The system has been configured to work on your PC with:
- Network interface: wlp0s20f3
- Video file: BigBuckBunny.mp4 (located in the Video_test directory)
- Server: localhost (127.0.0.1)

## How to Use

### Option 1: Automatic Startup (Recommended)

Use the provided bash script to start all components automatically:

```
./start_system.sh
```

This script will:
1. Check and activate the virtual environment
2. Start all components in separate terminal windows
3. Open the control panel in your browser

You will be prompted for your sudo password when starting the TC API server.

### Option 2: Manual Startup

**IMPORTANT: You must run all components in the correct order for the system to work properly.**

1. **Activate the virtual environment** (if you're using one):
   ```
   source .venv/bin/activate
   ```

2. **Start the receiver** (Terminal 1):
   ```
   python3 receive_video.py
   ```
   This will start a server on port 5001 that receives video frames.

3. **Start the streamer** (Terminal 2, with virtual environment activated):
   ```
   source .venv/bin/activate  # If using virtual environment
   python3 video_streamer.py
   ```
   This will start streaming the video to the receiver.
   **The streamer MUST be running for the control panel to work properly.**

4. **Start the TC API server** (Terminal 3, with sudo privileges):
   ```
   ./run_tc_api.sh
   ```
   This will start a server on port 5002 that can execute tc commands.
   **Sudo privileges are required for tc commands.**

5. **Start the control panel** (Terminal 4):
   ```
   python3 serve_control_panel_updated.py
   ```
   This will open the control panel in your browser.

6. **Alternative: Manual network control** (Optional, Terminal 5):
   ```
   bash dynamic_tc_control.sh
   ```
   This is an alternative to using the web interface for controlling network conditions.
   This will show a menu where you can:
   - Set network conditions (bandwidth, delay, packet loss)
   - Show current network statistics
   - Reset network conditions

## Viewing the Video

### Option 1: Direct Video Feeds
- Transmitter side: http://localhost:5000/tx_video_feed
- Receiver side: http://localhost:5001/rx_video_feed

### Option 2: Enhanced Control Panel (Recommended)
Use the updated Python script to serve the enhanced control panel and open it in your browser:
```
python3 serve_control_panel_updated.py
```

This will:
1. Start a simple HTTP server on port 8000
2. Open the enhanced control panel in your default browser
3. Allow you to view both video feeds side by side in larger frames
4. Provide complete control over all network parameters (delay, bandwidth, packet loss)
5. Offer preset network conditions for common scenarios

The enhanced control panel provides a comprehensive interface to test different network conditions with:
- Complete control over all network parameters:
  - Delay (latency) in milliseconds
  - Bandwidth (rate) in kbit/s, Mbit/s, or Gbit/s
  - Packet loss percentage
- Preset network conditions (Perfect, Good Broadband, Average Mobile, etc.)
- Custom input fields for precise testing
- Resizable video frames
- Real-time Quality of Service metrics that update based on all parameters

## Testing Network Conditions

### Method 1: Using Traffic Control (tc)

When setting network conditions in the dynamic_tc_control.sh script:
- **Rate**: Controls bandwidth (e.g., 10mbit, 1mbit, 500kbit)
  - You can enter just the number (e.g., "10") and "mbit" will be added automatically
- **Delay**: Controls latency (e.g., 10ms, 100ms, 200ms)
  - You can enter just the number (e.g., "100") and "ms" will be added automatically
- **Loss**: Controls packet loss percentage (e.g., 0%, 1%, 5%)
  - You can enter just the number (e.g., "5") and "%" will be added automatically

Suggested test scenarios:
- **Good connection**: Rate=10mbit, Delay=10ms, Loss=0%
- **Medium connection**: Rate=1mbit, Delay=100ms, Loss=1%
- **Poor connection**: Rate=500kbit, Delay=200ms, Loss=5%

## Understanding Network Quality Parameters

The dynamic_tc_control.sh script uses three key parameters to simulate different network conditions:

### 1. Rate (Bandwidth)

**What it is:** Rate limits the maximum bandwidth (data transfer speed) of the network connection.

**How it works:**
- Measured in bits per second (e.g., 1mbit = 1 megabit per second)
- Lower values simulate a slower connection (like a poor mobile connection)
- Higher values simulate a faster connection (like fiber broadband)

**Examples:**
- 10mbit: Good broadband connection (~1.25 MB/s download speed)
- 1mbit: Slow connection (like basic mobile data, ~125 KB/s)
- 500kbit: Very slow connection (like edge network, ~62.5 KB/s)

**Effect on video:** Lower bandwidth means the video may need to use lower quality/resolution or buffer frequently.

### 2. Delay (Latency)

**What it is:** Delay adds artificial latency to the network connection.

**How it works:**
- Measured in milliseconds (ms)
- Represents the time it takes for data to travel from sender to receiver
- Higher values mean longer travel time for data packets

**Examples:**
- 10ms: Excellent connection (like local fiber)
- 100ms: Average internet connection
- 200ms: Poor connection (like satellite internet)

**Effect on video:** Higher delay means longer wait time before video starts playing and slower response to interactions.

### 3. Loss (Packet Loss)

**What it is:** Loss simulates random packet loss in the network.

**How it works:**
- Measured as a percentage (%)
- Represents the percentage of data packets that never arrive at their destination
- Higher values mean more data needs to be resent

**Examples:**
- 0%: Perfect connection with no lost packets
- 1%: Slightly unstable connection
- 5%: Very poor connection
- 10%: Extremely unreliable connection

**Effect on video:** Higher packet loss causes video artifacts, freezing, and quality drops.

### How These Parameters Work Together

These three parameters can be combined to simulate various real-world network scenarios:

- **Perfect Connection:** Rate=10mbit, Delay=10ms, Loss=0%
- **Good Home Broadband:** Rate=5mbit, Delay=30ms, Loss=0.1%
- **Average Mobile Connection:** Rate=2mbit, Delay=100ms, Loss=1%
- **Poor Mobile Connection:** Rate=500kbit, Delay=200ms, Loss=5%
- **Satellite Internet:** Rate=1mbit, Delay=500ms, Loss=2%

## Testing Methods

### Method 1: Using Traffic Control (tc)

When setting network conditions in the dynamic_tc_control.sh script, you can adjust all three parameters (rate, delay, loss) for comprehensive network simulation.

### Method 2: Using Network Delay (Easier to observe)

You can simulate network conditions by adjusting the network delay directly in the video streamer.
The transmitter side will always show the video at normal speed, while the receiver side will show
the delayed version based on your settings.

The enhanced control panel provides multiple quality levels:

- **Perfect (30 fps)**: 0.033s delay - No noticeable delay
- **Very Good (15 fps)**: 0.066s delay - Minimal delay
- **Good (10 fps)**: 0.1s delay - Minor delay, still smooth
- **Fair (5 fps)**: 0.2s delay - Noticeable delay
- **Poor (3 fps)**: 0.33s delay - Significant delay
- **Very Poor (2 fps)**: 0.5s delay - Substantial delay
- **Bad (1.25 fps)**: 0.8s delay - Very high delay
- **Very Bad (1 fps)**: 1.0s delay - Extreme delay
- **Terrible (0.67 fps)**: 1.5s delay - Nearly unusable

You can also enter custom delay values for precise testing between 0.01 and 10.0 seconds.

**Note:** Values up to 5.0 seconds are safe. Values between 5.0 and 10.0 seconds may cause instability.

This method is more noticeable when testing locally, as it clearly shows the difference between
the original video and the delayed version.

## Requirements

- Python 3 with Flask, OpenCV, NumPy, and Requests
- Linux with tc (Traffic Control) utility
- Sudo privileges (for tc commands)