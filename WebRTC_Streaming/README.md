# Simple Video Streaming with Traffic Monitoring

A lightweight solution for streaming video between computers with real-time traffic monitoring and network control.

## Quick Start

### On the Receiver Computer (e.g., 192.168.2.169):

```bash
# Start the receiver WITH VIDEO DISPLAY (important!)
python direct_receiver.py --display
```

The `--display` flag is critical - without it, the receiver will process frames but not show the video.

### On the Sender Computer (e.g., 192.168.2.120):

```bash
# Start the sender
python direct_sender.py --ip 192.168.2.169 --video ../video/zidane.mp4
```

### What You'll See:

1. On the receiver: 
   - A window showing the video
   - Traffic statistics in the terminal

2. On the sender:
   - Traffic statistics in the terminal

## Command Options

### Sender Options:
- `--ip`: Receiver IP address (default: 192.168.2.169)
- `--port`: Receiver port (default: 9999)
- `--video`: Video file path (default: ../video/zidane.mp4)
- `--quality`: JPEG quality 1-100 (default: 90)
- `--scale`: Resolution scale factor (default: 1.0)

### Receiver Options:
- `--ip`: IP address to listen on (default: 0.0.0.0)
- `--port`: Port to listen on (default: 9999)
- `--display`: Display video (REQUIRED to see the video)

## Testing on a Single Computer

To test on a single computer:

1. First terminal (receiver):
   ```bash
   python direct_receiver.py --display
   ```

2. Second terminal (sender):
   ```bash
   python direct_sender.py --ip localhost
   ```

## Traffic Control

To simulate different network conditions:

```bash
# Run with sudo (required for tc commands)
sudo ./webrtc_tc_control.sh
```

Choose from the menu:
1. Set custom network conditions (bandwidth, delay, packet loss)
2. Apply preset network conditions (Excellent, Good, Fair, Poor)
3. Show current network statistics
4. Reset network conditions

## How It Works

1. The sender reads frames from a video file
2. Frames are compressed using JPEG encoding
3. Compressed frames are sent over TCP to the receiver
4. The receiver decodes and optionally displays the frames
5. Both sides monitor and display traffic statistics
6. The tc script can modify network conditions to test performance

## Troubleshooting

- Make sure both computers are on the same network
- Check if any firewalls are blocking port 9999
- **Video display issues**: 
  - Ensure you used the `--display` flag on the receiver
  - Make sure you have a GUI environment (X server) running
  - Check that OpenCV is installed correctly with GUI support
- If traffic control doesn't work, make sure you're running as root (sudo)