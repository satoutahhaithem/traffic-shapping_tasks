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

## Controlling Video Playback

### Fixing Video Speed Issues

If the video plays too fast or too slow:

```bash
# On the receiver - force a specific playback FPS
python direct_receiver.py --display --fps 30

# On the sender - control the sending rate
python direct_sender.py --ip 192.168.2.169 --fps 30
```

The system will automatically match the video's original frame rate by default, but you can override it with the `--fps` option on both sides.

### Fixing Video Blocking/Stuttering

If the video blocks or stutters:

1. **Increase buffer sizes** on both sender and receiver:
   ```bash
   # On receiver
   python direct_receiver.py --display --buffer 120
   
   # On sender
   python direct_sender.py --ip 192.168.2.169 --buffer 60
   ```

2. **Reduce video quality or resolution**:
   ```bash
   python direct_sender.py --ip 192.168.2.169 --quality 80 --scale 0.75
   ```

## Command Options

### Sender Options:
- `--ip`: Receiver IP address (default: 192.168.2.169)
- `--port`: Receiver port (default: 9999)
- `--video`: Video file path (default: ../video/zidane.mp4)
- `--quality`: JPEG quality 1-100 (default: 90)
- `--scale`: Resolution scale factor (default: 1.0)
- `--fps`: Target FPS (default: 0 = use video's FPS)
- `--buffer`: Frame buffer size (default: 30)

### Receiver Options:
- `--ip`: IP address to listen on (default: 0.0.0.0)
- `--port`: Port to listen on (default: 9999)
- `--display`: Display video (REQUIRED to see the video)
- `--buffer`: Frame buffer size (default: 60)
- `--fps`: Override playback FPS (default: 0 = use sender's FPS)

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

1. The sender reads frames from a video file at the video's natural frame rate
2. Frames are stored in a buffer on the sender side
3. A separate thread sends frames at the target FPS rate
4. Frames are compressed using JPEG encoding
5. Compressed frames are sent over TCP to the receiver
6. The receiver stores incoming frames in a buffer
7. A separate thread displays frames at the correct playback rate
8. Both sides monitor and display traffic statistics

## Troubleshooting

- Make sure both computers are on the same network
- Check if any firewalls are blocking port 9999
- **Video display issues**: 
  - Ensure you used the `--display` flag on the receiver
  - Make sure you have a GUI environment (X server) running
  - Check that OpenCV is installed correctly with GUI support
- **Video speed issues**:
  - Use the `--fps` option on both sender and receiver to set the same frame rate
  - Try matching the original video's frame rate
- **Video stuttering/blocking**:
  - Increase buffer sizes on both sender and receiver
  - Reduce video quality or resolution
  - Limit the frame rate
  - Check network conditions
- If traffic control doesn't work, make sure you're running as root (sudo)