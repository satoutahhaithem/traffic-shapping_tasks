# WebRTC Streaming

This project demonstrates a simple WebRTC video streaming setup using Python and the aiortc library. It consists of a sender script that captures video from a file and a receiver script that displays the received video using OpenCV.

## Prerequisites

- Python 3.6+
- Virtual environment (recommended)
- Required Python packages (install using `pip install -r requirements.txt`):
  - aiortc
  - opencv-python
  - numpy
  - av

## Setup

1. Clone this repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The WebRTC connection requires manual signaling between the sender and receiver. Follow these steps to establish a connection:

### Step 1: Start the Sender

In one terminal, run:

```bash
python webrtc_sender.py
```

The sender will generate an SDP offer and wait for an answer.

### Step 2: Start the Receiver

In another terminal, run:

```bash
python webrtc_receiver.py
```

The receiver will prompt you to paste the sender's offer.

### Step 3: Exchange SDP Information

1. Copy the entire JSON offer from the sender's terminal (including the curly braces)
2. Paste it into the receiver's terminal and press Enter
3. The receiver will generate an SDP answer
4. Copy the entire JSON answer from the receiver's terminal
5. Paste it into the sender's terminal and press Enter

### Step 4: Connection Establishment

Once the SDP exchange is complete, the WebRTC connection will be established:

1. The sender will stream the video file
2. The receiver will display the video in an OpenCV window
3. Press 'q' in the receiver's window to exit

## Troubleshooting

- **AttributeError: 'RTCPeerConnection' object has no attribute 'signal'**: This error occurs if you're using an older version of the code with a newer version of aiortc. The correct event registration method is `pc.on()`, not `pc.signal`.

- **No video appears**: Check that the video file path in `webrtc_sender.py` is correct. The default path is `video/zidane.mp4`.

- **ICE connection fails**: This can happen due to network restrictions. WebRTC works best on local networks or with proper STUN/TURN servers configured.

## Project Structure

- `webrtc_sender.py`: Captures video from a file and sends it over WebRTC
- `webrtc_receiver.py`: Receives WebRTC video stream and displays it using OpenCV
- `webrtc_receiver.html`: Placeholder for a browser-based receiver (not implemented)
- `requirements.txt`: List of required Python packages

## Future Improvements

- Implement a signaling server to automate the SDP exchange
- Add audio support
- Create a fully functional browser-based receiver
- Add STUN/TURN server configuration for NAT traversal