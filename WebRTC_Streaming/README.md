# WebRTC Streaming

This project demonstrates a WebRTC video streaming setup using Python and the aiortc library. It consists of a sender script that captures video from a file, a receiver script that displays the received video using OpenCV, and a signaling server that automates the SDP exchange.

## Prerequisites

- Python 3.6+
- Virtual environment (recommended)
- Required Python packages (install using `pip install -r requirements.txt`):
  - aiortc
  - opencv-python
  - numpy
  - av
  - websockets

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

### One-Click Start (Easiest)

The project includes a script that starts all components (signaling server, receiver, and sender) with a single command:

```bash
python start_streaming.py
```

This will automatically:
1. Start the signaling server
2. Start the receiver
3. Start the sender
4. Establish the WebRTC connection

Press Ctrl+C to stop all components when you're done.

### Automated Signaling (Alternative)

If you prefer to start each component separately:

1. Start the signaling server:
   ```bash
   python signaling_server.py
   ```

2. In a new terminal, start the receiver:
   ```bash
   python webrtc_receiver.py
   ```

3. In another terminal, start the sender:
   ```bash
   python webrtc_sender.py
   ```

The signaling server will automatically handle the exchange of SDP information between the sender and receiver, and the WebRTC connection will be established without manual intervention.

### Manual Signaling (Alternative)

If you prefer to manually exchange SDP information, you can use the older versions of the scripts:

1. Start the sender:
   ```bash
   python webrtc_sender_manual.py
   ```
   The sender will generate an SDP offer and wait for an answer.

2. Start the receiver:
   ```bash
   python webrtc_receiver_manual.py
   ```
   The receiver will prompt you to paste the sender's offer.

3. Copy the entire JSON offer from the sender's terminal and paste it into the receiver's terminal.
4. Copy the entire JSON answer from the receiver's terminal and paste it into the sender's terminal.

## How It Works

1. **Signaling Server**: Acts as an intermediary for exchanging SDP information between the sender and receiver.
2. **Sender**: Captures video from a file and sends it over WebRTC.
3. **Receiver**: Receives the WebRTC video stream and displays it using OpenCV.

The signaling process follows these steps:
1. Both sender and receiver connect to the signaling server with unique IDs.
2. The sender creates an offer and sends it to the signaling server.
3. The signaling server forwards the offer to the receiver.
4. The receiver creates an answer and sends it to the signaling server.
5. The signaling server forwards the answer to the sender.
6. The WebRTC connection is established directly between the sender and receiver.

## Troubleshooting

- **WebSocket connection fails**: Make sure the signaling server is running and accessible at ws://localhost:8765.
- **No video appears**: Check that the video file path in `webrtc_sender.py` is correct. The default path is `video/zidane.mp4`.
- **ICE connection fails**: This can happen due to network restrictions. WebRTC works best on local networks or with proper STUN/TURN servers configured.

## Project Structure

- `signaling_server.py`: WebSocket server that handles SDP exchange
- `webrtc_sender.py`: Captures video from a file and sends it over WebRTC
- `webrtc_receiver.py`: Receives WebRTC video stream and displays it using OpenCV
- `webrtc_receiver.html`: Placeholder for a browser-based receiver (not implemented)
- `requirements.txt`: List of required Python packages

## Cross-Machine Setup

### Quick Reference: What to Run on Each PC

#### On the Sender PC (e.g., 192.168.2.120):

1. First, find your IP address:
   ```bash
   python get_local_ip.py
   ```
   (Note the IP address, e.g., 192.168.2.120)

2. Start the signaling server:
   ```bash
   python signaling_server.py
   ```

3. In a new terminal, start the sender:
   ```bash
   python webrtc_sender.py
   ```

#### On the Receiver PC (e.g., 192.168.2.169):

1. Copy the WebRTC_Streaming directory to this PC
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the receiver with the sender's IP address:
   ```bash
   python webrtc_receiver.py --signaling-server ws://192.168.2.120:8765
   ```
   (Replace 192.168.2.120 with the actual IP address of the sender PC)

### Detailed Setup Instructions

You can run the sender and receiver on different machines in the same network. This is useful for testing real network conditions or when you want to display the video on a different computer.

#### Step 1: Determine the Server IP

On the machine that will run the signaling server (typically the sender machine), run:

```bash
python get_local_ip.py
```

This will display your local IP address (e.g., 192.168.2.120).

#### Step 2: Start the Signaling Server

On the server machine, start the signaling server:

```bash
python signaling_server.py
```

The server will listen on all network interfaces (0.0.0.0) on port 8765.

#### Step 3: Start the Sender

On the same machine as the signaling server, start the sender:

```bash
python webrtc_sender.py
```

If you need to specify a different video file:

```bash
python webrtc_sender.py --video-path /path/to/your/video.mp4
```

#### Step 4: Start the Receiver on Another Machine

On the second machine (e.g., 192.168.2.169), start the receiver with the signaling server IP:

```bash
python webrtc_receiver.py --signaling-server ws://192.168.2.120:8765
```

Replace 192.168.2.120 with the actual IP address of your signaling server (the one shown by get_local_ip.py).

### Troubleshooting Cross-Machine Setup

- **Connection fails**: Make sure both machines are on the same network and can reach each other. Try pinging the server IP from the client machine.
- **Firewall issues**: Make sure port 8765 is allowed through any firewalls on the server machine.
- **ICE connection fails**: WebRTC may have trouble with certain network configurations. Consider adding STUN/TURN server configuration for better NAT traversal.

## Future Improvements

- Add audio support
- Create a fully functional browser-based receiver
- Add STUN/TURN server configuration for NAT traversal
- Implement secure WebSocket connections (WSS)
- Add support for multiple simultaneous connections