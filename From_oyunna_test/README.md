# Network Video Streaming with Traffic Control

This system allows you to stream video between two computers and simulate different network conditions using Linux Traffic Control (tc). You can apply traffic control on either the sender or receiver PC to observe how network conditions affect video quality.

## System Components

1. **Video Streamer** (`video_streamer.py`): Reads a video file and streams it to a receiver
2. **Video Receiver** (`receive_video.py`): Receives video frames and displays them in a browser
3. **Traffic Control** (`dynamic_tc_control.sh`): Simulates different network conditions
4. **Setup Scripts**: Helper scripts for configuring the system on different PCs

## Setup Instructions

### Preparing Both PCs

1. Copy the `From_oyunna_test` folder to both the sender and receiver PCs
2. Make sure both PCs have the required Python packages installed:
   ```bash
   pip install flask opencv-python numpy requests
   ```
3. Ensure both PCs have the `tc` command available (part of the `iproute2` package)

### Sender PC Setup

1. Run the sender setup script:
   ```bash
   cd /path/to/git_mesurement_tc
   ./From_oyunna_test/sender_setup.sh
   ```
2. When prompted, enter the IP address of the receiver PC
3. The script will configure the system for sending video frames

### Receiver PC Setup

1. Run the receiver setup script:
   ```bash
   cd /path/to/git_mesurement_tc
   ./From_oyunna_test/receiver_setup.sh
   ```
2. The script will configure the system for receiving video frames

## Running the System

### On the Receiver PC

1. Start the video receiver:
   ```bash
   python3 From_oyunna_test/receive_video.py
   ```
2. The receiver will listen for frames on port 8081

### On the Sender PC

1. Start the video streamer:
   ```bash
   python3 From_oyunna_test/video_streamer.py
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

## Applying Traffic Control

You can apply traffic control on either the sender or receiver PC to simulate different network conditions:

### On the Sender PC

Running traffic control on the sender affects outgoing traffic (frames being sent to the receiver):
```bash
sudo bash From_oyunna_test/dynamic_tc_control.sh
```

### On the Receiver PC

Running traffic control on the receiver affects incoming traffic (frames being received):
```bash
sudo bash From_oyunna_test/dynamic_tc_control.sh
```

## Testing Different Network Scenarios

For noticeable effects on video quality, try these settings:

1. **High Latency**:
   - Rate: `5mbit`
   - Delay: `1000ms`
   - Loss: `0%`

2. **Low Bandwidth**:
   - Rate: `200kbit`
   - Delay: `100ms`
   - Loss: `0%`

3. **Packet Loss**:
   - Rate: `1mbit`
   - Delay: `100ms`
   - Loss: `20%`

4. **Combined Poor Conditions**:
   - Rate: `300kbit`
   - Delay: `500ms`
   - Loss: `10%`

## Comparing Effects

To compare the effects of applying traffic control at different points:

1. **Sender-side traffic control**: Affects how quickly frames can be sent out
2. **Receiver-side traffic control**: Affects how quickly frames can be received

The visual impact may be slightly different depending on where you apply the traffic control. Try both approaches to see which produces more noticeable effects for your specific testing needs.

## Troubleshooting

- If the video doesn't appear, check that both applications are running and that the IP addresses are configured correctly
- If traffic control doesn't seem to have an effect, try more extreme settings
- Make sure you're using the correct network interface in the dynamic_tc_control.sh script
- Check the terminal output for any error messages