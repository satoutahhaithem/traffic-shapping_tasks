# Video Streaming with Traffic Control

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

### On Receiver PC:
```bash
python direct_receiver.py --display --metrics-port 8001
```

### On Sender PC:
Run these in separate terminals:

```bash
# Terminal 1: Start traffic control
sudo ./auto_tc_control.sh

# Terminal 2: Start the sender
python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000
```
Replace RECEIVER_IP with the receiver's IP address.

## Performance Measurement Solutions

This system offers two different solutions for measuring and visualizing the performance of traffic control:

### Solution 1: Traffic Shaping on Sender, Measurement on Receiver

This approach keeps traffic control and performance measurement separate:

1. **On the Sender PC**: Run auto_tc_control.sh to apply traffic control
2. **On the Receiver PC**: Run tc_performance_manual.py to measure and plot

```bash
# On the sender PC
sudo ./auto_tc_control.sh

# On the receiver PC
sudo python tc_performance_manual.py --sender-ip SENDER_IP --receiver-ip localhost
```

This script has the auto_tc_control.sh cycle built-in, so it knows what the commanded values should be without having to detect them.

### Solution 2: Traffic Shaping and Measurement on Receiver

This approach combines traffic control and performance measurement in a single script that runs entirely on the receiver PC:

```bash
# On the receiver PC only
sudo python tc_all_in_one.py --sender-ip SENDER_IP --receiver-ip localhost
```

This script:
1. Applies traffic control directly on the receiver PC
2. Cycles through the same network conditions as auto_tc_control.sh
3. Measures the performance and plots the graphs
4. Resets the network conditions when done

### When to Do Traffic Shaping on the Receiver PC

You should use Solution 2 (tc_all_in_one.py) when:

1. **You want to control the network at the receiving end**:
   - This simulates bandwidth limitations, latency, and packet loss at the receiver
   - Useful when testing how a client handles poor network conditions

2. **You want a simpler setup with everything on one machine**:
   - No need to coordinate between two machines
   - Easier to start and stop the entire system

3. **You want to ensure perfect synchronization**:
   - The commanded values and measurements are taken on the same machine
   - This eliminates timing discrepancies between sender and receiver

4. **You want to plot graphs that accurately show both commanded and measured values**:
   - Since both traffic shaping and measurement happen on the same machine
   - The graphs will show perfect correlation between what was commanded and what was measured

### Important Note About Commanded Values

When running traffic control on the sender and measurement on the receiver (Solution 1), there's a challenge:
- The receiver doesn't know what traffic control settings are being applied on the sender
- This is why we created `tc_performance_manual.py` which has the auto_tc_control.sh cycle built-in
- It doesn't need to query the sender for the commanded values; it already knows them

If you want to do traffic shaping on the receiver PC (Solution 2), you get these benefits:
- The receiver directly knows what traffic control settings are being applied
- No need to synchronize or communicate the commanded values between machines
- The graphs will be more accurate because the commanded values are known with certainty

### Step-by-Step Guide for Traffic Shaping on Receiver PC

To implement Solution 2 (traffic shaping on the receiver PC):

1. **Start the sender without traffic control**:
   ```bash
   # On the sender PC
   python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000
   ```

2. **Start the receiver**:
   ```bash
   # On the receiver PC
   python direct_receiver.py --display --metrics-port 8001
   ```

3. **Run the all-in-one script on the receiver**:
   ```bash
   # On the receiver PC
   sudo python tc_all_in_one.py --sender-ip SENDER_IP --receiver-ip localhost
   ```
   Replace SENDER_IP with the actual IP address of the sender.

4. **Watch the results**:
   - The script will apply traffic control settings on the receiver PC
   - It will cycle through different network conditions every 20 seconds
   - It will collect performance metrics and display them in real-time
   - After 2 minutes (or when you press Ctrl+C), it will generate and display graphs
   - The graphs will show both commanded and measured values

## Troubleshooting

### Address already in use error

If you see this error when starting the receiver:
```
Error: [Errno 98] Address already in use
```

This means port 9999 (the default port) is already in use. You can:

1. **Find and kill the process using the port**:
   ```bash
   # Find the process using port 9999
   sudo lsof -i :9999
   
   # Kill the process (replace PID with the process ID from above)
   kill PID
   
   # Or force kill if needed
   kill -9 PID
   ```

2. **Use a different port**:
   ```bash
   # On the receiver
   python direct_receiver.py --display --metrics-port 8001 --port 9998
   
   # On the sender (use the same port)
   python direct_sender.py --ip RECEIVER_IP --video ../video/zidane.mp4 --metrics-port 8000 --port 9998
   ```

### Connection refused errors

If you see errors like:
```
Error connecting to sender metrics API: HTTPConnectionPool(host='192.168.2.120', port=8000): Max retries exceeded with url: /metrics (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7afc20369eb0>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

This means the script cannot connect to the metrics API. Check:

1. **Verify IP addresses are correct**:
   - Make sure you're using the correct IP address for the sender
   - On the sender, run `ip addr show` to confirm its IP address

2. **Ensure metrics ports are enabled**:
   - Confirm the sender is running with `--metrics-port 8000`
   - Confirm the receiver is running with `--metrics-port 8001`

3. **Check for firewall issues**:
   - Temporarily disable firewall to test: `sudo ufw disable` (Ubuntu)
   - Or add rules to allow the ports: `sudo ufw allow 8000/tcp` and `sudo ufw allow 8001/tcp`

4. **Run on the same machine for testing**:
   - For testing, you can run both sender and receiver on the same machine
   - Use `--sender-ip localhost` and `--receiver-ip localhost`

### No traffic control settings detected

If you see:
```
Warning: No traffic control settings detected
Make sure auto_tc_control.sh is running
```

This means the script cannot detect any traffic control settings. Check:

1. **Verify traffic control is running**:
   - Make sure you've started `auto_tc_control.sh` on the sender
   - Run it with sudo: `sudo ./auto_tc_control.sh`

2. **Check if traffic control is working**:
   - On the sender, run: `sudo tc qdisc show`
   - You should see output containing rate, delay, and loss settings

3. **Run traffic control manually**:
   - If auto_tc_control.sh isn't working, try manual settings:
   ```bash
   # Replace eth0 with your network interface
   sudo tc qdisc add dev eth0 root netem rate 5mbit delay 100ms loss 1%
   ```

4. **Find your network interface**:
   - Run: `ip route get 8.8.8.8`
   - Look for "dev XXX" in the output to identify your interface

### Reset Traffic Control

After stopping everything, reset traffic control:
```bash
sudo tc qdisc del dev INTERFACE root
```
Replace INTERFACE with your network interface (e.g., eth0, wlan0).