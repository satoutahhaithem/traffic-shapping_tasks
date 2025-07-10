# Video Streaming with Traffic Control

## Recommended Workflow: Synchronized `tc` Measurement

This is the most accurate way to measure the impact of `tc`-based traffic shaping. It uses a single, all-in-one script on the receiver to both apply the network conditions and measure the performance, ensuring perfect synchronization.

### **On the Sender PC (`192.168.2.169`)**

1.  **Start the Video Sender:**
    Open a terminal and run the following command to start sending the video stream to the receiver.
    ```bash
    python3 WebRTC_Streaming/direct_sender.py --ip 192.168.2.120
    ```

### **On the Receiver PC (`192.168.2.120`)**

1.  **Start the Video Receiver:**
    Open a new terminal and start the video receiver to display the incoming stream.
    ```bash
    python3 WebRTC_Streaming/direct_receiver.py --display
    ```

2.  **Start the All-in-One `tc` Test Script:**
    Finally, open one more terminal and run the all-in-one script with `sudo`. This will control the traffic shaping and measure the performance in a synchronized loop.
    ```bash
    sudo python3 WebRTC_Streaming/measurement_scripts/tc_all_in_one_synced.py --sender-ip 192.168.2.169
    ```

You should now see the performance graphs updating in real-time, with the measured values more closely matching the commanded values.

---

## Other Options

... (rest of the README content)