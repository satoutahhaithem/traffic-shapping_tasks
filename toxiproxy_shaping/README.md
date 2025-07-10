# Reliable Traffic Shaping with Toxiproxy

This guide provides a step-by-step process for setting up a reliable video streaming and traffic shaping test between two PCs using Toxiproxy.

## Overview

This solution uses Toxiproxy to create a controlled network environment where you can accurately measure the impact of network conditions on a video stream. The sender sends video to a Toxiproxy server on the receiver, which then forwards it to the receiver application while applying your specified network conditions.

---

## **Part 1: Initial Setup**

### **On Both Sender and Receiver PCs**

1.  **Install Python Dependencies:**
    Make sure you have the necessary Python libraries installed.
    ```bash
    pip install -r WebRTC_Streaming/requirements.txt
    pip install toxiproxy-client
    ```

### **On the Receiver PC (`192.168.2.120`) Only**

1.  **Install Toxiproxy Server:**
    The Toxiproxy server is not a Python package and must be installed manually.

    ```bash
    # Download the correct binary for your system (this example is for x86_64 Linux)
    wget https://github.com/Shopify/toxiproxy/releases/download/v2.5.0/toxiproxy-server-linux-amd64

    # Make the binary executable
    chmod +x toxiproxy-server-linux-amd64

    # Move the binary to your system's PATH
    sudo mv toxiproxy-server-linux-amd64 /usr/local/bin/toxiproxy-server
    ```

---

## **Part 2: Running the All-in-One Test**

This new, simplified approach uses a single script on the receiver to both control the traffic shaping and measure the performance, ensuring perfect synchronization.

### **On the Sender PC (`192.168.2.169`)**

1.  **Start the Video Sender:**
    Open a terminal and run the following command to start sending the video stream to the receiver's Toxiproxy server.
    ```bash
    python3 WebRTC_Streaming/direct_sender.py --ip 192.168.2.120 --port 8666
    ```

### **On the Receiver PC (`192.168.2.120`)**

1.  **Start the Toxiproxy Server:**
    Open a new terminal and start the Toxiproxy server.
    ```bash
    toxiproxy-server
    ```

2.  **Start the Video Receiver:**
    Open another new terminal and start the video receiver to display the incoming stream.
    ```bash
    python3 WebRTC_Streaming/direct_receiver.py --display
    ```

3.  **Start the All-in-One Test Script:**
    Finally, open one more terminal and run the all-in-one script. This will control the traffic shaping and measure the performance in a synchronized loop.
    ```bash
    python3 toxiproxy_shaping/toxiproxy_all_in_one.py --sender-ip 192.168.2.169
    ```

You should now see the performance graphs updating in real-time, with the measured values closely matching the commanded values.