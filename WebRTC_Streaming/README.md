# Real-Time Video Streaming with Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and the Linux `tc` (traffic control) utility to apply specific bandwidth limitations.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**.

## 2. Core Components

*   **`direct_sender.py`**: Captures video from a webcam and streams it to the receiver.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/dynamic_tc_control.sh`**: An interactive bash script to manually apply and manage traffic shaping rules on the receiver.
*   **`measurement_scripts/tc_all_in_one_synced.py`**: An automated script that runs on the receiver to apply a sequence of traffic shaping rules and plot the resulting performance in real-time.

## 3. Prerequisites

*   **OS:** A Linux distribution (e.g., Ubuntu 20.04+)
*   **Python:** Python 3.8 or newer.
*   **iproute2:** The package containing the `tc` utility (installed by default on most Linux systems).
*   **A webcam:** Connected to the Sender PC.

## 4. Setup Instructions

Follow these steps on both the **Sender** and **Receiver** PCs.

**Step 1: Create and Activate a Python Virtual Environment**
```bash
# Navigate to the WebRTC_Streaming directory
cd /path/to/traffic-shapping_tasks/WebRTC_Streaming

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate
```

**Step 2: Install Required Libraries**
```bash
pip install -r requirements.txt
```

## 5. How to Run the Test

The process involves starting the sender and receiver, applying traffic shaping rules, and then verifying that the rules are active.

### **Step 1: Start the Video Stream**

**On the Sender PC:**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Find the **Receiver's IP address** (`hostname -I`).
3.  Start the sender script, pointing it to the receiver's IP.
    ```bash
    python3 direct_sender.py --ip <RECEIVER_IP>
    ```

**On the Receiver PC:**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the receiver script in a dedicated terminal.
    ```bash
    python3 direct_receiver.py --display
    ```

---

### **Step 2: Apply Traffic Shaping (Choose One Method)**

Apply network constraints using either an automated script or a manual one. **Run one of the following in a new terminal on the Receiver PC.**

#### **Method A: Automated Measurement**
This script automatically applies a series of network conditions and plots the performance.
1.  Find the **Sender's IP address**.
2.  Run the script with `sudo`:
    ```bash
    sudo python3 measurement_scripts/tc_all_in_one_synced.py --sender-ip <SENDER_IP>
    ```

#### **Method B: Manual Control**
This script provides an interactive menu to set custom network conditions.
1.  Make the script executable: `chmod +x traffic_shapping/dynamic_tc_control.sh`
2.  Run the script with `sudo`:
    ```bash
    sudo ./traffic_shapping/dynamic_tc_control.sh
    ```
3.  Use the menu to set a custom rate, delay, and packet loss.

---

### **Step 3: Verify Traffic Shaping Rules**

After applying traffic shaping using either method, it is crucial to verify that the rules have been correctly applied by the system.

**On the Receiver PC**, open a new terminal and use the following command to inspect the active `tc` queueing disciplines (`qdisc`):

```bash
tc -s qdisc show dev <INTERFACE>
```
*Replace `<INTERFACE>` with your network interface name (e.g., `eth0`, `wlp0s20f3`). You can find it using `ip a`.*

**What to look for in the output:**

*   **`qdisc netem ...`**: This confirms that the Network Emulator (`netem`) is active.
*   **`limit`**: The maximum number of packets the qdisc can hold.
*   **`delay`**: The configured latency (e.g., `100.0ms`).
*   **`rate`**: The configured bandwidth limit (e.g., `1Mbit`).
*   **`loss`**: The configured packet loss percentage (e.g., `10%`).
*   **`dropped`**: A counter for the number of packets dropped due to the rate limit. If this number is increasing, your traffic shaping is actively limiting the bandwidth.
*   **`overlimits`**: A counter for the number of times the traffic exceeded the allocated bandwidth, causing packets to be delayed or dropped.

By checking this output, you can confirm that your traffic shaping rules are not just set, but are actively working and affecting the network traffic as intended.