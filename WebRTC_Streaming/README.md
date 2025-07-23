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

The process involves starting the sender and receiver, and then choosing a method on the receiver to apply traffic shaping.

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
At this point, the video stream is running but with no traffic shaping applied.

---

### **Step 2: Apply Traffic Shaping (Choose One Method)**

You can apply network constraints using either an automated script that cycles through presets or a manual script for interactive control. **Run one of the following in a new terminal on the Receiver PC.**

#### **Method A: Automated Measurement with `tc_all_in_one_synced.py`**

This script automatically applies a series of predefined network conditions and plots the performance in real-time. This is ideal for standardized testing.

1.  Find the **Sender's IP address**.
2.  Run the script with `sudo`:
    ```bash
    sudo python3 measurement_scripts/tc_all_in_one_synced.py --sender-ip <SENDER_IP>
    ```
    A Matplotlib window will appear showing real-time graphs of bitrate and latency as the script cycles through different network conditions.

#### **Method B: Manual Control with `dynamic_tc_control.sh`**

This script provides an interactive command-line menu to set custom network conditions on the fly. This is useful for experimentation.

1.  Make the script executable:
    ```bash
    chmod +x traffic_shapping/dynamic_tc_control.sh
    ```
2.  Run the script with `sudo`:
    ```bash
    sudo ./traffic_shapping/dynamic_tc_control.sh
    ```
3.  Use the menu to set a custom rate, delay, and packet loss. The changes will affect the video stream immediately. You can run the measurement script from Method A in a separate terminal to visualize the impact of your manual changes.