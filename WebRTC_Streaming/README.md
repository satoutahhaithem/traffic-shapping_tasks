# Real-Time Video Streaming with Manual Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and an interactive bash script to manually apply traffic shaping rules using the Linux `tc` utility.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**.

## 2. Core Components

*   **`direct_sender.py`**: Captures video from a webcam and streams it to the receiver.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/dynamic_tc_control.sh`**: An interactive bash script to manually apply and manage traffic shaping rules on the receiver.

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

The process involves starting the sender and receiver, and then using the interactive script on the receiver to apply and verify traffic shaping rules.

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

### **Step 2: Apply and Verify Traffic Shaping**

**On the Receiver PC**, in a new terminal:

1.  **Navigate to the correct directory:**
    ```bash
    cd /path/to/traffic-shapping_tasks/WebRTC_Streaming
    ```
2.  **Make the script executable:**
    ```bash
    chmod +x traffic_shapping/dynamic_tc_control.sh
    ```
3.  **Run the interactive script with `sudo`:**
    ```bash
    sudo ./traffic_shapping/dynamic_tc_control.sh
    ```
4.  **Use the menu** to set your desired network conditions (rate, delay, packet loss).
5.  **Verify the rules** by selecting the "Show current stats" option in the script's menu. This will execute `tc -s qdisc show` and display the active rules and their statistics, allowing you to confirm that your settings are active and affecting traffic.