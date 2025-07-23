# Real-Time Video Streaming with Traffic Shaping and Measurement

## 1. Overview

This project provides a complete workflow for testing real-time video streaming performance. It allows you to apply specific network constraints (traffic shaping) on the Sender PC and then measure the impact of those constraints on the Receiver PC.

## 2. Core Components

*   **`direct_sender.py`**: Captures and streams video from a webcam.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/`**: Contains scripts to apply network constraints on the **Sender PC**.
    *   `static_tc_control.sh`: Manually set specific network conditions.
    *   `auto_tc_control.sh`: Automatically cycle through predefined network conditions.
*   **`measurement/`**: Contains the script to measure performance on the **Receiver PC**.
    *   `performance_monitor.py`: Measures and plots bitrate and latency in real-time.

## 3. Prerequisites

*   **OS:** A Linux distribution (e.g., Ubuntu 20.04+)
*   **Python:** Python 3.8 or newer.
*   **iproute2:** The package containing the `tc` utility.
*   **A webcam:** Connected to the Sender PC.

## 4. Setup Instructions

Follow these steps on both the **Sender** and **Receiver** PCs.

**Step 1: Create and Activate a Python Virtual Environment**
```bash
cd /path/to/traffic-shapping_tasks/WebRTC_Streaming
python3 -m venv venv
source venv/bin/activate
```

**Step 2: Install Required Libraries**
```bash
pip install -r requirements.txt
```

## 5. Testing Workflow

The workflow involves three main steps, executed across terminals on both the Sender and Receiver PCs.

### **Step 1: Start the Receiver**

**On the Receiver PC (192.168.2.120):**
1.  Activate the virtual environment.
2.  Start the receiver script:
    ```bash
    python3 direct_receiver.py --display
    ```

### **Step 2: Start the Sender and Apply Traffic Shaping**

**On the Sender PC (192.168.2.169):**

**Terminal 1: Start the Video Sender**
1.  Activate the virtual environment.
2.  Start the sender script:
    ```bash
    python3 direct_sender.py --ip 192.168.2.120
    ```

**Terminal 2: Apply Traffic Shaping (Choose One)**
1.  Make the scripts executable:
    ```bash
    chmod +x traffic_shapping/static_tc_control.sh
    chmod +x traffic_shapping/auto_tc_control.sh
    ```
2.  Run either the manual or automatic script with `sudo`:
    ```bash
    # For manual control
    sudo ./traffic_shapping/static_tc_control.sh

    # For automatic control
    sudo ./traffic_shapping/auto_tc_control.sh
    ```

### **Step 3: Measure the Performance**

**On the Receiver PC (192.168.2.120):**

**Terminal 2: Start the Performance Monitor**
1.  Activate the virtual environment.
2.  Run the performance monitor script, pointing it to the Sender's IP:
    ```bash
    python3 measurement/performance_monitor.py --sender-ip 192.168.2.169
    ```

A Matplotlib window will appear on the receiver, showing real-time graphs of the measured bitrate, latency, and packet loss, allowing you to see the direct impact of the traffic shaping rules you applied on the sender.