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

## 5. Testing Workflow for PC 192.168.2.120 (Receiver)

Here is the precise command sequence for your debugging session.

---
### **On the SENDER PC (192.168.2.169)**

You will need **two terminals** on this machine.

**Terminal 1: Start the Video Sender**
1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the sender script, pointing to the receiver's IP address:
    ```bash
    python3 direct_sender.py --ip 192.168.2.120
    ```

**Terminal 2: Apply Traffic Shaping**
1.  Make the scripts executable:
    ```bash
    chmod +x traffic_shapping/static_tc_control.sh
    chmod +x traffic_shapping/auto_tc_control.sh
    ```
2.  Run your chosen traffic shaping script with `sudo`:
    ```bash
    # For automatic, cycling conditions:
    sudo ./traffic_shapping/auto_tc_control.sh

    # OR for manual, interactive control:
    sudo ./traffic_shapping/static_tc_control.sh
    ```

---
### **On the RECEIVER PC (192.168.2.120)**

You will need **two terminals** on this machine.

**Terminal 1: Start the Video Receiver**
1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the receiver script to display the incoming video:
    ```bash
    python3 direct_receiver.py --display
    ```

**Terminal 2: Start the Performance Monitor**
1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the performance monitor script, pointing it to the sender's IP address:
    ```bash
    python3 measurement/performance_monitor.py --sender-ip 192.168.2.169
    ```

This setup will show you the video stream on the receiver and a real-time graph of the performance, allowing you to debug the system effectively.