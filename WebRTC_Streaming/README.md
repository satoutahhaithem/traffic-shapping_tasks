# Real-Time Video Streaming with Receiver-Side Traffic Shaping and Measurement

## 1. Overview

This project provides a complete and simplified workflow for testing real-time video streaming performance. It allows you to apply specific network constraints (traffic shaping) and measure the impact of those constraints **on the same machine (the Receiver)**. This removes network complexity and makes debugging more reliable.

## 2. Core Components

*   **`direct_sender.py`**: Captures and streams video from a webcam.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/`**: Contains scripts to apply network constraints on the **Receiver PC**.
    *   `static_tc_control.sh`: Manually set specific network conditions.
    *   `auto_tc_control.sh`: Automatically cycle through predefined network conditions.
*   **`measurement/`**: Contains the script to measure performance on the **Receiver PC**.
    *   `performance_monitor.py`: Measures and plots commanded vs. measured values in real-time.

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

The workflow is now entirely focused on the Receiver PC for traffic control and measurement.

### **Step 1: Start the Sender**

**On the Sender PC (192.168.2.169):**
1.  Activate the virtual environment.
2.  Start the sender script, pointing it to the receiver's IP:
    ```bash
    python3 direct_sender.py --ip 192.168.2.120
    ```

---
### **Step 2: Start Receiver, Apply Traffic Shaping, and Measure**

**On the Receiver PC (192.168.2.120):**

You will need **three terminals** on this machine.

**Terminal 1: Start the Video Receiver**
1.  Activate the virtual environment.
2.  Start the receiver script:
    ```bash
    python3 direct_receiver.py --display
    ```

**Terminal 2: Apply Traffic Shaping (Choose One)**
1.  Make the scripts executable:
    ```bash
    chmod +x traffic_shapping/static_tc_control.sh
    chmod +x traffic_shapping/auto_tc_control.sh
    ```
2.  Run your chosen traffic shaping script with `sudo`:
    ```bash
    # For manual, interactive control:
    sudo ./traffic_shapping/static_tc_control.sh

    # OR for automatic, cycling conditions:
    sudo ./traffic_shapping/auto_tc_control.sh
    ```

**Terminal 3: Start the Performance Monitor**
1.  Activate the virtual environment.
2.  Run the performance monitor script, pointing it to the sender's IP:
    ```bash
    python3 measurement/performance_monitor.py --sender-ip 192.168.2.169
    ```

This setup will now correctly measure and plot both the commanded `tc` rules and the resulting performance, all from the receiver's perspective.