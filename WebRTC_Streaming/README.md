# Real-Time Video Streaming with Manual and Automatic Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and provides both manual and automatic bash scripts to apply traffic shaping rules using the Linux `tc` utility.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**.

## 2. Core Components

*   **`direct_sender.py`**: Captures video from a webcam and streams it to the receiver.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/static_tc_control.sh`**: An interactive bash script to manually apply and manage traffic shaping rules on the receiver.
*   **`traffic_shapping/auto_tc_control.sh`**: An automated bash script that cycles through predefined traffic shaping presets.

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

**On the Sender PC (192.168.2.169):**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the sender script, pointing it to the receiver's IP.
    ```bash
    python3 direct_sender.py --ip 192.168.2.120
    ```

**On the Receiver PC (192.168.2.120):**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the receiver script in a dedicated terminal.
    ```bash
    python3 direct_receiver.py --display
    ```

---

### **Step 2: Apply Traffic Shaping (Choose One Method)**

**On the Receiver PC (192.168.2.120)**, in a new terminal, choose one of the following methods:

#### **Method A: Manual Control with `static_tc_control.sh`**

This script provides an interactive menu to set custom network conditions.

1.  **Make the script executable:**
    ```bash
    chmod +x traffic_shapping/static_tc_control.sh
    ```
2.  **Run the interactive script with `sudo`:**
    ```bash
    sudo ./traffic_shapping/static_tc_control.sh
    ```
3.  **Use the menu** to set your desired network conditions (rate, delay, packet loss).

#### **Method B: Automatic Control with `auto_tc_control.sh`**

This script automatically cycles through a series of predefined network conditions.

1.  **Make the script executable:**
    ```bash
    chmod +x traffic_shapping/auto_tc_control.sh
    ```
2.  **Run the script with `sudo`:**
    ```bash
    sudo ./traffic_shapping/auto_tc_control.sh
    ```
The script will then cycle through network presets, from "VERY POOR" to "EXCELLENT", changing every 20 seconds.

### **Step 3: Verify Traffic Shaping Rules**

After applying traffic shaping, you can verify the rules are active by using the "Show current stats" option in the `static_tc_control.sh` script, or by running the following command in a separate terminal:

```bash
tc -s qdisc show dev <INTERFACE>
```
*Replace `<INTERFACE>` with your network interface name (e.g., `eth0`, `wlp0s20f3`).*