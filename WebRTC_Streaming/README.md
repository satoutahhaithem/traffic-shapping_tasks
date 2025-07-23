# Real-Time Video Streaming with Sender-Side Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and applies traffic shaping on the **Sender PC** to simulate a constrained uplink connection.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**.

## 2. Core Components

*   **`direct_sender.py`**: Captures video from a webcam and streams it to the receiver.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`traffic_shapping/static_tc_control.sh`**: An interactive bash script to manually apply and manage traffic shaping rules.
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

The process involves starting the receiver, then starting the sender, and finally applying traffic shaping **on the Sender PC**.

### **Step 1: Start the Video Receiver**

**On the Receiver PC (192.168.2.120):**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the receiver script in a dedicated terminal.
    ```bash
    python3 direct_receiver.py --display
    ```

---

### **Step 2: Start the Video Sender and Apply Traffic Shaping**

**On the Sender PC (192.168.2.169)**, you will need two terminals.

**Terminal 1: Start the Video Sender**
1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the sender script, pointing it to the receiver's IP.
    ```bash
    python3 direct_sender.py --ip 192.168.2.120
    ```

**Terminal 2: Apply Traffic Shaping**
Choose one of the following methods to apply traffic shaping on the sender's network interface.

#### **Method A: Manual Control with `static_tc_control.sh`**
1.  **Make the script executable:**
    ```bash
    chmod +x traffic_shapping/static_tc_control.sh
    ```
2.  **Run the interactive script with `sudo`:**
    ```bash
    sudo ./traffic_shapping/static_tc_control.sh
    ```
3.  **Use the menu** to set your desired network conditions.

#### **Method B: Automatic Control with `auto_tc_control.sh`**
1.  **Make the script executable:**
    ```bash
    chmod +x traffic_shapping/auto_tc_control.sh
    ```
2.  **Run the script with `sudo`:**
    ```bash
    sudo ./traffic_shapping/auto_tc_control.sh
    ```

### **Step 3: Verify Traffic Shaping Rules**

To verify that the rules are active, run the following command in a terminal on the **Sender PC**:

```bash
tc -s qdisc show dev <INTERFACE>
```
*Replace `<INTERFACE>` with your network interface name (e.g., `eth0`, `wlp0s20f3`).*