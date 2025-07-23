# Real-Time Video Streaming with Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and the Linux `tc` (traffic control) utility to apply specific bandwidth limitations.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**.

The key scripts are:
*   `direct_sender.py`: Captures video from a webcam and streams it to the receiver.
*   `direct_receiver.py`: Receives and displays the video stream.
*   `measurement_scripts/tc_all_in_one_synced.py`: An automated script for running synchronized performance measurements.
*   `traffic_shapping/dynamic_tc_control.sh`: An interactive bash script for manually applying traffic shaping rules.

## 2. Prerequisites

Before you begin, ensure you have the following installed on both the Sender and Receiver PCs:

*   **OS:** A Linux distribution (e.g., Ubuntu 20.04+)
*   **Python:** Python 3.8 or newer.
*   **iproute2:** The package that contains the `tc` utility. This is installed by default on most Linux systems. You can verify with `tc -V`.
*   **A webcam:** Connected to the Sender PC.

## 3. Setup Instructions

Follow these steps on both the **Sender** and **Receiver** PCs to prepare the environment.

**Step 1: Create a Python Virtual Environment**

```bash
# Navigate to the project directory
cd /path/to/traffic-shapping_tasks/WebRTC_Streaming

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
*Your terminal prompt should now be prefixed with `(venv)`.*

**Step 2: Install Required Libraries**

```bash
# Ensure you are in the WebRTC_Streaming directory
pip install -r requirements.txt
```

## 4. Usage

This project offers two primary modes of operation: **Synchronized Measurement** for automated testing and **Manual Control** for interactive adjustments.

---

### Option A: Synchronized Measurement (Recommended)

This workflow uses a Python script to automate the process of applying network conditions and measuring performance.

**Step 1: Identify IP Addresses**
*   **SENDER_IP**: The IP address of the Sender PC.
*   **RECEIVER_IP**: The IP address of the Receiver PC.

#### **On the Sender PC**
1.  Activate the virtual environment: `source venv/bin/activate`
2.  Start the sender:
    ```bash
    python3 direct_sender.py --ip RECEIVER_IP
    ```

#### **On the Receiver PC**
1.  Activate the virtual environment: `source venv/bin/activate`
2.  In a new terminal, start the receiver:
    ```bash
    python3 direct_receiver.py --display
    ```
3.  In another terminal, start the measurement script:
    ```bash
    sudo python3 measurement_scripts/tc_all_in_one_synced.py --sender-ip SENDER_IP
    ```

---

### Option B: Manual Control with Bash Script

This workflow uses an interactive bash script to manually apply traffic shaping rules.

#### **On the Sender PC**
Follow the same steps as in Option A to start the video sender.

#### **On the Receiver PC**

**Terminal 1: Start the Video Receiver**
Follow the same steps as in Option A to start the video receiver.

**Terminal 2: Run the Traffic Shaping Script**
1.  Make the script executable:
    ```bash
    chmod +x traffic_shapping/dynamic_tc_control.sh
    ```
2.  Run the script with `sudo`:
    ```bash
    sudo ./traffic_shapping/dynamic_tc_control.sh
    ```
3.  You will see a menu allowing you to set network conditions (rate, delay, loss), view stats, or reset the configuration.

## 5. Expected Output

*   **For Synchronized Measurement:** A Matplotlib window will appear showing real-time graphs of commanded vs. measured bitrate and latency.
*   **For Manual Control:** The interactive script will prompt you for traffic shaping values. The video window will show the stream quality changing in response to your adjustments.