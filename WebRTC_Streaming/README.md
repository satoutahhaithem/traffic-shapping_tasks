# Real-Time Video Streaming with Synchronized Traffic Shaping

## 1. Overview

This project provides a streamlined solution for testing real-time video streaming performance under custom network conditions. It uses WebRTC for peer-to-peer video transmission and the Linux `tc` (traffic control) utility to apply specific bandwidth limitations.

The workflow is designed for a two-PC setup: a **Sender** and a **Receiver**. The core of the project is a synchronized measurement script that runs on the Receiver, which simultaneously manages the traffic shaping rules and records performance metrics, ensuring that the measured impact directly corresponds to the applied network conditions.

The key scripts are:
*   `direct_sender.py`: Captures video from a webcam and streams it to the receiver.
*   `direct_receiver.py`: Receives and displays the video stream.
*   `measurement_scripts/tc_all_in_one_synced.py`: The main control script that applies `tc` rules and plots performance in real-time.

## 2. Prerequisites

Before you begin, ensure you have the following installed on both the Sender and Receiver PCs:

*   **OS:** A Linux distribution (e.g., Ubuntu 20.04+)
*   **Python:** Python 3.8 or newer.
*   **iproute2:** The package that contains the `tc` utility. This is installed by default on most Linux systems. You can verify with `tc -V`.
*   **A webcam:** Connected to the Sender PC.

## 3. Setup Instructions

Follow these steps on both the **Sender** and **Receiver** PCs to prepare the environment.

**Step 1: Create a Python Virtual Environment**

It is highly recommended to use a virtual environment to avoid conflicts with system-wide packages.

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

With the virtual environment activated, install the necessary Python packages using the provided `requirements.txt` file.

```bash
# Ensure you are in the WebRTC_Streaming directory
pip install -r requirements.txt
```

The environment is now ready on both machines.

## 4. Usage: Running the Streaming Test

To run the synchronized test, you will need three separate terminal windows on your Receiver PC and one on your Sender PC.

**Step 1: Identify IP Addresses**

*   Find the IP address of the **Receiver PC**. You will need this for the sender script.
*   Find the IP address of the **Sender PC**. You will need this for the measurement script.

You can find the IP address on most Linux systems with the command `ip a` or `hostname -I`. For this guide, we will use the following placeholders:
*   **SENDER_IP**: `192.168.1.10`
*   **RECEIVER_IP**: `192.168.1.20`

**Remember to replace these with your actual IP addresses.**

---

### **On the Sender PC**

**Terminal 1: Start the Video Sender**

1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the `direct_sender.py` script, pointing it to the Receiver's IP address.

```bash
python3 direct_sender.py --ip RECEIVER_IP
# Example: python3 direct_sender.py --ip 192.168.1.20
```
The sender will start capturing video from the webcam and attempt to stream it.

---

### **On the Receiver PC**

**Terminal 1: Start the Video Receiver**

1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the `direct_receiver.py` script with the `--display` flag to show the incoming video.

```bash
python3 direct_receiver.py --display
```
A window titled "Video" should appear. It will remain black until the measurement script starts applying network rules.

**Terminal 2: Start the Synchronized Measurement Script**

1.  Activate the virtual environment: `source venv/bin/activate`
2.  Run the `tc_all_in_one_synced.py` script with `sudo` powers, pointing it to the Sender's IP address. `sudo` is required for `tc` to modify network interfaces.

```bash
sudo python3 measurement_scripts/tc_all_in_one_synced.py --sender-ip SENDER_IP
# Example: sudo python3 measurement_scripts/tc_all_in_one_synced.py --sender-ip 192.168.1.10
```

## 5. Expected Output

Once all scripts are running, you should observe the following:

1.  **On the Sender:** The terminal will log messages indicating it is sending video frames.
2.  **On the Receiver:**
    *   The "Video" window will display the live stream from the sender's webcam.
    *   A Matplotlib window titled "Real-time TC Performance" will appear, showing two graphs that update every few seconds:
        *   The top graph plots the **Commanded vs. Measured Bitrate**.
        *   The bottom graph plots the **Commanded vs. Measured Latency**.
    *   The terminal running the measurement script will print the current `tc` rules being applied and the corresponding performance metrics.

This setup allows you to accurately observe how the network conditions you command with `tc` affect the performance of the WebRTC video stream in real-time.