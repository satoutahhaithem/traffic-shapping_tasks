# Real-Time Video Streaming and Traffic Shaping Analysis

## Project Overview

This repository contains a streamlined set of tools for testing and analyzing real-time video streaming performance under various network conditions. The primary focus is on a WebRTC-based streaming solution coupled with Linux `tc` (traffic control) for synchronized performance measurement.

This setup allows for precise testing of how network limitations, such as constrained bandwidth and added latency, affect the quality of a live video stream.

## Core Functionality

The main workflow is located in the `WebRTC_Streaming` directory and consists of three key scripts:

*   **`direct_sender.py`**: Streams video from a webcam.
*   **`direct_receiver.py`**: Receives and displays the video stream.
*   **`measurement_scripts/tc_all_in_one_synced.py`**: A synchronized script that applies traffic shaping rules and plots performance metrics in real-time.

## Getting Started

For detailed setup instructions, prerequisites, and step-by-step usage guides, please refer to the README file within the core project directory:

**[Go to WebRTC Streaming Instructions](./WebRTC_Streaming/README.md)**