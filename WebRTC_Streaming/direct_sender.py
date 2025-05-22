#!/usr/bin/env python3
"""
Video Sender Script

This script captures video from a file and sends it to a receiver over a TCP socket.
It displays real-time traffic statistics in the terminal.

Usage:
    python direct_sender.py --ip RECEIVER_IP --video VIDEO_PATH

Author: Roo AI Assistant
Date: May 2025
"""

import cv2
import socket
import pickle
import struct
import time
import argparse
import numpy as np
import threading
from collections import deque

# Traffic statistics variables
bytes_sent = 0       # Total bytes sent
packets_sent = 0     # Total packets sent
frame_sizes = deque(maxlen=30)  # Last 30 frame sizes for averaging
frame_times = deque(maxlen=30)  # Last 30 frame processing times
start_time = time.time()        # When the script started

# Frame buffering for smoother transmission
frame_buffer = deque(maxlen=30)  # Buffer to store frames
buffer_lock = threading.Lock()   # Lock for thread-safe buffer access
send_thread = None               # Thread for sending frames
running = True                   # Flag to control threads

def print_stats():
    """
    Print current traffic statistics to the terminal.
    Clears the terminal and shows a formatted display of all metrics.
    """
    global bytes_sent, packets_sent, start_time, frame_sizes, frame_times, frame_buffer
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Calculate rates
    bytes_sent_rate = bytes_sent / elapsed if elapsed > 0 else 0
    packets_sent_rate = packets_sent / elapsed if elapsed > 0 else 0
    
    # Calculate averages
    avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
    
    # Calculate buffer fullness
    buffer_percent = (len(frame_buffer) / frame_buffer.maxlen * 100) if frame_buffer.maxlen > 0 else 0
    
    # Clear terminal and print stats
    print("\033c", end="")  # Clear terminal
    
    print("\n" + "="*50)
    print("VIDEO SENDER - TRAFFIC MONITOR")
    print("="*50)
    print(f"Running time: {elapsed:.1f} seconds")
    print(f"Connected to: {receiver_ip}:{receiver_port}")
    print("\nTRAFFIC STATISTICS:")
    print(f"  Bytes sent:     {bytes_sent} bytes ({bytes_sent/1024/1024:.2f} MB)")
    print(f"  Packets sent:   {packets_sent}")
    print(f"  Send rate:      {bytes_sent_rate/1024/1024:.2f} MB/s")
    print(f"  Packet rate:    {packets_sent_rate:.1f} packets/s")
    print("\nVIDEO STATISTICS:")
    print(f"  Resolution:     {frame_width}x{frame_height}")
    print(f"  Avg frame size: {avg_frame_size/1024:.1f} KB")
    print(f"  FPS:            {fps:.1f}")
    print(f"  Quality:        {jpeg_quality}%")
    print(f"  Buffer fullness: {len(frame_buffer)}/{frame_buffer.maxlen} ({buffer_percent:.1f}%)")
    print("="*50)

def send_frame(client_socket, frame, quality=90):
    """
    Compress and send a frame to the receiver
    
    Args:
        client_socket: Socket connected to the receiver
        frame: The video frame to send
        quality: JPEG compression quality (1-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    global bytes_sent, packets_sent, frame_sizes, frame_times
    
    # Record start time for performance measurement
    start_process = time.time()
    
    # Encode frame as JPEG (compress the image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
    
    if not result:
        print("Error encoding frame")
        return False
    
    # Serialize frame for network transmission
    data = pickle.dumps(encoded_frame)
    
    # Get the size of the data
    size = len(data)
    frame_sizes.append(size)
    
    try:
        # Send the size of the data first (4 bytes)
        client_socket.sendall(struct.pack(">L", size))
        
        # Send the actual frame data
        client_socket.sendall(data)
        
        # Update traffic statistics
        bytes_sent += size + 4  # +4 for the size header
        packets_sent += 1
        
        # Calculate frame processing time
        frame_times.append(time.time() - start_process)
        
        return True
    
    except Exception as e:
        print(f"Error sending frame: {e}")
        return False

def send_frames_thread(client_socket, fps):
    """
    Thread function to continuously send frames from the buffer
    
    Args:
        client_socket: Socket connected to the receiver
        fps: Target frames per second
    """
    global frame_buffer, running
    
    target_frame_time = 1.0 / fps
    last_frame_time = time.time()
    
    while running:
        # Get frame from buffer
        frame = None
        with buffer_lock:
            if frame_buffer:
                frame = frame_buffer.popleft()
        
        if frame is not None:
            # Send the frame
            success = send_frame(client_socket, frame, jpeg_quality)
            
            if not success:
                # Try to reconnect if sending fails
                try:
                    client_socket.close()
                    new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_socket.connect((receiver_ip, receiver_port))
                    client_socket = new_socket
                    
                    # Re-send video info
                    video_info = {
                        "width": frame_width,
                        "height": frame_height,
                        "fps": fps,
                        "quality": jpeg_quality
                    }
                    info_data = pickle.dumps(video_info)
                    client_socket.sendall(struct.pack(">L", len(info_data)))
                    client_socket.sendall(info_data)
                    
                    print("Reconnected to receiver")
                except:
                    print("Failed to reconnect")
                    time.sleep(1.0)  # Wait before trying again
            
            # Control frame rate for smooth transmission
            current_time = time.time()
            elapsed = current_time - last_frame_time
            sleep_time = max(0, target_frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()
        else:
            # No frames in buffer, wait a bit
            time.sleep(0.01)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Sender")
    parser.add_argument("--ip", default="192.168.2.169", help="Receiver IP address")
    parser.add_argument("--port", type=int, default=9999, help="Receiver port")
    parser.add_argument("--video", default="../video/zidane.mp4", help="Video file path")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scale factor")
    parser.add_argument("--fps", type=float, default=0, help="Target FPS (0=use video's FPS)")
    parser.add_argument("--buffer", type=int, default=30, help="Frame buffer size")
    
    args = parser.parse_args()
    
    # Set variables from arguments
    receiver_ip = args.ip
    receiver_port = args.port
    video_path = args.video
    jpeg_quality = args.quality
    scale_factor = args.scale
    target_fps_arg = args.fps
    frame_buffer = deque(maxlen=args.buffer)
    
    # Create TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to receiver
        print(f"Connecting to {receiver_ip}:{receiver_port}...")
        client_socket.connect((receiver_ip, receiver_port))
        print("Connected to receiver")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            client_socket.close()
            exit()
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use target FPS from argument or video's FPS
        fps = target_fps_arg if target_fps_arg > 0 else video_fps
        
        print(f"Video opened: {frame_width}x{frame_height} at {fps} FPS")
        
        # Send video info to receiver
        video_info = {
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "quality": jpeg_quality
        }
        info_data = pickle.dumps(video_info)
        client_socket.sendall(struct.pack(">L", len(info_data)))
        client_socket.sendall(info_data)
        
        # Start sending thread
        send_thread = threading.Thread(target=send_frames_thread, args=(client_socket, fps))
        send_thread.daemon = True
        send_thread.start()
        
        # Start reading frames
        frame_count = 0
        last_stats_time = time.time()
        
        # Pre-fill buffer
        print("Pre-filling buffer...")
        while len(frame_buffer) < frame_buffer.maxlen and running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Resize frame if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height))
                
            # Add frame to buffer
            with buffer_lock:
                frame_buffer.append(frame)
                
            print(f"Buffer: {len(frame_buffer)}/{frame_buffer.maxlen}", end="\r")
            
        print("\nBuffer filled, starting transmission")
        
        # Main loop - read frames and add to buffer
        while running:
            # Read a frame from the video
            ret, frame = cap.read()
            
            if not ret:
                # Loop back to the beginning of the video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize frame if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Add frame to buffer if there's space
            with buffer_lock:
                if len(frame_buffer) < frame_buffer.maxlen:
                    frame_buffer.append(frame)
            
            frame_count += 1
            
            # Print statistics every second
            current_time = time.time()
            if current_time - last_stats_time >= 1.0:
                print_stats()
                last_stats_time = current_time
            
            # Control reading rate to avoid filling buffer too quickly
            # Only sleep if buffer is getting full
            with buffer_lock:
                buffer_fullness = len(frame_buffer) / frame_buffer.maxlen
                
            if buffer_fullness > 0.8:  # Buffer is more than 80% full
                time.sleep(0.01)  # Short sleep to let sending thread catch up
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up resources
        running = False
        if send_thread and send_thread.is_alive():
            send_thread.join(timeout=1.0)
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        
        client_socket.close()
        print("Socket closed")