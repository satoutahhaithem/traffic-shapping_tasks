#!/usr/bin/env python3
"""
Video Receiver Script

This script receives video frames from a sender over a TCP socket and optionally
displays them. It shows real-time traffic statistics in the terminal.

Usage:
    python direct_receiver.py [--display]

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
bytes_received = 0    # Total bytes received
packets_received = 0  # Total packets received
frame_sizes = deque(maxlen=30)  # Last 30 frame sizes for averaging
frame_times = deque(maxlen=30)  # Last 30 frame processing times
start_time = time.time()        # When the script started
frames_received = 0   # Total frames received
frames_displayed = 0  # Total frames displayed
frames_dropped = 0    # Total frames dropped
video_info = None     # Video information from sender

# Frame buffering for smoother playback
frame_buffer = deque(maxlen=60)  # Buffer to store frames (increased size)
buffer_lock = threading.Lock()   # Lock for thread-safe buffer access
buffer_thread = None             # Thread for receiving frames
running = True                   # Flag to control threads

def print_stats():
    """
    Print current traffic statistics to the terminal.
    Clears the terminal and shows a formatted display of all metrics.
    """
    global bytes_received, packets_received, start_time, frame_sizes, frame_times
    global frames_received, frames_displayed, frames_dropped, video_info, frame_buffer
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Calculate rates
    bytes_received_rate = bytes_received / elapsed if elapsed > 0 else 0
    packets_received_rate = packets_received / elapsed if elapsed > 0 else 0
    
    # Calculate averages
    avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
    
    # Calculate drop rate
    drop_rate = (frames_dropped / frames_received * 100) if frames_received > 0 else 0
    
    # Calculate buffer fullness
    buffer_percent = (len(frame_buffer) / frame_buffer.maxlen * 100) if frame_buffer.maxlen > 0 else 0
    
    # Clear terminal and print stats
    print("\033c", end="")  # Clear terminal
    
    print("\n" + "="*50)
    print("VIDEO RECEIVER - TRAFFIC MONITOR")
    print("="*50)
    print(f"Running time: {elapsed:.1f} seconds")
    print(f"Listening on: {server_ip}:{server_port}")
    print("\nTRAFFIC STATISTICS:")
    print(f"  Bytes received: {bytes_received} bytes ({bytes_received/1024/1024:.2f} MB)")
    print(f"  Packets received: {packets_received}")
    print(f"  Receive rate:   {bytes_received_rate/1024/1024:.2f} MB/s")
    print(f"  Packet rate:    {packets_received_rate:.1f} packets/s")
    print("\nVIDEO STATISTICS:")
    if video_info:
        print(f"  Resolution:     {video_info['width']}x{video_info['height']}")
        print(f"  Target FPS:     {video_info['fps']:.1f}")
        print(f"  Playback FPS:   {playback_fps:.1f}")
        print(f"  Quality:        {video_info['quality']}%")
    print(f"  Actual FPS:     {fps:.1f}")
    print(f"  Frames received: {frames_received}")
    print(f"  Frames displayed: {frames_displayed}")
    print(f"  Frames dropped:  {frames_dropped} ({drop_rate:.1f}%)")
    print(f"  Avg frame size: {avg_frame_size/1024:.1f} KB")
    print(f"  Buffer fullness: {len(frame_buffer)}/{frame_buffer.maxlen} ({buffer_percent:.1f}%)")
    print("="*50)

def receive_frame(client_socket):
    """
    Receive a frame from the sender
    
    Args:
        client_socket: Socket connected to the sender
        
    Returns:
        numpy.ndarray: The received frame, or None if failed
    """
    global bytes_received, packets_received, frame_sizes, frame_times
    global frames_received, frames_dropped
    
    # Record start time for performance measurement
    start_process = time.time()
    
    try:
        # Receive the size of the data (4 bytes)
        size_data = client_socket.recv(4)
        if not size_data:
            return None
        
        # Unpack the size value from the received bytes
        size = struct.unpack(">L", size_data)[0]
        
        # Receive the actual frame data
        data = b""
        while len(data) < size:
            # Receive in chunks to handle large frames
            packet = client_socket.recv(min(size - len(data), 4096))
            if not packet:
                return None
            data += packet
        
        # Update traffic statistics
        bytes_received += len(data) + 4  # +4 for the size header
        packets_received += 1
        frame_sizes.append(size)
        frames_received += 1
        
        # Deserialize the data
        encoded_frame = pickle.loads(data)
        
        # Decode the JPEG frame
        frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
        
        if frame is None:
            frames_dropped += 1
            return None
        
        # Calculate frame processing time
        frame_times.append(time.time() - start_process)
        
        return frame
    
    except Exception as e:
        print(f"Error receiving frame: {e}")
        frames_dropped += 1
        return None

def receive_video_info(client_socket):
    """
    Receive video information from the sender
    
    Args:
        client_socket: Socket connected to the sender
        
    Returns:
        dict: Video information (width, height, fps, quality), or None if failed
    """
    try:
        # Receive the size of the data (4 bytes)
        size_data = client_socket.recv(4)
        if not size_data:
            return None
        
        # Unpack the size value
        size = struct.unpack(">L", size_data)[0]
        
        # Receive the data
        data = b""
        while len(data) < size:
            packet = client_socket.recv(min(size - len(data), 4096))
            if not packet:
                return None
            data += packet
        
        # Deserialize the data to get video info
        video_info = pickle.loads(data)
        
        return video_info
    
    except Exception as e:
        print(f"Error receiving video info: {e}")
        return None

def buffer_frames(client_socket):
    """
    Continuously receive frames and add them to the buffer
    
    Args:
        client_socket: Socket connected to the sender
    """
    global frame_buffer, running
    
    while running:
        frame = receive_frame(client_socket)
        if frame is not None:
            with buffer_lock:
                # If buffer is full, remove oldest frame
                if len(frame_buffer) >= frame_buffer.maxlen:
                    frame_buffer.popleft()
                frame_buffer.append(frame)
        else:
            # If we failed to receive a frame, wait a bit before trying again
            time.sleep(0.01)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Receiver")
    parser.add_argument("--ip", default="0.0.0.0", help="IP address to listen on")
    parser.add_argument("--port", type=int, default=9999, help="Port to listen on")
    parser.add_argument("--display", action="store_true", help="Display video (requires GUI)")
    parser.add_argument("--buffer", type=int, default=60, help="Frame buffer size")
    parser.add_argument("--fps", type=float, default=0, help="Override playback FPS (0=use sender's FPS)")
    
    args = parser.parse_args()
    
    # Set variables from arguments
    server_ip = args.ip
    server_port = args.port
    display_video = args.display
    frame_buffer = deque(maxlen=args.buffer)
    override_fps = args.fps
    playback_fps = override_fps  # Will be updated with video_info if not overridden
    
    # Create TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind socket to address and port
        server_socket.bind((server_ip, server_port))
        server_socket.listen(5)
        print(f"Listening on {server_ip}:{server_port}...")
        
        # Accept connection from sender
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        
        # Receive video information
        video_info = receive_video_info(client_socket)
        if not video_info:
            print("Failed to receive video info")
            client_socket.close()
            server_socket.close()
            exit()
        
        print(f"Received video info: {video_info}")
        
        # Set playback FPS from video_info if not overridden
        if override_fps <= 0 and video_info and 'fps' in video_info:
            playback_fps = video_info['fps']
        elif override_fps > 0:
            playback_fps = override_fps
        else:
            playback_fps = 30.0  # Default if no FPS info available
            
        print(f"Using playback FPS: {playback_fps}")
        
        # Start frame buffering thread
        buffer_thread = threading.Thread(target=buffer_frames, args=(client_socket,))
        buffer_thread.daemon = True
        buffer_thread.start()
        
        # Wait for buffer to fill initially
        print("Buffering frames...")
        buffer_fill_start = time.time()
        while len(frame_buffer) < min(30, frame_buffer.maxlen):
            # Don't wait more than 5 seconds for buffer to fill
            if time.time() - buffer_fill_start > 5.0:
                break
            time.sleep(0.1)
            print(f"Buffer: {len(frame_buffer)}/{frame_buffer.maxlen}", end="\r")
        print("\nBuffer filled, starting playback")
        
        # Start displaying frames
        last_stats_time = time.time()
        last_frame_time = time.time()
        
        # Calculate target frame time based on playback FPS
        target_frame_time = 1.0 / playback_fps
        
        while running:
            # Calculate time since last frame
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Only display a new frame if enough time has passed (control playback speed)
            if elapsed >= target_frame_time:
                # Get frame from buffer
                frame = None
                with buffer_lock:
                    if frame_buffer:
                        frame = frame_buffer.popleft()
                
                if frame is not None:
                    # Display frame if requested
                    if display_video:
                        cv2.imshow('Received Video', frame)
                        frames_displayed += 1
                        
                        # Press 'q' to quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            running = False
                            break
                    
                    # Update last frame time for consistent playback speed
                    last_frame_time = current_time
                else:
                    # No frames in buffer, wait a bit
                    time.sleep(0.01)
            else:
                # Not time for next frame yet, short sleep
                time.sleep(0.001)
            
            # Print statistics every second
            if current_time - last_stats_time >= 1.0:
                print_stats()
                last_stats_time = current_time
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up resources
        running = False
        if buffer_thread and buffer_thread.is_alive():
            buffer_thread.join(timeout=1.0)
        
        if 'client_socket' in locals():
            client_socket.close()
        
        server_socket.close()
        
        if display_video:
            cv2.destroyAllWindows()
        
        print("Socket closed")