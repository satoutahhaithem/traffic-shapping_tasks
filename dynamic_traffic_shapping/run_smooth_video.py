#!/usr/bin/env python3
import os
import time
import subprocess
import webbrowser
import signal
import sys
import threading
import socket
import ipaddress

def run_command(command, wait=False):
    """Run a command in a new process"""
    print(f"Running: {command}")
    if wait:
        return subprocess.run(command, shell=True)
    else:
        return subprocess.Popen(command, shell=True)

def open_browser(url, delay=2):
    """Open a URL in the default browser after a delay"""
    time.sleep(delay)
    print(f"Opening browser at: {url}")
    webbrowser.open(url)

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully terminate all processes"""
    print("\nShutting down all processes...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a socket to determine the outgoing IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's DNS server
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return "127.0.0.1"  # Fallback to localhost

def is_valid_ip(ip):
    """Check if the provided IP address is valid"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def update_streamer_ip(receiver_ip):
    """Update the receiver_ip in video_streamer.py"""
    try:
        with open("video_streamer.py", "r") as f:
            content = f.read()
        
        # Replace the receiver_ip line
        import re
        new_content = re.sub(r'receiver_ip = "[^"]*"', f'receiver_ip = "{receiver_ip}"', content)
        
        with open("video_streamer.py", "w") as f:
            f.write(new_content)
        
        print(f"Updated video_streamer.py to use receiver IP: {receiver_ip}")
        return True
    except Exception as e:
        print(f"Error updating streamer IP: {e}")
        return False

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Store all processes to terminate them later
processes = []

def main():
    print("Starting Two-PC Smooth Video Streaming System")
    print("===========================================")
    
    # Determine if this is the sender or receiver PC
    local_ip = get_local_ip()
    print(f"Your local IP address is: {local_ip}")
    
    mode = input("\nIs this the SENDER or RECEIVER PC? (s/r): ").lower()
    
    if mode == 's':
        # This is the sender PC
        print("\nRunning in SENDER mode")
        
        # Get the receiver's IP address
        receiver_ip = input("\nEnter the IP address of the RECEIVER PC: ")
        while not is_valid_ip(receiver_ip):
            print("Invalid IP address. Please try again.")
            receiver_ip = input("Enter the IP address of the RECEIVER PC: ")
        
        # Update the video_streamer.py file with the receiver's IP
        update_streamer_ip(receiver_ip)
        
        # Step 1: Apply optimal network conditions
        print("\nStep 1: Applying optimal network conditions for smooth playback...")
        # Check if we have sudo privileges
        if os.geteuid() == 0:
            # We're running as root, apply network conditions directly
            tc_process = run_command("bash dynamic_tc_control.sh", wait=True)
            # Select option 6 (Apply ULTRA-SMOOTH streaming conditions)
            tc_process = run_command("echo '6' | bash dynamic_tc_control.sh", wait=True)
        else:
            # We're not running as root, prompt user to run the command separately
            print("To apply ultra-smooth network conditions, please run this command in another terminal:")
            print("sudo bash dynamic_tc_control.sh")
            print("Then select option 6 to apply ULTRA-SMOOTH streaming conditions.")
            input("Press Enter to continue after applying network conditions...")
        
        # Step 2: Start the video streamer
        print("\nStep 2: Starting video streamer...")
        print(f"Streaming video to {receiver_ip}:8081")
        streamer_process = run_command("python3 video_streamer.py")
        processes.append(streamer_process)
        
        # Wait for streamer to initialize
        time.sleep(2)
        
        # Step 3: Open browser windows to view the streams
        print("\nStep 3: Opening browser windows...")
        
        # Open streamer quality controls in a new thread
        threading.Thread(target=open_browser, args=(f"http://localhost:5000/quality_controls",)).start()
        
        # Open streamer local view in a new thread
        threading.Thread(target=open_browser, args=(f"http://localhost:5000/tx_video_feed", 3)).start()
        
        print("\nSender is now running!")
        print(f"- Video streamer is running on port 5000")
        print(f"- Streaming to receiver at {receiver_ip}:8081")
        print("- Browser windows should open automatically")
        print(f"\nOn the RECEIVER PC, you can view the stream at: http://{receiver_ip}:8081/rx_video_feed")
        print("\nTo stop the system, press Ctrl+C")
        
    elif mode == 'r':
        # This is the receiver PC
        print("\nRunning in RECEIVER mode")
        
        # Step 1: Start the video receiver
        print("\nStep 1: Starting video receiver...")
        receiver_process = run_command("python3 receive_video.py")
        processes.append(receiver_process)
        
        # Wait for receiver to initialize
        time.sleep(2)
        
        # Step 2: Open browser windows to view the streams
        print("\nStep 2: Opening browser windows...")
        
        # Open receiver stream in a new thread to avoid blocking
        threading.Thread(target=open_browser, args=(f"http://localhost:8081/rx_video_feed",)).start()
        
        print("\nReceiver is now running!")
        print(f"- Video receiver is running on port 8081")
        print(f"- Your IP address is {local_ip}")
        print("- Browser window should open automatically")
        print(f"\nMake sure the SENDER PC is configured to stream to this IP address: {local_ip}")
        print("\nTo stop the system, press Ctrl+C")
        
    else:
        print("Invalid mode. Please run the script again and enter 's' for sender or 'r' for receiver.")
        return
    
    # Keep the script running until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()