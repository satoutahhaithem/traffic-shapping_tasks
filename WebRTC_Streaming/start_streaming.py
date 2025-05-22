import subprocess
import sys
import time
import os

def main():
    """
    Start the WebRTC streaming components in the correct order:
    1. Signaling server
    2. Receiver
    3. Sender
    """
    print("Starting WebRTC Streaming System...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start the signaling server
    print("\n1. Starting signaling server...")
    signaling_process = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, "signaling_server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the signaling server to start
    time.sleep(2)
    if signaling_process.poll() is not None:
        print("Error: Signaling server failed to start")
        out, err = signaling_process.communicate()
        print(f"Output: {out}")
        print(f"Error: {err}")
        return
    
    print("   Signaling server started successfully")
    
    # Start the receiver
    print("\n2. Starting receiver...")
    receiver_process = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, "webrtc_receiver.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the receiver to start
    time.sleep(2)
    if receiver_process.poll() is not None:
        print("Error: Receiver failed to start")
        out, err = receiver_process.communicate()
        print(f"Output: {out}")
        print(f"Error: {err}")
        signaling_process.terminate()
        return
    
    print("   Receiver started successfully")
    
    # Start the sender
    print("\n3. Starting sender...")
    sender_process = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, "webrtc_sender.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the sender to start
    time.sleep(2)
    if sender_process.poll() is not None:
        print("Error: Sender failed to start")
        out, err = sender_process.communicate()
        print(f"Output: {out}")
        print(f"Error: {err}")
        receiver_process.terminate()
        signaling_process.terminate()
        return
    
    print("   Sender started successfully")
    
    print("\nAll components started successfully!")
    print("The video should now be streaming from the sender to the receiver.")
    print("Press Ctrl+C to stop all components.")
    
    try:
        # Wait for user to press Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all components...")
    finally:
        # Stop all processes
        sender_process.terminate()
        receiver_process.terminate()
        signaling_process.terminate()
        
        print("All components stopped.")

if __name__ == "__main__":
    main()