import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a socket that connects to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect but helps determine the IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error determining local IP: {e}")
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Your local IP address is: {local_ip}")
    print(f"Use this IP address for the signaling server when connecting from other machines.")
    print(f"Example for receiver: python webrtc_receiver.py --signaling-server ws://{local_ip}:8765")