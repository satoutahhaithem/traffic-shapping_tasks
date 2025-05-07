#!/usr/bin/env python3
import argparse
import requests
import sys
import time

def test_connection(host, port, path=""):
    """Test if we can connect to the sender server."""
    url = f"http://{host}:{port}{path}"
    print(f"Testing connection to {url}...")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Successfully connected to {url}")
            print(f"Response status: {response.status_code}")
            print(f"Response size: {len(response.text)} bytes")
            return True
        else:
            print(f"❌ Connected but received status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Failed to connect to {url} - Connection refused")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Connection to {url} timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test connection to WebRTC sender")
    parser.add_argument("--host", default="localhost", help="Sender host (default: localhost)")
    parser.add_argument("--port", type=int, default=8090, help="Sender port (default: 8090)")
    args = parser.parse_args()
    
    print("WebRTC Connection Tester")
    print("========================")
    
    # Test basic connection
    if not test_connection(args.host, args.port):
        print("\nTroubleshooting tips:")
        print("1. Make sure the sender is running")
        print("2. Check if the host and port are correct")
        print("3. Check if there's a firewall blocking the connection")
        print("4. If using 'localhost', try using '127.0.0.1' instead")
        sys.exit(1)
    
    # Test WebRTC connection
    print("\nBasic connection successful. Testing WebRTC connection...")
    print("This will attempt to fetch the JavaScript file needed for WebRTC.")
    
    if test_connection(args.host, args.port, "/webrtc.js"):
        print("\n✅ WebRTC connection test successful!")
        print("The receiver should be able to connect to the sender.")
    else:
        print("\n❌ WebRTC connection test failed.")
        print("The sender might not be properly configured for WebRTC.")
    
    print("\nIf you're still having issues, try the following:")
    print("1. Make sure both sender and receiver have all dependencies installed")
    print("2. Check the console output for any errors")
    print("3. Try restarting both the sender and receiver")
    print("4. If on different machines, make sure they can reach each other")

if __name__ == "__main__":
    main()