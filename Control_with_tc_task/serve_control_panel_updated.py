#!/usr/bin/env python3
"""
Simple HTTP server to serve the updated control panel
"""
import http.server
import socketserver
import os
import webbrowser
import socket
import sys

# Try these ports in order
PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8080, 8888, 9000]

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def do_GET(self):
        # Redirect requests for /control.html to /control_updated.html
        if self.path == '/control.html':
            self.path = '/control_updated.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def main():
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Try each port in the list
    for port in PORTS:
        try:
            # Create the server
            httpd = socketserver.TCPServer(("", port), MyHandler)
            print(f"Serving updated control panel at http://localhost:{port}/control_updated.html")
            
            # Open the control panel in the default browser
            webbrowser.open(f"http://localhost:{port}/control_updated.html")
            
            # Serve until interrupted
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
                httpd.server_close()
                return
                
            # If we get here, the server was started successfully
            break
            
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"Port {port} is already in use, trying next port...")
            else:
                print(f"Error starting server on port {port}: {e}")
                sys.exit(1)
    else:
        # If we get here, all ports failed
        print("All ports are in use. Please close some applications and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()