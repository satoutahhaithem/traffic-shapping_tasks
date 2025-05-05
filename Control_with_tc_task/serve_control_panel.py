#!/usr/bin/env python3
"""
Simple HTTP server to serve the control panel
"""
import http.server
import socketserver
import os
import webbrowser

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def main():
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create the server
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving control panel at http://localhost:{PORT}/control.html")
        
        # Open the control panel in the default browser
        webbrowser.open(f"http://localhost:{PORT}/control.html")
        
        # Serve until interrupted
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()