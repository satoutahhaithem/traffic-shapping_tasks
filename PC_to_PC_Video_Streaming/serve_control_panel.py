#!/usr/bin/env python3
"""
Simple HTTP server to serve the control panel for PC-to-PC video streaming.
"""
import http.server
import socketserver
import os
import webbrowser
import threading
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('control_panel_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'port': 8000,
    'sender_ip': '127.0.0.1',
    'sender_port': 5000,
    'receiver_ip': '127.0.0.1',
    'receiver_port': 5001,
    'auto_open_browser': True,
}

# Current configuration (will be updated by command line args)
config = DEFAULT_CONFIG.copy()

class ControlPanelHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for serving the control panel."""
    
    def end_headers(self):
        """Add CORS headers to allow cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()

def parse_arguments():
    """Parse command line arguments and update configuration."""
    global config
    
    parser = argparse.ArgumentParser(description='Control Panel Server for PC-to-PC Video Streaming')
    
    parser.add_argument('--port', type=int, help='Port to listen on')
    parser.add_argument('--sender-ip', type=str, help='IP address of the sender')
    parser.add_argument('--sender-port', type=int, help='Port of the sender')
    parser.add_argument('--receiver-ip', type=str, help='IP address of the receiver')
    parser.add_argument('--receiver-port', type=int, help='Port of the receiver')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.port:
        config['port'] = args.port
    
    if args.sender_ip:
        config['sender_ip'] = args.sender_ip
    
    if args.sender_port:
        config['sender_port'] = args.sender_port
    
    if args.receiver_ip:
        config['receiver_ip'] = args.receiver_ip
    
    if args.receiver_port:
        config['receiver_port'] = args.receiver_port
    
    if args.no_browser:
        config['auto_open_browser'] = False
    
    return config

def open_browser():
    """Open the control panel in the default browser."""
    url = f"http://localhost:{config['port']}/control_panel.html"
    logger.info(f"Opening control panel in browser: {url}")
    webbrowser.open(url)

def main():
    """Main function to start the server."""
    # Parse command line arguments
    config = parse_arguments()
    
    # Log the configuration
    logger.info(f"Configuration: {config}")
    
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create the server
    handler = ControlPanelHandler
    httpd = socketserver.TCPServer(("", config['port']), handler)
    
    # Open the control panel in the browser if auto_open_browser is enabled
    if config['auto_open_browser']:
        # Wait a bit for the server to start
        threading.Thread(target=lambda: (time.sleep(1), open_browser()), daemon=True).start()
    
    # Start the server
    logger.info(f"Starting control panel server on port {config['port']}")
    logger.info(f"Control panel URL: http://localhost:{config['port']}/control_panel.html")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        httpd.server_close()
        logger.info("Server closed")

if __name__ == '__main__':
    main()