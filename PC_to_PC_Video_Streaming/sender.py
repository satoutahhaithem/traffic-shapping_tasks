#!/usr/bin/env python3
"""
Sender script for PC-to-PC video streaming.
This script captures video from a file or camera and streams it to a receiver.
"""
import cv2
import base64
import time
import requests
import socket
import argparse
import json
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sender.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default configuration (can be overridden via command line arguments)
DEFAULT_CONFIG = {
    'video_source': 0,  # 0 for webcam, or path to video file
    'receiver_ip': '127.0.0.1',  # Default to localhost, should be changed to receiver's IP
    'receiver_port': 5001,
    'sender_port': 5000,
    'frame_rate': 30,  # Target frame rate (fps)
    'quality': 90,  # JPEG quality (0-100)
    'resolution': (640, 480),  # Output resolution (width, height)
    'bandwidth_limit': 0,  # 0 means no limit, otherwise in bytes/second
    'artificial_delay': 0,  # Additional delay in seconds
    'packet_loss': 0,  # Simulated packet loss (0-100%)
}

# Current configuration (will be updated by command line args and API)
config = DEFAULT_CONFIG.copy()

# Shared state
is_streaming = False
cap = None
stream_thread = None

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google DNS to determine local IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Error getting local IP: {e}")
        return "127.0.0.1"  # Fallback to localhost

def open_video_source():
    """Open the video source (file or camera)."""
    global cap
    
    source = config['video_source']
    logger.info(f"Opening video source: {source}")
    
    if cap is not None:
        cap.release()  # Release previous capture if any
    
    # If source is an integer, it's a camera index
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
    else:
        # Otherwise, it's a file path
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return False
    
    # Set resolution if using a camera
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        width, height = config['resolution']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    logger.info(f"Video source opened successfully: {source}")
    return True

def send_frame(frame):
    """Send a frame to the receiver."""
    # Apply quality settings
    quality = config['quality']
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Resize frame if needed
    width, height = config['resolution']
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    
    # Encode the frame
    ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
    if not ret:
        logger.error("Failed to encode frame")
        return False
    
    # Apply artificial delay if configured
    delay = config['artificial_delay']
    if delay > 0:
        time.sleep(delay)
    
    # Simulate packet loss if configured
    loss = config['packet_loss']
    if loss > 0:
        import random
        if random.random() < loss / 100:
            logger.debug("Simulating packet loss - frame dropped")
            return True  # Pretend we sent it
    
    # Prepare the frame data
    encoded_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    
    # Apply bandwidth limit if configured
    bandwidth_limit = config['bandwidth_limit']
    if bandwidth_limit > 0:
        # Calculate how long this frame should take to send
        frame_size = len(encoded_frame)
        delay_time = frame_size / bandwidth_limit
        time.sleep(delay_time)
    
    # Send the frame to the receiver
    receiver_url = f"http://{config['receiver_ip']}:{config['receiver_port']}/receive_frame"
    try:
        response = requests.post(
            receiver_url, 
            json={
                'frame': encoded_frame,
                'timestamp': time.time(),
                'frame_id': int(time.time() * 1000),  # Unique ID for the frame
                'sender_ip': get_local_ip()
            },
            timeout=1.0  # Short timeout to prevent blocking
        )
        
        if response.status_code != 200:
            logger.warning(f"Receiver returned non-200 status: {response.status_code}")
            return False
        
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending frame to receiver: {e}")
        return False

def streaming_thread():
    """Background thread for capturing and sending frames."""
    global is_streaming, cap
    
    logger.info("Streaming thread started")
    
    if not open_video_source():
        logger.error("Failed to open video source, streaming thread exiting")
        is_streaming = False
        return
    
    frame_time = 1.0 / config['frame_rate']
    frames_sent = 0
    start_time = time.time()
    
    while is_streaming:
        loop_start = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream or error capturing frame")
            # For video files, loop back to the beginning
            if not (isinstance(config['video_source'], int) or 
                   (isinstance(config['video_source'], str) and config['video_source'].isdigit())):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Send the frame
        success = send_frame(frame)
        if success:
            frames_sent += 1
        
        # Calculate and log FPS every 30 frames
        if frames_sent % 30 == 0:
            elapsed = time.time() - start_time
            fps = frames_sent / elapsed if elapsed > 0 else 0
            logger.info(f"Streaming at {fps:.2f} fps")
        
        # Sleep to maintain target frame rate
        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Clean up
    if cap is not None:
        cap.release()
    
    logger.info("Streaming thread stopped")

def start_streaming():
    """Start the streaming thread if not already running."""
    global is_streaming, stream_thread
    
    if is_streaming:
        logger.info("Streaming is already active")
        return True
    
    is_streaming = True
    stream_thread = threading.Thread(target=streaming_thread)
    stream_thread.daemon = True
    stream_thread.start()
    
    logger.info("Streaming started")
    return True

def stop_streaming():
    """Stop the streaming thread if running."""
    global is_streaming
    
    if not is_streaming:
        logger.info("Streaming is not active")
        return True
    
    is_streaming = False
    logger.info("Streaming stopping (may take a moment to complete)")
    return True

# API Routes

@app.route('/')
def home():
    """Home page with basic information."""
    return f"""
    <html>
    <head><title>PC-to-PC Video Sender</title></head>
    <body>
        <h1>PC-to-PC Video Sender</h1>
        <p>Status: {'Streaming' if is_streaming else 'Not Streaming'}</p>
        <p>Sending to: {config['receiver_ip']}:{config['receiver_port']}</p>
        <p>Video Source: {config['video_source']}</p>
        <p>Frame Rate: {config['frame_rate']} fps</p>
        <p>Resolution: {config['resolution'][0]}x{config['resolution'][1]}</p>
        <p>Quality: {config['quality']}%</p>
        <p>Artificial Delay: {config['artificial_delay']} seconds</p>
        <p>Packet Loss Simulation: {config['packet_loss']}%</p>
        <p>Bandwidth Limit: {config['bandwidth_limit']} bytes/second</p>
        <h2>API Endpoints:</h2>
        <ul>
            <li><a href="/start">Start Streaming</a></li>
            <li><a href="/stop">Stop Streaming</a></li>
            <li><a href="/status">Get Status</a></li>
            <li>POST to /config to update configuration</li>
        </ul>
    </body>
    </html>
    """

@app.route('/start')
def api_start():
    """Start streaming API endpoint."""
    if start_streaming():
        return jsonify({'status': 'success', 'message': 'Streaming started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start streaming'}), 500

@app.route('/stop')
def api_stop():
    """Stop streaming API endpoint."""
    if stop_streaming():
        return jsonify({'status': 'success', 'message': 'Streaming stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop streaming'}), 500

@app.route('/status')
def api_status():
    """Get current status API endpoint."""
    return jsonify({
        'streaming': is_streaming,
        'config': config,
        'sender_ip': get_local_ip()
    })

@app.route('/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration API endpoint."""
    global config
    
    if request.method == 'GET':
        return jsonify(config)
    
    # Handle POST to update config
    try:
        new_config = request.json
        if not new_config:
            return jsonify({'status': 'error', 'message': 'No configuration provided'}), 400
        
        # Update config with new values
        for key, value in new_config.items():
            if key in config:
                config[key] = value
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Ignoring unknown config parameter: {key}")
        
        # If video source changed, reopen it
        if 'video_source' in new_config and is_streaming:
            if not open_video_source():
                return jsonify({'status': 'error', 'message': 'Failed to open new video source'}), 500
        
        return jsonify({'status': 'success', 'message': 'Configuration updated', 'config': config})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def parse_arguments():
    """Parse command line arguments and update configuration."""
    global config
    
    parser = argparse.ArgumentParser(description='PC-to-PC Video Sender')
    
    parser.add_argument('--video-source', type=str, help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--receiver-ip', type=str, help='IP address of the receiver')
    parser.add_argument('--receiver-port', type=int, help='Port of the receiver')
    parser.add_argument('--sender-port', type=int, help='Port for this sender to listen on')
    parser.add_argument('--frame-rate', type=int, help='Target frame rate (fps)')
    parser.add_argument('--quality', type=int, help='JPEG quality (0-100)')
    parser.add_argument('--width', type=int, help='Output width')
    parser.add_argument('--height', type=int, help='Output height')
    parser.add_argument('--bandwidth-limit', type=int, help='Bandwidth limit in bytes/second')
    parser.add_argument('--delay', type=float, help='Artificial delay in seconds')
    parser.add_argument('--packet-loss', type=float, help='Simulated packet loss percentage')
    parser.add_argument('--auto-start', action='store_true', help='Start streaming automatically')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.video_source is not None:
        if args.video_source.isdigit():
            config['video_source'] = int(args.video_source)
        else:
            config['video_source'] = args.video_source
    
    if args.receiver_ip:
        config['receiver_ip'] = args.receiver_ip
    
    if args.receiver_port:
        config['receiver_port'] = args.receiver_port
    
    if args.sender_port:
        config['sender_port'] = args.sender_port
    
    if args.frame_rate:
        config['frame_rate'] = args.frame_rate
    
    if args.quality:
        config['quality'] = max(1, min(100, args.quality))
    
    if args.width and args.height:
        config['resolution'] = (args.width, args.height)
    
    if args.bandwidth_limit is not None:
        config['bandwidth_limit'] = args.bandwidth_limit
    
    if args.delay is not None:
        config['artificial_delay'] = args.delay
    
    if args.packet_loss is not None:
        config['packet_loss'] = max(0, min(100, args.packet_loss))
    
    return args.auto_start

if __name__ == '__main__':
    # Parse command line arguments
    auto_start = parse_arguments()
    
    # Log the configuration
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Auto-start streaming if requested
    if auto_start:
        logger.info("Auto-starting streaming")
        start_streaming()
    
    # Start the Flask app
    logger.info(f"Starting sender on port {config['sender_port']}")
    app.run(host='0.0.0.0', port=config['sender_port'], debug=False, threaded=True)