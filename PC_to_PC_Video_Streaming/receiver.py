#!/usr/bin/env python3
"""
Receiver script for PC-to-PC video streaming.
This script receives video frames from a sender and displays them.
"""
import cv2
import base64
import time
import numpy as np
import argparse
import json
import threading
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('receiver.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default configuration (can be overridden via command line arguments)
DEFAULT_CONFIG = {
    'port': 5001,
    'buffer_size': 5,  # Number of frames to buffer
    'display_scale': 1.0,  # Scale factor for display
    'save_video': False,  # Whether to save received video
    'output_file': 'received_video.avi',  # Output file for saved video
    'stats_window': 30,  # Number of frames to calculate statistics over
}

# Current configuration (will be updated by command line args and API)
config = DEFAULT_CONFIG.copy()

# Shared state
current_frame = None
frame_buffer = deque(maxlen=10)  # Adjustable buffer
video_writer = None
is_displaying = False
display_thread = None

# Statistics
stats = {
    'frames_received': 0,
    'frames_displayed': 0,
    'start_time': time.time(),
    'last_frame_time': time.time(),
    'frame_times': deque(maxlen=100),  # Store last 100 frame arrival times
    'frame_sizes': deque(maxlen=100),  # Store last 100 frame sizes
    'frame_delays': deque(maxlen=100),  # Store last 100 frame delays (time from sender to receiver)
    'sender_ips': set(),  # Track unique sender IPs
}

def process_received_frame(frame_data, timestamp=None, frame_id=None, sender_ip=None):
    """Process a received frame and add it to the buffer."""
    global current_frame, frame_buffer, stats, video_writer
    
    try:
        # Decode the frame from base64
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode frame")
            return False
        
        # Update statistics
        current_time = time.time()
        stats['frames_received'] += 1
        stats['frame_times'].append(current_time)
        stats['frame_sizes'].append(len(frame_data))
        stats['last_frame_time'] = current_time
        
        if sender_ip:
            stats['sender_ips'].add(sender_ip)
        
        if timestamp:
            # Calculate delay from sender to receiver
            delay = current_time - timestamp
            stats['frame_delays'].append(delay)
        
        # Add frame to buffer
        frame_info = {
            'frame': frame,
            'timestamp': timestamp or current_time,
            'frame_id': frame_id or stats['frames_received'],
            'receive_time': current_time,
            'sender_ip': sender_ip
        }
        frame_buffer.append(frame_info)
        
        # Update current frame for direct access
        current_frame = frame
        
        # Save to video file if enabled
        if config['save_video'] and video_writer is not None:
            video_writer.write(frame)
        
        return True
    except Exception as e:
        logger.error(f"Error processing received frame: {e}")
        return False

def display_thread_function():
    """Background thread for displaying frames from the buffer."""
    global frame_buffer, stats, is_displaying
    
    logger.info("Display thread started")
    
    # Create a named window
    window_name = "PC-to-PC Video Receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while is_displaying:
        if frame_buffer:
            # Get the next frame from the buffer
            frame_info = frame_buffer.popleft()
            frame = frame_info['frame']
            
            # Apply display scaling if needed
            scale = config['display_scale']
            if scale != 1.0:
                width = int(frame.shape[1] * scale)
                height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (width, height))
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Update statistics
            stats['frames_displayed'] += 1
            
            # Add overlay with statistics
            if stats['frames_displayed'] % 30 == 0:
                # Calculate FPS
                elapsed = time.time() - stats['start_time']
                fps = stats['frames_displayed'] / elapsed if elapsed > 0 else 0
                
                # Calculate average delay
                avg_delay = sum(stats['frame_delays']) / len(stats['frame_delays']) if stats['frame_delays'] else 0
                
                logger.info(f"Displaying at {fps:.2f} fps, avg delay: {avg_delay*1000:.1f} ms")
        
        # Check for key press (q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("User pressed 'q', stopping display")
            is_displaying = False
            break
        
        # Sleep a bit to prevent high CPU usage when buffer is empty
        if not frame_buffer:
            time.sleep(0.01)
    
    # Clean up
    cv2.destroyAllWindows()
    logger.info("Display thread stopped")

def start_display():
    """Start the display thread if not already running."""
    global is_displaying, display_thread
    
    if is_displaying:
        logger.info("Display is already active")
        return True
    
    is_displaying = True
    display_thread = threading.Thread(target=display_thread_function)
    display_thread.daemon = True
    display_thread.start()
    
    logger.info("Display started")
    return True

def stop_display():
    """Stop the display thread if running."""
    global is_displaying
    
    if not is_displaying:
        logger.info("Display is not active")
        return True
    
    is_displaying = False
    logger.info("Display stopping (may take a moment to complete)")
    return True

def setup_video_writer():
    """Set up the video writer for saving received video."""
    global video_writer, current_frame
    
    if not config['save_video']:
        return False
    
    # Wait for the first frame to determine resolution
    timeout = 10  # seconds
    start_time = time.time()
    while current_frame is None:
        if time.time() - start_time > timeout:
            logger.error("Timeout waiting for first frame to setup video writer")
            return False
        time.sleep(0.1)
    
    # Get frame dimensions
    height, width = current_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        config['output_file'],
        fourcc,
        30.0,  # Assume 30 fps
        (width, height)
    )
    
    if not video_writer.isOpened():
        logger.error(f"Failed to open video writer for {config['output_file']}")
        return False
    
    logger.info(f"Video writer set up for {config['output_file']}")
    return True

# API Routes

@app.route('/')
def home():
    """Home page with basic information."""
    # Calculate statistics
    elapsed = time.time() - stats['start_time']
    receive_fps = stats['frames_received'] / elapsed if elapsed > 0 else 0
    display_fps = stats['frames_displayed'] / elapsed if elapsed > 0 else 0
    
    avg_delay = sum(stats['frame_delays']) / len(stats['frame_delays']) if stats['frame_delays'] else 0
    avg_size = sum(stats['frame_sizes']) / len(stats['frame_sizes']) if stats['frame_sizes'] else 0
    
    buffer_status = f"{len(frame_buffer)}/{frame_buffer.maxlen}" if frame_buffer else "0/0"
    
    return f"""
    <html>
    <head>
        <title>PC-to-PC Video Receiver</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .video {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>PC-to-PC Video Receiver</h1>
        
        <div class="stats">
            <h2>Statistics</h2>
            <p>Running for: {elapsed:.1f} seconds</p>
            <p>Frames Received: {stats['frames_received']}</p>
            <p>Frames Displayed: {stats['frames_displayed']}</p>
            <p>Receive FPS: {receive_fps:.2f}</p>
            <p>Display FPS: {display_fps:.2f}</p>
            <p>Average Delay: {avg_delay*1000:.1f} ms</p>
            <p>Average Frame Size: {avg_size:.0f} bytes</p>
            <p>Buffer Status: {buffer_status}</p>
            <p>Sender IPs: {', '.join(stats['sender_ips'])}</p>
        </div>
        
        <div class="video">
            <h2>Video Feed</h2>
            <img src="/video_feed" width="640" height="480" />
        </div>
        
        <h2>API Endpoints:</h2>
        <ul>
            <li><a href="/start_display">Start Display</a></li>
            <li><a href="/stop_display">Stop Display</a></li>
            <li><a href="/stats">Get Statistics</a></li>
            <li><a href="/video_feed">Video Feed</a></li>
            <li>POST to /config to update configuration</li>
        </ul>
    </body>
    </html>
    """

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """Endpoint for receiving frames from the sender."""
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({'status': 'error', 'message': 'No frame data provided'}), 400
        
        frame_data = data['frame']
        timestamp = data.get('timestamp')
        frame_id = data.get('frame_id')
        sender_ip = data.get('sender_ip') or request.remote_addr
        
        success = process_received_frame(frame_data, timestamp, frame_id, sender_ip)
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process frame'}), 500
    except Exception as e:
        logger.error(f"Error in receive_frame: {e}")
        return jsonify({'status': 'error', 'message': f'Exception: {str(e)}'}), 500

def generate_video_feed():
    """Generate frames for the video feed endpoint."""
    while True:
        if current_frame is not None:
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            # If no frame is available, yield a blank frame
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_frame, "Waiting for video...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        # Sleep a bit to control frame rate
        time.sleep(0.033)  # ~30 fps

@app.route('/video_feed')
def video_feed():
    """Endpoint for streaming video feed to browsers."""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_display')
def api_start_display():
    """Start the display thread."""
    if start_display():
        return jsonify({'status': 'success', 'message': 'Display started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start display'}), 500

@app.route('/stop_display')
def api_stop_display():
    """Stop the display thread."""
    if stop_display():
        return jsonify({'status': 'success', 'message': 'Display stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop display'}), 500

@app.route('/stats')
def api_stats():
    """Get current statistics."""
    # Calculate derived statistics
    elapsed = time.time() - stats['start_time']
    receive_fps = stats['frames_received'] / elapsed if elapsed > 0 else 0
    display_fps = stats['frames_displayed'] / elapsed if elapsed > 0 else 0
    
    avg_delay = sum(stats['frame_delays']) / len(stats['frame_delays']) if stats['frame_delays'] else 0
    avg_size = sum(stats['frame_sizes']) / len(stats['frame_sizes']) if stats['frame_sizes'] else 0
    
    # Calculate jitter (variation in delay)
    jitter = 0
    if len(stats['frame_delays']) > 1:
        delays = list(stats['frame_delays'])
        differences = [abs(delays[i] - delays[i-1]) for i in range(1, len(delays))]
        jitter = sum(differences) / len(differences)
    
    return jsonify({
        'frames_received': stats['frames_received'],
        'frames_displayed': stats['frames_displayed'],
        'uptime': elapsed,
        'receive_fps': receive_fps,
        'display_fps': display_fps,
        'buffer_size': len(frame_buffer),
        'buffer_capacity': frame_buffer.maxlen,
        'avg_delay_ms': avg_delay * 1000,
        'avg_frame_size_bytes': avg_size,
        'jitter_ms': jitter * 1000,
        'sender_ips': list(stats['sender_ips'])
    })

@app.route('/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration."""
    global config, frame_buffer
    
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
                # Special handling for buffer_size
                if key == 'buffer_size' and value != config['buffer_size']:
                    # Create a new buffer with the new size
                    new_buffer = deque(maxlen=value)
                    # Copy over existing frames if any
                    while frame_buffer and len(new_buffer) < value:
                        new_buffer.append(frame_buffer.popleft())
                    frame_buffer = new_buffer
                
                config[key] = value
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Ignoring unknown config parameter: {key}")
        
        # Handle save_video changes
        if 'save_video' in new_config and new_config['save_video'] and video_writer is None:
            setup_video_writer()
        
        return jsonify({'status': 'success', 'message': 'Configuration updated', 'config': config})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def parse_arguments():
    """Parse command line arguments and update configuration."""
    global config
    
    parser = argparse.ArgumentParser(description='PC-to-PC Video Receiver')
    
    parser.add_argument('--port', type=int, help='Port to listen on')
    parser.add_argument('--buffer-size', type=int, help='Number of frames to buffer')
    parser.add_argument('--display-scale', type=float, help='Scale factor for display')
    parser.add_argument('--save-video', action='store_true', help='Save received video to file')
    parser.add_argument('--output-file', type=str, help='Output file for saved video')
    parser.add_argument('--auto-display', action='store_true', help='Start display automatically')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.port:
        config['port'] = args.port
    
    if args.buffer_size:
        config['buffer_size'] = args.buffer_size
        frame_buffer = deque(maxlen=config['buffer_size'])
    
    if args.display_scale:
        config['display_scale'] = args.display_scale
    
    if args.save_video:
        config['save_video'] = True
    
    if args.output_file:
        config['output_file'] = args.output_file
    
    return args.auto_display

if __name__ == '__main__':
    # Parse command line arguments
    auto_display = parse_arguments()
    
    # Log the configuration
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Auto-start display if requested
    if auto_display:
        logger.info("Auto-starting display")
        start_display()
    
    # Setup video writer if saving is enabled
    if config['save_video']:
        # Start in a separate thread to not block startup
        threading.Thread(target=setup_video_writer, daemon=True).start()
    
    # Start the Flask app
    logger.info(f"Starting receiver on port {config['port']}")
    app.run(host='0.0.0.0', port=config['port'], debug=False, threaded=True)