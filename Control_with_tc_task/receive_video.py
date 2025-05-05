import cv2
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import base64
import numpy as np
import time
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize a variable to hold the current frame
current_frame = None

# Initialize counters for statistics
frames_received = 0
last_frame_time = time.time()
frame_sizes = []

# Function to generate MJPEG stream from received frames
def generate():
    global current_frame
    while True:
        if current_frame is not None:
            # Log the frame size for debugging
            print(f"Sending frame with shape: {current_frame.shape}")

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            jpeg_bytes = jpeg.tobytes()

            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')

@app.route('/')
def home():
    return "Welcome to the Video Receiver! Visit /rx_video_feed to view the stream."

@app.route('/receive_video', methods=['POST'])
def receive_video():
    global current_frame, frames_received, last_frame_time, frame_sizes

    # Log reception time and request details
    current_time = time.time()
    time_since_last = current_time - last_frame_time
    last_frame_time = current_time
    
    # Get client IP
    client_ip = request.remote_addr
    
    # Get the base64-encoded frame from the POST request
    data = request.json
    frame_data = data['frame']
    frame_size = len(frame_data)
    frame_sizes.append(frame_size)
    
    # Keep only the last 30 frame sizes for statistics
    if len(frame_sizes) > 30:
        frame_sizes.pop(0)
    
    # Calculate average frame size and rate
    avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    frames_received += 1
    
    # Log detailed information every 10 frames
    if frames_received % 10 == 0:
        print(f"[INFO] Received frame #{frames_received} from {client_ip}")
        print(f"[INFO] Time since last frame: {time_since_last:.4f} seconds ({1/time_since_last:.2f} fps)")
        print(f"[INFO] Frame size: {frame_size} bytes, Avg size: {avg_frame_size:.2f} bytes")
        
        # Calculate estimated bandwidth
        estimated_bandwidth = (avg_frame_size * 8) / time_since_last  # bits per second
        if estimated_bandwidth > 1000000:
            print(f"[INFO] Estimated bandwidth: {estimated_bandwidth/1000000:.2f} Mbps")
        elif estimated_bandwidth > 1000:
            print(f"[INFO] Estimated bandwidth: {estimated_bandwidth/1000:.2f} Kbps")
        else:
            print(f"[INFO] Estimated bandwidth: {estimated_bandwidth:.2f} bps")

    # Decode the frame from base64 format
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Set the current frame to be used in the MJPEG stream
            current_frame = frame
            return jsonify({'status': 'success'}), 200
        else:
            print(f"[ERROR] Failed to decode frame: frame is None")
            return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400
    except Exception as e:
        print(f"[ERROR] Exception decoding frame: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Exception: {str(e)}'}), 400

@app.route('/rx_video_feed')
def video_feed():
    # Return the MJPEG stream to the browser
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Return statistics about received frames"""
    global frames_received, frame_sizes, last_frame_time
    
    current_time = time.time()
    time_since_last = current_time - last_frame_time
    avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    
    stats = {
        'frames_received': frames_received,
        'time_since_last_frame': f"{time_since_last:.4f} seconds",
        'estimated_fps': f"{1/time_since_last:.2f}" if time_since_last > 0 else "N/A",
        'avg_frame_size': f"{avg_frame_size:.2f} bytes",
        'estimated_bandwidth': f"{(avg_frame_size * 8) / time_since_last:.2f} bps" if time_since_last > 0 else "N/A"
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    print(f"[INFO] Starting receive_video.py on port 5001...")
    print(f"[INFO] Listening on all interfaces (0.0.0.0)")
    print(f"[INFO] Access statistics at http://localhost:5001/stats")
    
    # Run the Flask app on all interfaces and port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)