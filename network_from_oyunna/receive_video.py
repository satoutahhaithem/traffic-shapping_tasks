import cv2
from flask import Flask, request, jsonify, Response
import base64
import numpy as np
import time

app = Flask(__name__)

# Initialize variables
current_frame = None
frames_received = 0
last_frame_time = time.time()
fps_estimate = 0

# Function to generate MJPEG stream from received frames
def generate():
    global current_frame, frames_received
    print("MJPEG stream generator started - waiting for frames...")
    local_frame_count = 0
    last_log_time = time.time()
    
    while True:
        if current_frame is not None:
            local_frame_count += 1
            
            # Log only once per second to reduce console output
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                print(f"Streaming frame #{local_frame_count} with shape: {current_frame.shape}")
                last_log_time = current_time
            
            # Encode the frame as JPEG with quality parameter
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, jpeg = cv2.imencode('.jpg', current_frame, encode_params)
            if not ret:
                print("Failed to encode frame for streaming!")
                continue
            
            jpeg_bytes = jpeg.tobytes()
            
            # Only log size occasionally
            if local_frame_count % 30 == 0:
                print(f"Encoded JPEG size: {len(jpeg_bytes)} bytes")
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
        else:
            # Reduce waiting messages
            if time.time() - last_log_time >= 5.0:
                print("Waiting for frames from streamer...")
                last_log_time = time.time()
            
            time.sleep(0.1)  # Check more frequently but log less

@app.route('/status')
def status():
    return f"""
    <html>
    <head>
        <title>Video Receiver Status</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .status-box {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .good {{ background-color: #d4edda; }}
            .warning {{ background-color: #fff3cd; }}
            .error {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>Video Receiver Status</h1>
        
        <div class="status-box {'good' if current_frame is not None else 'error'}">
            <h2>Stream Status</h2>
            <p>Status: {"Receiving frames" if current_frame is not None else "No frames received"}</p>
            <p>Frames Received: {frames_received}</p>
            <p>Current FPS: {fps_estimate:.1f}</p>
            <p>Last Frame: {time.strftime('%H:%M:%S', time.localtime(last_frame_time))}</p>
        </div>
        
        <div class="status-box good">
            <h2>Connection Information</h2>
            <p>Listening on: <strong>0.0.0.0:8081/receive_video</strong></p>
        </div>
        
        <p><a href="/rx_video_feed">View Video Stream</a></p>
        <p><a href="/">Back to Home</a></p>
        <p><small>This page refreshes automatically every 5 seconds</small></p>
    </body>
    </html>
    """

@app.route('/')
def home():
    return """
    <html>
    <head><title>Video Receiver</title></head>
    <body>
        <h1>Video Receiver</h1>
        <p>This application receives video frames and displays them as a stream.</p>
        <ul>
            <li><a href="/rx_video_feed" target="_blank">View the received video stream</a></li>
            <li><a href="/status" target="_blank">View receiver status and statistics</a></li>
        </ul>
        <p>Listening for frames on: <strong>0.0.0.0:8081/receive_video</strong></p>
        <p>Current status: {}</p>
    </body>
    </html>
    """.format("Received frames" if current_frame is not None else "Waiting for frames...")

@app.route('/receive_video', methods=['POST'])
def receive_video():
    global current_frame, frames_received, last_frame_time, fps_estimate
    
    # Calculate FPS
    current_time = time.time()
    time_diff = current_time - last_frame_time
    last_frame_time = current_time
    
    # Only log every 10th frame to reduce console output
    frames_received += 1
    if frames_received % 10 == 0:
        print(f"Received frame #{frames_received}")
        
        # Update FPS estimate (with smoothing)
        if time_diff > 0:
            new_fps = 1.0 / time_diff
            fps_estimate = 0.7 * fps_estimate + 0.3 * new_fps  # Smoothed average
            print(f"Current FPS estimate: {fps_estimate:.1f}")

    # Get the base64-encoded frame from the POST request
    data = request.json
    frame_data = data['frame']
    
    # Only log data length occasionally
    if frames_received % 30 == 0:
        print(f"Frame data length: {len(frame_data)}")

    # Decode the frame from base64 format
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Only log occasionally
            if frames_received % 30 == 0:
                print(f"Successfully decoded frame with shape: {frame.shape}")
            
            # Set the current frame to be used in the MJPEG stream
            current_frame = frame
            return jsonify({'status': 'success', 'message': 'Frame received and processed successfully'}), 200
        else:
            print("Failed to decode frame!")
            return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 400

@app.route('/rx_video_feed')
def video_feed():
    # Return the MJPEG stream to the browser
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on all interfaces (0.0.0.0) to allow external connections
    app.run(host='0.0.0.0', port=8081)