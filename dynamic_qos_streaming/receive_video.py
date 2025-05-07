import cv2
from flask import Flask, request, jsonify, Response
import base64
import numpy as np
import time
import json

app = Flask(__name__)

# Initialize variables
current_frame = None
frames_received = 0
last_frame_time = time.time()
fps_estimate = 0

# QoS metrics
frame_sizes = []  # List of recent frame sizes
frame_quality = {}  # QoS parameters from sender
network_latency = []  # List of recent latency measurements

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
            
            # Only log size very occasionally to reduce processing overhead
            if local_frame_count % 60 == 0:
                print(f"Encoded JPEG size: {len(jpeg_bytes)} bytes")
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
        else:
            # Reduce waiting messages
            if time.time() - last_log_time >= 5.0:
                print("Waiting for frames from streamer...")
                last_log_time = time.time()
            
            time.sleep(0.03)  # Check frequently (33Hz) for better responsiveness at 30 FPS

@app.route('/status')
def status():
    # Calculate average frame size
    avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    
    # Calculate average latency
    avg_latency = sum(network_latency) / len(network_latency) if network_latency else 0
    
    # Get QoS parameters
    resolution_scale = frame_quality.get('resolution_scale', 'Unknown')
    jpeg_quality = frame_quality.get('jpeg_quality', 'Unknown')
    target_fps = frame_quality.get('target_fps', 'Unknown')
    auto_qos = frame_quality.get('auto_qos', 'Unknown')
    
    return f"""
    <html>
    <head>
        <title>Dynamic QoS Receiver Status</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .status-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .good {{ background-color: #d4edda; }}
            .warning {{ background-color: #fff3cd; }}
            .error {{ background-color: #f8d7da; }}
            .metric {{ display: flex; margin: 5px 0; }}
            .metric-name {{ width: 200px; font-weight: bold; }}
            .metric-value {{ flex-grow: 1; }}
            .progress-bar {{ height: 20px; background-color: #e9ecef; border-radius: 5px; margin-top: 5px; }}
            .progress {{ height: 100%; border-radius: 5px; }}
            .progress-good {{ background-color: #28a745; }}
            .progress-warning {{ background-color: #ffc107; }}
            .progress-error {{ background-color: #dc3545; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; 
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dynamic QoS Receiver Status</h1>
            
            <div class="status-box {'good' if current_frame is not None else 'error'}">
                <h2>Stream Status</h2>
                
                <div class="metric">
                    <div class="metric-name">Status:</div>
                    <div class="metric-value">{"Receiving frames" if current_frame is not None else "No frames received"}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Frames Received:</div>
                    <div class="metric-value">{frames_received}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Current FPS:</div>
                    <div class="metric-value">
                        {fps_estimate:.1f}
                        <div class="progress-bar">
                            <div class="progress {'progress-good' if fps_estimate > 20 else 'progress-warning' if fps_estimate > 10 else 'progress-error'}" 
                                 style="width: {min(100, (fps_estimate / 30) * 100)}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Average Frame Size:</div>
                    <div class="metric-value">
                        {avg_frame_size / 1024:.1f} KB
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Network Latency:</div>
                    <div class="metric-value">
                        {avg_latency * 1000:.1f} ms
                        <div class="progress-bar">
                            <div class="progress {'progress-good' if avg_latency < 0.1 else 'progress-warning' if avg_latency < 0.3 else 'progress-error'}" 
                                 style="width: {min(100, (avg_latency / 0.5) * 100)}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Last Frame:</div>
                    <div class="metric-value">{time.strftime('%H:%M:%S', time.localtime(last_frame_time))}</div>
                </div>
            </div>
            
            <div class="status-box good">
                <h2>Sender QoS Parameters</h2>
                
                <div class="metric">
                    <div class="metric-name">Auto QoS:</div>
                    <div class="metric-value">{auto_qos}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Resolution Scale:</div>
                    <div class="metric-value">
                        {resolution_scale}
                        <div class="progress-bar">
                            <div class="progress progress-good" style="width: {float(resolution_scale) * 100 if isinstance(resolution_scale, (int, float)) else 0}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">JPEG Quality:</div>
                    <div class="metric-value">
                        {jpeg_quality}%
                        <div class="progress-bar">
                            <div class="progress progress-good" style="width: {jpeg_quality if isinstance(jpeg_quality, (int, float)) else 0}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Target Frame Rate:</div>
                    <div class="metric-value">
                        {target_fps} FPS
                        <div class="progress-bar">
                            <div class="progress progress-good" style="width: {(target_fps / 60) * 100 if isinstance(target_fps, (int, float)) else 0}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="status-box good">
                <h2>Connection Information</h2>
                <p>Listening on: <strong>0.0.0.0:8081/receive_video</strong></p>
                <p>Frame resolution: {current_frame.shape[1]}x{current_frame.shape[0] if current_frame is not None else 'Unknown'}</p>
            </div>
            
            <p><a href="/rx_video_feed" class="button">View Video Stream</a></p>
            <p><small>This page refreshes automatically every 5 seconds</small></p>
        </div>
    </body>
    </html>
    """

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Dynamic QoS Video Receiver</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            .info-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .button { display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; 
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }
            .button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dynamic QoS Video Receiver</h1>
            <p>This application receives video frames with dynamic quality of service adjustments.</p>
            
            <div class="info-box">
                <h2>Quick Links</h2>
                <a href="/rx_video_feed" target="_blank" class="button">View Video Stream</a>
                <a href="/status" target="_blank" class="button">View Status</a>
            </div>
            
            <div class="info-box">
                <h2>System Information</h2>
                <p>Listening for frames on: <strong>0.0.0.0:8081/receive_video</strong></p>
                <p>Current status: {}</p>
                <p>Frames received: {}</p>
                <p>Current FPS: {:.1f}</p>
            </div>
            
            <div class="info-box">
                <h2>About Dynamic QoS</h2>
                <p>The sender automatically adjusts video quality parameters based on network conditions:</p>
                <ul>
                    <li>Resolution scaling to optimize bandwidth usage</li>
                    <li>JPEG quality adjustment for compression efficiency</li>
                    <li>Frame rate control for smooth playback</li>
                </ul>
                <p>Check the status page for real-time information about current QoS parameters.</p>
            </div>
        </div>
    </body>
    </html>
    """.format(
        "Receiving frames" if current_frame is not None else "Waiting for frames...",
        frames_received,
        fps_estimate
    )

@app.route('/receive_video', methods=['POST'])
def receive_video():
    global current_frame, frames_received, last_frame_time, fps_estimate
    global frame_sizes, frame_quality, network_latency
    
    # Record start time for latency calculation
    start_time = time.time()
    
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
            fps_estimate = 0.5 * fps_estimate + 0.5 * new_fps  # Smoothed average
            print(f"Current FPS estimate: {fps_estimate:.1f}")

    # Get the base64-encoded frame from the POST request
    data = request.json
    frame_data = data['frame']
    
    # Extract QoS parameters if provided
    if 'qos_params' in data:
        frame_quality = data['qos_params']
        if frames_received % 30 == 0:  # Log occasionally
            print(f"Received QoS parameters: {frame_quality}")
    
    # Track frame size
    frame_sizes.append(len(frame_data))
    if len(frame_sizes) > 30:  # Keep last 30 measurements
        frame_sizes.pop(0)
    
    # Minimize logging to reduce processing overhead
    if frames_received % 60 == 0:  # Only log every 60th frame
        print(f"Frame data length: {len(frame_data)}")

    # Decode the frame from base64 format
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Minimize logging to reduce processing overhead
            if frames_received % 60 == 0:  # Only log every 60th frame
                print(f"Successfully decoded frame with shape: {frame.shape}")
            
            # Set the current frame to be used in the MJPEG stream
            current_frame = frame
            
            # Calculate and record latency
            end_time = time.time()
            latency = end_time - start_time
            network_latency.append(latency)
            if len(network_latency) > 30:  # Keep last 30 measurements
                network_latency.pop(0)
            
            # Prepare response with network metrics
            response_data = {
                'status': 'success', 
                'message': 'Frame received and processed successfully',
                'metrics': {
                    'latency': latency,
                    'fps': fps_estimate,
                    'frame_size': len(frame_data)
                }
            }
            
            return jsonify(response_data), 200
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