import cv2
from flask import Flask, request, jsonify, Response
import base64
import numpy as np
import time
import statistics
from collections import deque

app = Flask(__name__)

# Initialize variables
current_frame = None
frames_received = 0
last_frame_time = time.time()
fps_estimate = 0

# Performance metrics
frame_sizes = deque(maxlen=30)       # Last 30 frame sizes
frame_times = deque(maxlen=30)       # Last 30 frame processing times
frame_intervals = deque(maxlen=30)   # Last 30 intervals between frames
failed_decodes = 0                   # Count of failed frame decodes

# Function to generate MJPEG stream from received frames
def generate():
    global current_frame
    print("MJPEG stream generator started - waiting for frames...")
    frame_count = 0
    last_log_time = time.time()
    
    while True:
        if current_frame is not None:
            frame_count += 1
            
            # Log only once per second to reduce console output
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                print(f"Streaming frame #{frame_count} with shape: {current_frame.shape}")
                last_log_time = current_time
            
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            if not ret:
                print("Failed to encode frame for streaming!")
                continue
            
            jpeg_bytes = jpeg.tobytes()
            
            # Only log size occasionally to reduce console spam
            if frame_count % 30 == 0:
                print(f"Encoded JPEG size: {len(jpeg_bytes)} bytes")
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
        else:
            # Reduce waiting messages
            if time.time() - last_log_time >= 5.0:
                print("Waiting for frames from streamer...")
                last_log_time = time.time()
            
            time.sleep(0.1)  # Check less frequently when no frames

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Dynamic Quality Testing - Video Receiver</title>
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
            <h1>Dynamic Quality Testing - Video Receiver</h1>
            <p>This application receives video frames with adjustable quality for testing.</p>
            
            <div class="info-box">
                <h2>Quick Links</h2>
                <a href="/rx_video_feed" target="_blank" class="button">View Video Stream</a>
                <a href="/status" target="_blank" class="button">View Status</a>
            </div>
            
            <div class="info-box">
                <h2>Current Status</h2>
                <p>Status: {}</p>
                <p>Frames Received: {}</p>
                <p>Current FPS: {:.1f}</p>
            </div>
            
            <div class="info-box">
                <h2>Testing Information</h2>
                <p>This receiver works with the dynamic quality testing system.</p>
                <p>Use the <code>run_quality_tests.py</code> script on the sender to automatically test different quality settings.</p>
            </div>
        </div>
    </body>
    </html>
    """.format(
        "Receiving frames" if current_frame is not None else "Waiting for frames...",
        frames_received,
        fps_estimate
    )

@app.route('/status')
def status():
    # Calculate metrics
    avg_frame_size = statistics.mean(frame_sizes) if frame_sizes else 0
    avg_process_time = statistics.mean(frame_times) if frame_times else 0
    avg_interval = statistics.mean(frame_intervals) if frame_intervals else 0
    
    if frames_received > 0:
        failure_rate = (failed_decodes / frames_received) * 100
    else:
        failure_rate = 0
    
    actual_fps = 1 / avg_interval if avg_interval > 0 else 0
    
    return f"""
    <html>
    <head>
        <title>Video Receiver Status</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .status-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metric {{ display: flex; margin: 5px 0; }}
            .metric-name {{ width: 200px; font-weight: bold; }}
            .metric-value {{ flex-grow: 1; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; 
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Receiver Status</h1>
            
            <div class="status-box">
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
                    <div class="metric-value">{fps_estimate:.1f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Last Frame:</div>
                    <div class="metric-value">{time.strftime('%H:%M:%S', time.localtime(last_frame_time))}</div>
                </div>
            </div>
            
            <div class="status-box">
                <h2>Performance Metrics</h2>
                
                <div class="metric">
                    <div class="metric-name">Average Frame Size:</div>
                    <div class="metric-value">{avg_frame_size / 1024:.1f} KB</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Processing Time:</div>
                    <div class="metric-value">{avg_process_time * 1000:.1f} ms</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Frame Interval:</div>
                    <div class="metric-value">{avg_interval * 1000:.1f} ms</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Actual FPS:</div>
                    <div class="metric-value">{actual_fps:.1f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Decode Failure Rate:</div>
                    <div class="metric-value">{failure_rate:.1f}%</div>
                </div>
            </div>
            
            <div class="status-box">
                <h2>Frame Information</h2>
                <p>Current Frame Resolution: {current_frame.shape[1]}x{current_frame.shape[0] if current_frame is not None else 'Unknown'}</p>
            </div>
            
            <p><a href="/rx_video_feed" class="button">View Video Stream</a></p>
            <p><small>This page refreshes automatically every 5 seconds</small></p>
        </div>
    </body>
    </html>
    """

@app.route('/receive_video', methods=['POST'])
def receive_video():
    global current_frame, frames_received, last_frame_time, fps_estimate
    global frame_sizes, frame_times, frame_intervals, failed_decodes
    
    # Record start time for processing time calculation
    start_time = time.time()
    
    # Calculate FPS
    current_time = time.time()
    time_diff = current_time - last_frame_time
    frame_intervals.append(time_diff)
    last_frame_time = current_time
    
    # Update FPS estimate with smoothing
    if time_diff > 0:
        new_fps = 1.0 / time_diff
        fps_estimate = 0.7 * fps_estimate + 0.3 * new_fps  # Smoothed average
    
    frames_received += 1
    if frames_received % 30 == 0:  # Log less frequently
        print(f"Received frame #{frames_received}, FPS: {fps_estimate:.1f}")

    # Get the base64-encoded frame from the POST request
    data = request.json
    frame_data = data['frame']
    
    # Record frame size
    frame_sizes.append(len(frame_data))
    
    # Decode the frame from base64 format
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            if frames_received % 60 == 0:  # Log less frequently
                print(f"Successfully decoded frame with shape: {frame.shape}")
            
            # Set the current frame to be used in the MJPEG stream
            current_frame = frame
            
            # Record processing time
            process_time = time.time() - start_time
            frame_times.append(process_time)
            
            return jsonify({'status': 'success', 'message': 'Frame received and processed successfully'}), 200
        else:
            print("Failed to decode frame!")
            failed_decodes += 1
            return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400
    except Exception as e:
        print(f"Error processing frame: {e}")
        failed_decodes += 1
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 400

@app.route('/get_metrics')
def get_metrics():
    # Calculate metrics
    avg_frame_size = statistics.mean(frame_sizes) if frame_sizes else 0
    avg_process_time = statistics.mean(frame_times) if frame_times else 0
    avg_interval = statistics.mean(frame_intervals) if frame_intervals else 0
    
    if frames_received > 0:
        failure_rate = (failed_decodes / frames_received) * 100
    else:
        failure_rate = 0
    
    actual_fps = 1 / avg_interval if avg_interval > 0 else 0
    
    # Calculate latency (frame interval)
    latency = avg_interval
    
    # Calculate quality metrics
    if current_frame is not None:
        # Estimate quality based on resolution
        frame_resolution = current_frame.shape[0] * current_frame.shape[1]
        max_resolution = 1920 * 1080  # Full HD as reference
        resolution_quality = min(100, (frame_resolution / max_resolution) * 100)
        
        # Estimate quality based on frame size (larger = better quality)
        size_quality = min(100, (avg_frame_size / 100000) * 100)  # 100KB as reference
        
        # Combined quality score
        quality_score = (resolution_quality * 0.7) + (size_quality * 0.3)
    else:
        quality_score = 0
    
    metrics = {
        "frame_delivery_time": latency,
        "frame_drop_rate": failure_rate,
        "visual_quality_score": quality_score,
        "actual_fps": actual_fps,
        "avg_frame_size": avg_frame_size,
        "avg_process_time": avg_process_time,
        "frames_received": frames_received
    }
    
    return jsonify(metrics)

@app.route('/rx_video_feed')
def video_feed():
    # Return the MJPEG stream to the browser
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on all interfaces to allow external connections
    app.run(host='0.0.0.0', port=8081)