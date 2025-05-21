import cv2
from flask import Flask, request, jsonify, Response
import base64
import numpy as np
import time
import statistics
import threading
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

# Advanced frame buffering for ultra-smooth playback
frame_buffer = deque(maxlen=60)      # Doubled buffer size again for ultra-smooth playback
display_buffer = deque(maxlen=10)    # Separate buffer for display to ensure consistent frame rate
use_buffering = True                 # Enable frame buffering
buffer_lock = threading.Lock()       # Lock for thread-safe buffer access
display_lock = threading.Lock()      # Lock for thread-safe display buffer access
last_good_frame = None               # Store the last successfully decoded frame
last_keyframe = None                 # Store the last keyframe for reference
frame_counter = 0                    # Counter for frame sequencing
last_display_time = time.time()      # Time of last frame display
target_display_interval = 1.0/30.0   # Target interval between frames (30 FPS)
received_frames = {}                 # Dictionary to store received frames by sequence number
max_sequence_gap = 30                # Maximum allowed gap in sequence numbers

# Function to fill the display buffer from the frame buffer
def fill_display_buffer():
    global frame_buffer, display_buffer, current_frame, last_good_frame
    
    with buffer_lock:
        # If frame buffer has frames, move some to display buffer
        if frame_buffer and len(display_buffer) < 5:  # Keep display buffer partially filled
            with display_lock:
                while frame_buffer and len(display_buffer) < 8:  # Fill up to 8 frames
                    display_buffer.append(frame_buffer.popleft())
        
        # If display buffer is still empty but we have a current frame
        if not display_buffer and current_frame is not None:
            with display_lock:
                display_buffer.append(current_frame.copy())
        
        # If display buffer is still empty but we have a last good frame
        if not display_buffer and last_good_frame is not None:
            with display_lock:
                # Add the last good frame multiple times to prevent blinking
                for _ in range(3):
                    display_buffer.append(last_good_frame.copy())

# Function to generate MJPEG stream from received frames
def generate():
    global current_frame, frame_buffer, display_buffer, last_display_time, frame_counter
    print("MJPEG stream generator started - waiting for frames...")
    frame_count = 0
    last_log_time = time.time()
    
    # Start a background thread to continuously fill the display buffer
    buffer_filler = threading.Thread(target=lambda: [fill_display_buffer(), time.sleep(0.01)] * 1000000)
    buffer_filler.daemon = True
    buffer_filler.start()
    
    while True:
        current_time = time.time()
        time_since_last_frame = current_time - last_display_time
        
        # Maintain consistent frame rate
        if time_since_last_frame < target_display_interval * 0.8:
            # Not enough time has passed for next frame
            time.sleep(target_display_interval * 0.1)  # Short sleep
            continue
            
        # Get frame from display buffer or fallback options
        frame_to_encode = None
        
        # First try to get from display buffer
        with display_lock:
            if display_buffer:
                frame_to_encode = display_buffer.popleft()
        
        # If no frame in display buffer, try other sources
        if frame_to_encode is None:
            if current_frame is not None:
                frame_to_encode = current_frame.copy()
            elif last_good_frame is not None:
                frame_to_encode = last_good_frame.copy()
                
        # Fill display buffer if it's getting low
        if len(display_buffer) < 3:
            fill_display_buffer()
            
        if frame_to_encode is not None:
            frame_count += 1
            frame_counter += 1
            last_display_time = current_time  # Update last display time
            
            # Log only once per second to reduce console output
            if current_time - last_log_time >= 1.0:
                print(f"Streaming frame #{frame_count} with shape: {frame_to_encode.shape}, Buffer: {len(display_buffer)}/{len(frame_buffer)}")
                last_log_time = current_time
            
            # Apply subtle frame smoothing to reduce flickering
            frame_to_encode = cv2.GaussianBlur(frame_to_encode, (3, 3), 0)
            
            # Encode the frame as JPEG with higher quality for smoother playback
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # Increased quality
            ret, jpeg = cv2.imencode('.jpg', frame_to_encode, encode_params)
            if not ret:
                print("Failed to encode frame for streaming!")
                time.sleep(0.005)  # Shorter sleep to prevent CPU spinning
                continue
            
            jpeg_bytes = jpeg.tobytes()
            
            # Only log size occasionally to reduce console spam
            if frame_count % 60 == 0:
                print(f"Encoded JPEG size: {len(jpeg_bytes)} bytes")
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            
            # Precise timing control
            elapsed = time.time() - current_time
            remaining_time = target_display_interval - elapsed
            
            if remaining_time > 0:
                # Use a more precise sleep method
                sleep_start = time.time()
                while time.time() - sleep_start < remaining_time * 0.8:
                    time.sleep(0.001)  # Micro-sleep for more precise timing
        else:
            # Reduce waiting messages
            if current_time - last_log_time >= 5.0:
                print("Waiting for frames from streamer...")
                last_log_time = current_time
            
            time.sleep(0.02)  # Check more frequently for frames but avoid CPU spinning

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Dynamic Quality Testing - Video Receiver</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #121212; color: #f0f0f0; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #f0f0f0; }}
            .info-box {{ background-color: #1e1e1e; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white;
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .button:hover {{ background-color: #0056b3; }}
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
                <p>Current Frame Resolution: {f"{current_frame.shape[1]}x{current_frame.shape[0]}" if current_frame is not None else "Unknown"}</p>
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
    global frame_sizes, frame_times, frame_intervals, failed_decodes, frame_buffer
    
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
    if frames_received % 60 == 0:  # Log less frequently
        print(f"Received frame #{frames_received}, FPS: {fps_estimate:.1f}")

    # Get data from the POST request
    data = request.json
    frame_data = data['frame']
    sequence_number = data.get('sequence', 0)
    is_keyframe = data.get('is_keyframe', False)
    
    # Record frame size
    frame_sizes.append(len(frame_data))
    
    # Clean up old frames from received_frames dictionary
    current_keys = list(received_frames.keys())
    if current_keys:
        min_seq = min(current_keys)
        max_seq = max(current_keys)
        # Remove frames that are too old
        for seq in current_keys:
            if seq < max_seq - max_sequence_gap:
                del received_frames[seq]
    
    # Decode the frame from base64 format
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            if frames_received % 60 == 0:  # Log less frequently
                print(f"Successfully decoded frame with shape: {frame.shape}, Sequence: {sequence_number}, Keyframe: {is_keyframe}")
            
            # Store this as the last good frame
            global last_good_frame
            last_good_frame = frame.copy()
            
            # If this is a keyframe, store it separately
            if is_keyframe:
                global last_keyframe
                last_keyframe = frame.copy()
                if frames_received % 60 == 0:
                    print(f"Stored new keyframe at sequence {sequence_number}")
            
            # Store frame in sequence dictionary
            received_frames[sequence_number] = frame.copy()
            
            # Add frame to buffer if buffering is enabled
            if use_buffering:
                with buffer_lock:
                    # Add to buffer based on priority
                    if is_keyframe:
                        # Add keyframes to the front of the buffer for priority
                        frame_buffer.appendleft(frame)
                    else:
                        # Add regular frames to the end
                        frame_buffer.append(frame)
                    
                    # If this is the first frame, also set current_frame
                    if current_frame is None:
                        current_frame = frame
                        
                    # If display buffer is low, add directly to it as well
                    if len(display_buffer) < 2:
                        with display_lock:
                            display_buffer.append(frame.copy())
            else:
                # Set the current frame directly
                current_frame = frame
            
            # Record processing time
            process_time = time.time() - start_time
            frame_times.append(process_time)
            
            # Return success immediately to reduce round-trip time
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
    return """
    <html>
    <head>
        <title>Video Stream</title>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }}
            body {{ background-color: #000; }}
            .video-container {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; }}
            img {{ position: absolute; width: 100%; height: 100%; object-fit: contain; }}
        </style>
        <script>
            // Add fullscreen functionality
            document.addEventListener('DOMContentLoaded', function() {
                var videoImg = document.getElementById('videoStream');
                videoImg.addEventListener('click', function() {
                    if (document.fullscreenElement) {
                        document.exitFullscreen();
                    } else {
                        document.documentElement.requestFullscreen().catch(err => {
                            console.log(`Error attempting to enable full-screen mode: ${err.message}`);
                        });
                    }
                });
            });
        </script>
    </head>
    <body>
        <div class="video-container">
            <img id="videoStream" src="/video_stream_data" alt="Video Stream" />
        </div>
    </body>
    </html>
    """

@app.route('/video_stream_data')
def video_stream_data():
    # Return the MJPEG stream to the browser
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on all interfaces to allow external connections
    app.run(host='0.0.0.0', port=8081)