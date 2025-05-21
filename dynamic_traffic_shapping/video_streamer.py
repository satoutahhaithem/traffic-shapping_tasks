import cv2
import base64
import time
import requests
import json
from flask import Flask, Response, request, jsonify
import threading
import statistics
from collections import deque

app = Flask(__name__)

# Video file path
video_path = '/home/sattoutah/Bureau/git_mesurement_tc/video/zidane.mp4'

# Receiver configuration
receiver_ip = "192.168.2.169"  # Change this to the IP address of your receiver
receiver_port = 8081

# Quality parameters
resolution_scale = 1.0  # Scale factor for resolution (1.0 = original, 0.5 = half, etc.)
jpeg_quality = 85       # JPEG quality (0-100)
target_fps = 30         # Target frames per second

# Buffer settings for ultra-smooth playback
buffer_size = 60        # Doubled buffer size for ultra-smooth playback
use_buffering = True    # Enable frame buffering
frame_buffer = []       # Buffer to store frames
max_retries = 8         # Further increased number of retries for failed transmissions
retry_delay = 0.003     # Further reduced delay between retries (3ms)
thread_pool = []        # Thread pool for sending frames
max_threads = 15        # Increased maximum number of threads in the pool
priority_queue = []     # Priority queue for important frames
frame_sequence = 0      # Frame sequence counter for ordering

# Performance metrics
frame_sizes = deque(maxlen=30)       # Last 30 frame sizes
frame_times = deque(maxlen=30)       # Last 30 frame processing times
transmission_times = deque(maxlen=30) # Last 30 frame transmission times
failed_frames = 0                    # Count of failed frame transmissions
total_frames = 0                     # Total frames processed

# Check if the video file exists and can be opened
test_cap = cv2.VideoCapture(video_path)
if not test_cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    print("Please check if the file exists and is accessible")
    exit()

# Get the original resolution of the video
frame_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original resolution: {frame_width}x{frame_height}")

# Release the test capture object
test_cap.release()

# Function to send frames to receiver
def send_frame_to_receiver(jpeg_bytes, is_keyframe=False):
    global failed_frames, transmission_times, total_frames, frame_sequence
    
    # Only log occasionally to reduce console spam, but send EVERY frame
    should_log = (total_frames % 60 == 0)
    
    # Increment frame sequence
    current_sequence = frame_sequence
    frame_sequence += 1
    
    # Add frame sequence and keyframe flag to the data
    encoded_frame = base64.b64encode(jpeg_bytes).decode('utf-8')
    data = {
        'frame': encoded_frame,
        'sequence': current_sequence,
        'is_keyframe': is_keyframe
    }
    
    start_time = time.time()
    
    # Try multiple times with exponential backoff
    for attempt in range(max_retries):
        try:
            if should_log and attempt == 0:
                print("Sending frame to receiver...")
            
            # Reduce timeout for faster recovery from network issues
            timeout = 1.0 if attempt == 0 else 0.5
            
            # Use a session for connection pooling
            session = requests.Session()
            
            # Set headers for better performance
            headers = {
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            }
            
            response = session.post(
                f'http://{receiver_ip}:{receiver_port}/receive_video',
                json=data,
                timeout=timeout,
                headers=headers
            )
            
            # Record transmission time
            end_time = time.time()
            transmission_time = end_time - start_time
            transmission_times.append(transmission_time)
            
            if should_log:
                print(f"Response from receiver: {response.status_code} - {response.text}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            # Only log detailed errors occasionally to reduce console spam
            if (should_log or "Connection refused" in str(e)) and attempt == 0:
                print(f"Error sending frame to receiver (attempt {attempt+1}/{max_retries}): {e}")
            
            # Last attempt failed
            if attempt == max_retries - 1:
                failed_frames += 1
                return False
                
            # Wait before retrying with exponential backoff - use shorter delays
            time.sleep(retry_delay * (1.3 ** attempt))

# Function to encode and send frames
def generate():
    print("Entering generate function...")
    global total_frames, frame_sizes, frame_times, frame_buffer
    
    # Create a new capture object each time to avoid thread safety issues
    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        print("Error: Could not open video file in generate function.")
        return
    
    # Pre-fill buffer if buffering is enabled
    if use_buffering:
        print("Pre-filling frame buffer...")
        frame_index = 0
        while len(frame_buffer) < buffer_size and local_cap.isOpened():
            ret, frame = local_cap.read()
            if not ret:
                local_cap.release()
                local_cap = cv2.VideoCapture(video_path)
                continue
                
            # Apply resolution scaling if needed
            if resolution_scale != 1.0:
                new_width = int(frame_width * resolution_scale)
                new_height = int(frame_height * resolution_scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Apply subtle sharpening to improve visual quality
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)
            
            # Encode the frame in JPEG format with specified quality
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue
            
            # Mark every 10th frame as a keyframe for better synchronization
            is_keyframe = (frame_index % 10 == 0)
            frame_buffer.append((jpeg.tobytes(), is_keyframe))
            frame_index += 1
        
        print(f"Buffer filled with {len(frame_buffer)} frames")
    
    try:
        frame_count = 0
        last_frame_time = time.time()
        
        while local_cap.isOpened():
            start_time = time.time()
            
            # Calculate time since last frame
            frame_time = start_time - last_frame_time
            last_frame_time = start_time
            
            # Get frame (either from buffer or directly)
            if use_buffering and frame_buffer:
                # Get frame from buffer
                jpeg_bytes = frame_buffer.pop(0)
                
                # Read a new frame to refill the buffer
                ret, frame = local_cap.read()
                if ret:
                    # Process the new frame for the buffer
                    if resolution_scale != 1.0:
                        new_width = int(frame_width * resolution_scale)
                        new_height = int(frame_height * resolution_scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                    if ret:
                        frame_buffer.append(jpeg.tobytes())
                else:
                    # Try to loop the video by reopening it
                    local_cap.release()
                    local_cap = cv2.VideoCapture(video_path)
                    if not local_cap.isOpened():
                        print("Failed to reopen video file.")
                        break
            else:
                # Read and process frame directly
                ret, frame = local_cap.read()
                if not ret:
                    print("End of video or failed to capture frame.")
                    # Try to loop the video by reopening it
                    local_cap.release()
                    local_cap = cv2.VideoCapture(video_path)
                    if not local_cap.isOpened():
                        print("Failed to reopen video file.")
                        break
                    continue
                
                # Apply resolution scaling if needed
                if resolution_scale != 1.0:
                    new_width = int(frame_width * resolution_scale)
                    new_height = int(frame_height * resolution_scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode the frame in JPEG format with specified quality
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                if not ret:
                    print("Error: Failed to encode frame.")
                    continue
                
                jpeg_bytes = jpeg.tobytes()
            
            frame_count += 1
            total_frames += 1
            
            if frame_count % 5 == 0:  # Only print every 5th frame to reduce console spam
                print(f"Processing frame #{frame_count} at {target_fps} FPS, Resolution scale: {resolution_scale}, Quality: {jpeg_quality}")
            
            # Record frame size
            frame_sizes.append(len(jpeg_bytes))
            
            # Determine if this is a keyframe (every 10th frame)
            is_keyframe = (frame_count % 10 == 0)
            
            # Apply subtle sharpening to improve visual quality
            if is_keyframe:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                frame = cv2.filter2D(frame, -1, kernel)
            
            # Send frame to receiver using thread pool to avoid blocking the main thread
            # Clean up completed threads from the pool
            global thread_pool
            thread_pool = [t for t in thread_pool if t.is_alive()]
            
            # If thread pool is full, wait for a thread to complete
            while len(thread_pool) >= max_threads:
                time.sleep(0.0005)  # Shorter sleep
                thread_pool = [t for t in thread_pool if t.is_alive()]
            
            # Prioritize keyframes by using a higher thread priority
            if is_keyframe:
                # Create and start a new thread with higher priority for keyframes
                send_thread = threading.Thread(
                    target=lambda: send_frame_to_receiver(jpeg_bytes, is_keyframe=True)
                )
                send_thread.daemon = True
                send_thread.start()
                thread_pool.insert(0, send_thread)  # Add to front of pool for priority
            else:
                # Create and start a normal thread for regular frames
                send_thread = threading.Thread(
                    target=lambda: send_frame_to_receiver(jpeg_bytes, is_keyframe=False)
                )
                send_thread.daemon = True
                send_thread.start()
                thread_pool.append(send_thread)
            
            success = True  # Assume success for smoother playback
            
            # Yield the frame to stream to browser
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            
            # Calculate processing time
            process_time = time.time() - start_time
            frame_times.append(process_time)
            
            # Adaptive timing based on network conditions
            frame_duration = 1.0 / target_fps
            
            # Adaptive timing based on buffer state and thread pool size
            if not success or len(thread_pool) > max_threads * 0.8:
                # Network is struggling or thread pool is getting full
                extra_delay = 0.02  # 20ms extra delay (reduced from 30ms)
                sleep_time = max(0, frame_duration - process_time + extra_delay)
                
                # If buffer is low, add extra time to refill it
                if use_buffering and len(frame_buffer) < buffer_size / 2:
                    sleep_time += 0.005  # Add 5ms to allow buffer refill
            else:
                # Normal operation - precise timing
                sleep_time = max(0, frame_duration - process_time)
                
                # Adaptive timing based on buffer state
                if use_buffering:
                    if len(frame_buffer) < buffer_size / 3:
                        # Buffer is getting low, slow down slightly
                        sleep_time += 0.003
                    elif len(frame_buffer) > buffer_size * 0.8:
                        # Buffer is well-filled, can be more aggressive
                        sleep_time = max(0, sleep_time - 0.001)
            
            # Use a more precise sleep
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"Error in generate function: {e}")
    finally:
        # Always release the capture object
        local_cap.release()
        print("Video capture released.")

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Dynamic Quality Testing - Video Streamer</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .info-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white;
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .button:hover {{ background-color: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dynamic Quality Testing - Video Streamer</h1>
            <p>This application streams video with adjustable quality parameters for testing.</p>
            
            <div class="info-box">
                <h2>Quick Links</h2>
                <a href="/tx_video_feed" target="_blank" class="button">View Video Stream</a>
                <a href="/status" target="_blank" class="button">View Status</a>
                <a href="/quality_controls" target="_blank" class="button">Quality Controls</a>
            </div>
            
            <div class="info-box">
                <h2>Current Settings</h2>
                <p>Video File: {}</p>
                <p>Original Resolution: {}x{}</p>
                <p>Current Resolution: {}x{} ({}% of original)</p>
                <p>JPEG Quality: {}%</p>
                <p>Target FPS: {}</p>
            </div>
            
            <div class="info-box">
                <h2>Testing Information</h2>
                <p>Use the <code>run_quality_tests.py</code> script to automatically test different quality settings.</p>
                <p>You can also manually adjust quality parameters using the Quality Controls page.</p>
            </div>
        </div>
    </body>
    </html>
    """.format(
        video_path, 
        frame_width, frame_height,
        int(frame_width * resolution_scale), int(frame_height * resolution_scale),
        int(resolution_scale * 100),
        jpeg_quality,
        target_fps
    )

@app.route('/status')
def status():
    # Calculate metrics
    avg_frame_size = statistics.mean(frame_sizes) if frame_sizes else 0
    avg_process_time = statistics.mean(frame_times) if frame_times else 0
    avg_transmission_time = statistics.mean(transmission_times) if transmission_times else 0
    
    if total_frames > 0:
        failure_rate = (failed_frames / total_frames) * 100
    else:
        failure_rate = 0
    
    actual_fps = 1 / avg_process_time if avg_process_time > 0 else 0
    
    return f"""
    <html>
    <head>
        <title>Video Streamer Status</title>
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
            <h1>Video Streamer Status</h1>
            
            <div class="status-box">
                <h2>Quality Settings</h2>
                
                <div class="metric">
                    <div class="metric-name">Resolution Scale:</div>
                    <div class="metric-value">{resolution_scale:.2f} ({int(frame_width * resolution_scale)}x{int(frame_height * resolution_scale)})</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">JPEG Quality:</div>
                    <div class="metric-value">{jpeg_quality}%</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Target FPS:</div>
                    <div class="metric-value">{target_fps}</div>
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
                    <div class="metric-name">Transmission Time:</div>
                    <div class="metric-value">{avg_transmission_time * 1000:.1f} ms</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Actual FPS:</div>
                    <div class="metric-value">{actual_fps:.1f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Frame Failure Rate:</div>
                    <div class="metric-value">{failure_rate:.1f}%</div>
                </div>
                
                <div class="metric">
                    <div class="metric-name">Total Frames:</div>
                    <div class="metric-value">{total_frames}</div>
                </div>
            </div>
            
            <p><a href="/quality_controls" class="button">Adjust Quality Settings</a></p>
            <p><small>This page refreshes automatically every 5 seconds</small></p>
        </div>
    </body>
    </html>
    """

@app.route('/quality_controls')
def quality_controls():
    return f"""
    <html>
    <head>
        <title>Quality Controls</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .control-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .slider-container {{ margin: 10px 0; }}
            .slider {{ width: 80%; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; 
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
        </style>
        <script>
            function setResolution(value) {{
                fetch('/set_resolution/' + value)
                    .then(response => response.text())
                    .then(data => document.getElementById('resolution-value').textContent = value);
            }}
            
            function setQuality(value) {{
                fetch('/set_quality/' + value)
                    .then(response => response.text())
                    .then(data => document.getElementById('quality-value').textContent = value);
            }}
            
            function setFps(value) {{
                fetch('/set_fps/' + value)
                    .then(response => response.text())
                    .then(data => document.getElementById('fps-value').textContent = value);
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Quality Controls</h1>
            
            <div class="control-box">
                <h2>Adjust Quality Parameters</h2>
                
                <div class="slider-container">
                    <label for="resolution">Resolution Scale: <span id="resolution-value">{resolution_scale:.2f}</span></label><br>
                    <input type="range" id="resolution" class="slider" 
                           min="0.1" max="1.0" step="0.05" value="{resolution_scale}"
                           onchange="setResolution(this.value)">
                    <div><small>0.1 = 10% of original size, 1.0 = 100% of original size</small></div>
                </div>
                
                <div class="slider-container">
                    <label for="quality">JPEG Quality: <span id="quality-value">{jpeg_quality}</span></label><br>
                    <input type="range" id="quality" class="slider" 
                           min="10" max="100" step="5" value="{jpeg_quality}"
                           onchange="setQuality(this.value)">
                    <div><small>10 = lowest quality, 100 = highest quality</small></div>
                </div>
                
                <div class="slider-container">
                    <label for="fps">Frame Rate: <span id="fps-value">{target_fps}</span> FPS</label><br>
                    <input type="range" id="fps" class="slider" 
                           min="1" max="60" step="1" value="{target_fps}"
                           onchange="setFps(this.value)">
                    <div><small>1 = lowest frame rate, 60 = highest frame rate</small></div>
                </div>
            </div>
            
            <div class="control-box">
                <h2>Preset Configurations</h2>
                <a href="/set_preset/low" class="button">Low Quality</a>
                <a href="/set_preset/medium" class="button">Medium Quality</a>
                <a href="/set_preset/high" class="button">High Quality</a>
                <a href="/set_preset/ultra" class="button">Ultra Quality</a>
            </div>
            
            <p><a href="/status" class="button">View Status</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/set_resolution/<float:scale>')
def set_resolution(scale):
    global resolution_scale
    if 0.1 <= scale <= 1.0:  # Limit scale to reasonable range
        resolution_scale = scale
        return f"Resolution scale set to {scale}"
    else:
        return "Resolution scale must be between 0.1 and 1.0", 400

@app.route('/set_quality/<int:quality>')
def set_quality(quality):
    global jpeg_quality
    if 10 <= quality <= 100:  # Limit quality to reasonable range
        jpeg_quality = quality
        return f"JPEG quality set to {quality}"
    else:
        return "JPEG quality must be between 10 and 100", 400

@app.route('/set_fps/<int:fps>')
def set_fps(fps):
    global target_fps
    if 1 <= fps <= 60:  # Limit FPS to reasonable range
        target_fps = fps
        return f"FPS set to {fps}"
    else:
        return "FPS must be between 1 and 60", 400

@app.route('/set_preset/<preset>')
def set_preset(preset):
    global resolution_scale, jpeg_quality, target_fps
    
    if preset == 'low':
        resolution_scale = 0.25
        jpeg_quality = 50
        target_fps = 10
    elif preset == 'medium':
        resolution_scale = 0.5
        jpeg_quality = 65
        target_fps = 15
    elif preset == 'high':
        resolution_scale = 0.75
        jpeg_quality = 80
        target_fps = 20
    elif preset == 'ultra':
        resolution_scale = 1.0
        jpeg_quality = 95
        target_fps = 30
    else:
        return "Unknown preset", 400
    
    return f"Applied {preset} quality preset: Resolution={resolution_scale}, Quality={jpeg_quality}, FPS={target_fps}"

@app.route('/get_metrics')
def get_metrics():
    # Calculate metrics
    avg_frame_size = statistics.mean(frame_sizes) if frame_sizes else 0
    avg_process_time = statistics.mean(frame_times) if frame_times else 0
    avg_transmission_time = statistics.mean(transmission_times) if transmission_times else 0
    
    if total_frames > 0:
        failure_rate = (failed_frames / total_frames) * 100
    else:
        failure_rate = 0
    
    actual_fps = 1 / avg_process_time if avg_process_time > 0 else 0
    
    # Calculate bandwidth usage (bytes per second)
    bandwidth_usage = avg_frame_size * actual_fps
    
    # Calculate visual quality score (0-100)
    # Based on resolution scale and JPEG quality
    resolution_factor = resolution_scale * 100  # 0-100
    visual_quality_score = (resolution_factor * 0.6) + (jpeg_quality * 0.4)
    
    # Calculate smoothness score (0-100)
    # Based on actual FPS and target FPS
    fps_ratio = min(1.0, actual_fps / target_fps)
    smoothness_score = fps_ratio * 100
    
    metrics = {
        "bandwidth_usage": bandwidth_usage,
        "frame_delivery_time": avg_transmission_time,
        "frame_drop_rate": failure_rate,
        "visual_quality_score": visual_quality_score,
        "smoothness_score": smoothness_score,
        "actual_fps": actual_fps,
        "avg_frame_size": avg_frame_size,
        "avg_process_time": avg_process_time,
        "total_frames": total_frames
    }
    
    return jsonify(metrics)

@app.route('/tx_video_feed')
def tx_video_feed():
    print("Entering tx_video_feed route...")
    return """
    <html>
    <head>
        <title>Video Stream</title>
        <style>
            html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
            body { background-color: #000; }
            .video-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; }
            img { position: absolute; width: 100%; height: 100%; object-fit: contain; }
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
    print("Entering video_stream_data route...")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['GET'])
def start_stream():
    print("Route /start_stream triggered")  # Debugging line
    return "Visit /tx_video_feed to view the stream and send frames to the receiver."

if __name__ == '__main__':
    print("Starting Flask app...")  # Debugging line
    # Run the Flask app on port 5000, binding to all interfaces for network access
    app.run(host='0.0.0.0', port=5000)
