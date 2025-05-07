import cv2
import base64
import time
import requests
from flask import Flask, Response, request, jsonify
import threading
import statistics
import math

app = Flask(__name__)

# Configuration
receiver_ip = "192.168.2.169"  # Change this to the IP address of the machine running receive_video.py
receiver_port = 8081       # Port of the receiver
video_path = '/home/sattoutah/Bureau/git_mesurement_tc/Video_test/BigBuckBunny.mp4'

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

# Network condition tracking
network_error_count = 0
last_successful_send = time.time()
adaptive_fps = 30  # Start with 30 fps

# QoS parameters
adaptive_resolution = 0.75  # Start with 75% of original resolution
adaptive_quality = 80       # Start with 80% JPEG quality
auto_qos_enabled = True     # Enable automatic QoS adjustments

# Network performance metrics
response_times = []         # List of recent response times
bandwidth_estimates = []    # List of recent bandwidth estimates
packet_loss_rate = 0        # Estimated packet loss rate

# QoS thresholds
MAX_RESPONSE_TIME = 500     # ms
MIN_BANDWIDTH = 1000000     # bytes per second (approximately 1 Mbps)
MAX_PACKET_LOSS = 10        # percent

# Global frame counter for logging
frame_count = 0

# Function to calculate optimal QoS parameters based on network conditions
def calculate_qos_parameters():
    global adaptive_resolution, adaptive_quality, adaptive_fps
    
    if not auto_qos_enabled or not response_times:
        return  # Don't adjust if auto QoS is disabled or no data available
    
    # Calculate average response time (in ms)
    avg_response_time = statistics.mean(response_times) * 1000 if response_times else 100
    
    # Calculate average bandwidth (in bytes per second)
    avg_bandwidth = statistics.mean(bandwidth_estimates) if bandwidth_estimates else 5000000
    
    # Calculate network score (0-100, higher is better)
    response_score = max(0, 100 - (avg_response_time / MAX_RESPONSE_TIME * 100))
    bandwidth_score = min(100, (avg_bandwidth / MIN_BANDWIDTH) * 50)
    loss_score = max(0, 100 - (packet_loss_rate / MAX_PACKET_LOSS * 100))
    
    # Weighted average of scores
    network_score = (response_score * 0.3) + (bandwidth_score * 0.5) + (loss_score * 0.2)
    
    print(f"Network metrics - Response: {avg_response_time:.1f}ms, Bandwidth: {avg_bandwidth/1000000:.2f}Mbps, Loss: {packet_loss_rate}%")
    print(f"Network score: {network_score:.1f}/100")
    
    # Adjust QoS parameters based on network score
    if network_score >= 80:  # Excellent network
        new_resolution = 1.0
        new_quality = 95
        new_fps = 30
    elif network_score >= 60:  # Good network
        new_resolution = 0.75
        new_quality = 85
        new_fps = 30
    elif network_score >= 40:  # Fair network
        new_resolution = 0.5
        new_quality = 75
        new_fps = 20
    elif network_score >= 20:  # Poor network
        new_resolution = 0.35
        new_quality = 65
        new_fps = 15
    else:  # Very poor network
        new_resolution = 0.25
        new_quality = 50
        new_fps = 10
    
    # Only log if there's a significant change
    if (abs(adaptive_resolution - new_resolution) > 0.05 or 
        abs(adaptive_quality - new_quality) > 5 or 
        abs(adaptive_fps - new_fps) > 2):
        print(f"Adjusting QoS - Resolution: {adaptive_resolution:.2f} → {new_resolution:.2f}, "
              f"Quality: {adaptive_quality} → {new_quality}, "
              f"FPS: {adaptive_fps} → {new_fps}")
    
    # Apply new parameters
    adaptive_resolution = new_resolution
    adaptive_quality = new_quality
    adaptive_fps = new_fps

# Function to send frames to receiver
def send_frame_to_receiver(jpeg_bytes):
    global network_error_count, last_successful_send, adaptive_fps, frame_count
    global response_times, bandwidth_estimates, packet_loss_rate
    
    encoded_frame = base64.b64encode(jpeg_bytes).decode('utf-8')
    receiver_url = f"http://{receiver_ip}:{receiver_port}/receive_video"
    
    # Implement retry logic
    max_retries = 2
    retry_count = 0
    
    start_time = time.time()
    data_size = len(jpeg_bytes)
    
    while retry_count <= max_retries:
        try:
            # Only print every 10th frame to reduce console output
            if frame_count % 10 == 0:
                print(f"Sending frame to receiver at {receiver_url}...")
            
            # Send the frame and QoS parameters to the receiver
            response = requests.post(receiver_url, json={
                'frame': encoded_frame,
                'qos_params': {
                    'resolution_scale': adaptive_resolution,
                    'jpeg_quality': adaptive_quality,
                    'target_fps': adaptive_fps,
                    'auto_qos': auto_qos_enabled
                }
            }, timeout=5)
            
            # Calculate response time and bandwidth
            end_time = time.time()
            response_time = end_time - start_time
            bandwidth = data_size / response_time if response_time > 0 else 0
            
            # Update metrics (keep last 10 measurements)
            response_times.append(response_time)
            bandwidth_estimates.append(bandwidth)
            if len(response_times) > 10:
                response_times.pop(0)
            if len(bandwidth_estimates) > 10:
                bandwidth_estimates.pop(0)
            
            # Only print every 10th frame to reduce console output
            if frame_count % 10 == 0:
                print(f"Response from receiver: {response.status_code} - {response.text}")
                print(f"Response time: {response_time*1000:.1f}ms, Bandwidth: {bandwidth/1000000:.2f}Mbps")
            
            # Reset error count on success and update last successful time
            network_error_count = 0
            last_successful_send = time.time()
            
            # Recalculate packet loss rate (successful transmission)
            packet_loss_rate = max(0, packet_loss_rate - 1) if packet_loss_rate > 0 else 0
            
            # Calculate optimal QoS parameters based on network conditions
            if frame_count % 30 == 0:  # Recalculate every 30 frames
                calculate_qos_parameters()
                
            return True
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            network_error_count += 1
            
            # Increase packet loss rate estimate
            packet_loss_rate = min(100, packet_loss_rate + 5)
            
            # If this is the last retry, log the error
            if retry_count > max_retries:
                print(f"Error sending frame to receiver after {max_retries} retries: {e}")
                
                # Recalculate QoS parameters immediately on failure
                calculate_qos_parameters()
                
                return False
            else:
                print(f"Retry {retry_count}/{max_retries} after error: {e}")
                time.sleep(0.2)  # Shorter retry delay

# Function to encode and send frames
def generate():
    print("Entering generate function...")
    global network_error_count, adaptive_fps, frame_count
    global adaptive_resolution, adaptive_quality
    
    # Create a new capture object each time to avoid thread safety issues
    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        print("Error: Could not open video file in generate function.")
        return
    
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        # Reset frame count when starting a new stream
        frame_count = 0
        while local_cap.isOpened():
            # Check if we've had too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"Too many consecutive failures ({consecutive_failures}). Pausing for recovery...")
                time.sleep(5)  # Pause for 5 seconds to allow network to recover
                consecutive_failures = 0
                network_error_count = 0
                
                # Force recalculation of QoS parameters
                calculate_qos_parameters()
            
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
            
            frame_count += 1
            if frame_count % 5 == 0:  # Only print every 5th frame to reduce console spam
                print(f"Processing frame #{frame_count} at {adaptive_fps} FPS")
            
            # SEPARATE PROCESSING FOR LOCAL VIEW AND NETWORK TRANSMISSION
            
            # 1. Process frame for local viewing (full quality)
            local_frame = frame.copy()  # Make a copy for local viewing
            # Encode the local frame with high quality for local viewing
            local_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 95% quality for local view
            ret_local, local_jpeg = cv2.imencode('.jpg', local_frame, local_encode_params)
            if not ret_local:
                print("Error: Failed to encode local frame.")
                consecutive_failures += 1
                continue
            
            local_jpeg_bytes = local_jpeg.tobytes()
            
            # 2. Process frame for network transmission (dynamically optimized)
            # Calculate dynamic resolution based on adaptive_resolution parameter
            new_width = int(frame_width * adaptive_resolution)
            new_height = int(frame_height * adaptive_resolution)
            network_frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode the network frame with dynamic quality
            network_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(adaptive_quality)]
            ret_network, network_jpeg = cv2.imencode('.jpg', network_frame, network_encode_params)
            if not ret_network:
                print("Error: Failed to encode network frame.")
                consecutive_failures += 1
                continue
            
            network_jpeg_bytes = network_jpeg.tobytes()
            
            # Send optimized frame to receiver
            if send_frame_to_receiver(network_jpeg_bytes):
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
            
            # Yield the HIGH QUALITY frame to local browser
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + local_jpeg_bytes + b'\r\n\r\n')
            
            # Use adaptive frame rate based on network conditions
            time.sleep(1 / adaptive_fps)  # Adaptive FPS based on network conditions
    
    except Exception as e:
        print(f"Error in generate function: {e}")
    finally:
        # Always release the capture object
        local_cap.release()
        print("Video capture released.")

# Flask route to trigger the video stream (this is the route that VLC will use)
@app.route('/tx_video_feed')
def video_feed():
    print("Entering video_feed route...")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Dynamic QoS Video Streamer</title>
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
            <h1>Dynamic QoS Video Streamer</h1>
            <p>This application streams video with dynamic quality of service adjustments based on network conditions.</p>
            
            <div class="info-box">
                <h2>Quick Links</h2>
                <a href="/tx_video_feed" target="_blank" class="button">View Local Stream</a>
                <a href="/status" target="_blank" class="button">View QoS Status</a>
                <a href="/qos_controls" target="_blank" class="button">QoS Controls</a>
            </div>
            
            <div class="info-box">
                <h2>Video Information</h2>
                <p>Current video file: {}</p>
                <p>Original Resolution: {}x{}</p>
                <p>Local View: Full resolution, 95% quality</p>
                <p>Network Transmission: Dynamic resolution and quality based on network conditions</p>
            </div>
            
            <div class="info-box">
                <h2>Connection Information</h2>
                <p>Sending frames to: <strong>http://{}:{}/receive_video</strong></p>
                <p>To view the received video, visit: <strong>http://{}:{}/rx_video_feed</strong> in a browser</p>
            </div>
        </div>
    </body>
    </html>
    """.format(video_path, frame_width, frame_height, receiver_ip, receiver_port, receiver_ip, receiver_port)

@app.route('/start_stream', methods=['GET'])
def start_stream():
    print("Route /start_stream triggered")  # Debugging line
    return "Visit /tx_video_feed to view the stream and send frames to the receiver."

@app.route('/set_fps/<int:fps>')
def set_fps(fps):
    global adaptive_fps, auto_qos_enabled
    if 1 <= fps <= 60:  # Limit FPS to reasonable range
        adaptive_fps = fps
        auto_qos_enabled = False  # Disable auto QoS when manually setting parameters
        return f"FPS set to {fps} (Auto QoS disabled)"
    else:
        return "FPS must be between 1 and 60", 400

@app.route('/set_resolution/<float:scale>')
def set_resolution(scale):
    global adaptive_resolution, auto_qos_enabled
    if 0.1 <= scale <= 1.0:  # Limit scale to reasonable range
        adaptive_resolution = scale
        auto_qos_enabled = False  # Disable auto QoS when manually setting parameters
        return f"Resolution scale set to {scale} (Auto QoS disabled)"
    else:
        return "Resolution scale must be between 0.1 and 1.0", 400

@app.route('/set_quality/<int:quality>')
def set_quality(quality):
    global adaptive_quality, auto_qos_enabled
    if 10 <= quality <= 100:  # Limit quality to reasonable range
        adaptive_quality = quality
        auto_qos_enabled = False  # Disable auto QoS when manually setting parameters
        return f"JPEG quality set to {quality} (Auto QoS disabled)"
    else:
        return "JPEG quality must be between 10 and 100", 400

@app.route('/toggle_auto_qos/<int:enabled>')
def toggle_auto_qos(enabled):
    global auto_qos_enabled
    auto_qos_enabled = bool(enabled)
    return f"Auto QoS {'enabled' if auto_qos_enabled else 'disabled'}"

@app.route('/qos_controls')
def qos_controls():
    return f"""
    <html>
    <head>
        <title>QoS Controls</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .control-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .slider-container {{ margin: 10px 0; }}
            .slider {{ width: 80%; }}
            .button {{ display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; 
                     text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .button:hover {{ background-color: #0056b3; }}
            .button.active {{ background-color: #28a745; }}
            .button.inactive {{ background-color: #dc3545; }}
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
            
            function toggleAutoQos(enabled) {{
                fetch('/toggle_auto_qos/' + (enabled ? 1 : 0))
                    .then(response => response.text())
                    .then(data => {{
                        document.getElementById('auto-on').className = enabled ? 'button active' : 'button inactive';
                        document.getElementById('auto-off').className = enabled ? 'button inactive' : 'button active';
                        
                        // Enable/disable manual controls
                        const controls = document.querySelectorAll('.manual-control');
                        controls.forEach(control => {{
                            control.disabled = enabled;
                        }});
                    }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Dynamic QoS Controls</h1>
            
            <div class="control-box">
                <h2>Auto QoS</h2>
                <p>Current status: <strong>{('Enabled' if auto_qos_enabled else 'Disabled')}</strong></p>
                <div>
                    <a id="auto-on" href="javascript:toggleAutoQos(true)" 
                       class="button {'active' if auto_qos_enabled else 'inactive'}">Enable Auto QoS</a>
                    <a id="auto-off" href="javascript:toggleAutoQos(false)" 
                       class="button {'inactive' if auto_qos_enabled else 'active'}">Disable Auto QoS</a>
                </div>
                <p><small>Auto QoS automatically adjusts quality parameters based on network conditions</small></p>
            </div>
            
            <div class="control-box">
                <h2>Manual QoS Controls</h2>
                
                <div class="slider-container">
                    <label for="resolution">Resolution Scale: <span id="resolution-value">{adaptive_resolution:.2f}</span></label><br>
                    <input type="range" id="resolution" class="slider manual-control" 
                           min="0.1" max="1.0" step="0.05" value="{adaptive_resolution}"
                           onchange="setResolution(this.value)" {'disabled' if auto_qos_enabled else ''}>
                    <div><small>0.1 = 10% of original size, 1.0 = 100% of original size</small></div>
                </div>
                
                <div class="slider-container">
                    <label for="quality">JPEG Quality: <span id="quality-value">{adaptive_quality}</span></label><br>
                    <input type="range" id="quality" class="slider manual-control" 
                           min="10" max="100" step="5" value="{adaptive_quality}"
                           onchange="setQuality(this.value)" {'disabled' if auto_qos_enabled else ''}>
                    <div><small>10 = lowest quality, 100 = highest quality</small></div>
                </div>
                
                <div class="slider-container">
                    <label for="fps">Frame Rate: <span id="fps-value">{adaptive_fps}</span> FPS</label><br>
                    <input type="range" id="fps" class="slider manual-control" 
                           min="1" max="60" step="1" value="{adaptive_fps}"
                           onchange="setFps(this.value)" {'disabled' if auto_qos_enabled else ''}>
                    <div><small>1 = lowest frame rate, 60 = highest frame rate</small></div>
                </div>
            </div>
            
            <div class="control-box">
                <h2>Current Network Metrics</h2>
                <p>Response Time: {statistics.mean(response_times)*1000:.1f if response_times else 0} ms</p>
                <p>Bandwidth: {statistics.mean(bandwidth_estimates)/1000000:.2f if bandwidth_estimates else 0} Mbps</p>
                <p>Packet Loss: {packet_loss_rate}%</p>
            </div>
            
            <p><a href="/status" class="button">View Detailed Status</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/status')
def status():
    return f"""
    <html>
    <head>
        <title>Dynamic QoS Status</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .status-box {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
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
        </style>
    </head>
    <body>
        <h1>Dynamic QoS Status</h1>
        
        <div class="status-box {'good' if network_error_count == 0 else 'warning' if network_error_count < 5 else 'error'}">
            <h2>Network Status</h2>
            
            <div class="metric">
                <div class="metric-name">Response Time:</div>
                <div class="metric-value">
                    {statistics.mean(response_times)*1000:.1f if response_times else 0} ms
                    <div class="progress-bar">
                        <div class="progress {'progress-good' if statistics.mean(response_times)*1000 < 100 else 'progress-warning' if statistics.mean(response_times)*1000 < 300 else 'progress-error'}" 
                             style="width: {min(100, (statistics.mean(response_times)*1000 / MAX_RESPONSE_TIME) * 100) if response_times else 0}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Bandwidth:</div>
                <div class="metric-value">
                    {statistics.mean(bandwidth_estimates)/1000000:.2f if bandwidth_estimates else 0} Mbps
                    <div class="progress-bar">
                        <div class="progress {'progress-good' if statistics.mean(bandwidth_estimates) > 5000000 else 'progress-warning' if statistics.mean(bandwidth_estimates) > 2000000 else 'progress-error'}" 
                             style="width: {min(100, (statistics.mean(bandwidth_estimates) / MIN_BANDWIDTH) * 50) if bandwidth_estimates else 0}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Packet Loss:</div>
                <div class="metric-value">
                    {packet_loss_rate}%
                    <div class="progress-bar">
                        <div class="progress {'progress-good' if packet_loss_rate < 2 else 'progress-warning' if packet_loss_rate < 5 else 'progress-error'}" 
                             style="width: {packet_loss_rate}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Error Count:</div>
                <div class="metric-value">{network_error_count}</div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Frames Processed:</div>
                <div class="metric-value">{frame_count}</div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Last Successful Send:</div>
                <div class="metric-value">{time.strftime('%H:%M:%S', time.localtime(last_successful_send))}</div>
            </div>
        </div>
        
        <div class="status-box good">
            <h2>QoS Parameters</h2>
            
            <div class="metric">
                <div class="metric-name">Auto QoS:</div>
                <div class="metric-value">{('Enabled' if auto_qos_enabled else 'Disabled')}</div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Resolution Scale:</div>
                <div class="metric-value">
                    {adaptive_resolution:.2f} ({int(frame_width * adaptive_resolution)}x{int(frame_height * adaptive_resolution)})
                    <div class="progress-bar">
                        <div class="progress progress-good" style="width: {adaptive_resolution * 100}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-name">JPEG Quality:</div>
                <div class="metric-value">
                    {adaptive_quality}%
                    <div class="progress-bar">
                        <div class="progress progress-good" style="width: {adaptive_quality}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-name">Frame Rate:</div>
                <div class="metric-value">
                    {adaptive_fps} FPS
                    <div class="progress-bar">
                        <div class="progress progress-good" style="width: {(adaptive_fps / 60) * 100}%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="status-box good">
            <h2>Video Information</h2>
            <p>Video File: {video_path}</p>
            <p>Original Resolution: {frame_width}x{frame_height}</p>
            <p>Local View: Full resolution, 95% quality</p>
            <p>Network Transmission: {int(frame_width * adaptive_resolution)}x{int(frame_height * adaptive_resolution)} ({int(adaptive_resolution * 100)}% of original), {adaptive_quality}% quality</p>
        </div>
        
        <div class="status-box good">
            <h2>Connection Information</h2>
            <p>Sending to: {receiver_ip}:{receiver_port}</p>
        </div>
        
        <p><a href="/qos_controls">Adjust QoS Settings</a> | <a href="/tx_video_feed">View Video Stream</a></p>
        <p><small>This page refreshes automatically every 5 seconds</small></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Flask app...")  # Debugging line
    # Run the Flask app on port 5000, binding to all interfaces
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on port 5000, accessible from other machines
