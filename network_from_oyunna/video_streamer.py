import cv2
import base64
import time
import requests
from flask import Flask, Response, request
import threading

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
adaptive_fps = 60  # Start with 60 fps, will be reduced if network issues occur

# Global frame counter for logging
frame_count = 0

# Function to send frames to receiver
def send_frame_to_receiver(jpeg_bytes):
    global network_error_count, last_successful_send, adaptive_fps, frame_count
    
    encoded_frame = base64.b64encode(jpeg_bytes).decode('utf-8')
    receiver_url = f"http://{receiver_ip}:{receiver_port}/receive_video"
    
    # Implement retry logic
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Only print every 10th frame to reduce console output
            if frame_count % 10 == 0:
                print(f"Sending frame to receiver at {receiver_url}...")
            
            # Reduced timeout for faster error detection at high frame rates
            response = requests.post(receiver_url, json={'frame': encoded_frame}, timeout=5)
            
            # Only print every 10th frame to reduce console output
            if frame_count % 10 == 0:
                print(f"Response from receiver: {response.status_code} - {response.text}")
            
            # Reset error count on success and update last successful time
            network_error_count = 0
            last_successful_send = time.time()
            
            # If we've been successful for a while, gradually increase FPS back up
            if time.time() - last_successful_send > 10 and adaptive_fps < 60:
                adaptive_fps += 5
                print(f"Network conditions improving, increasing FPS to {adaptive_fps}")
                
            return True
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            network_error_count += 1
            
            # If this is the last retry, log the error
            if retry_count > max_retries:
                print(f"Error sending frame to receiver after {max_retries} retries: {e}")
                
                # Reduce FPS if we're having network issues
                if network_error_count > 5 and adaptive_fps > 3:
                    adaptive_fps -= 2
                    print(f"Network conditions deteriorating, reducing FPS to {adaptive_fps}")
                
                return False
            else:
                print(f"Retry {retry_count}/{max_retries} after error: {e}")
                time.sleep(0.2)  # Shorter retry delay for 60 FPS operation

# Function to encode and send frames
def generate():
    print("Entering generate function...")
    global network_error_count, adaptive_fps, frame_count
    
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
                adaptive_fps = max(3, adaptive_fps - 2)  # Reduce FPS but not below 3
            
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
            
            # Resize the frame to reduce bandwidth for 60 FPS streaming
            # Smaller frame size allows for faster transmission at high frame rates
            frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
            
            # Encode the frame in JPEG format with quality parameter (0-100)
            # Lower value = smaller file size but lower quality
            # Reduced quality for better performance at 60 FPS
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 70% quality for faster transmission
            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                print("Error: Failed to encode frame.")
                consecutive_failures += 1
                continue
            
            jpeg_bytes = jpeg.tobytes()
            
            # Send frame to receiver (not in a separate thread to avoid thread safety issues)
            if send_frame_to_receiver(jpeg_bytes):
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
            
            # Yield the frame to stream to browser
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            
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
    <head><title>Video Streamer</title></head>
    <body>
        <h1>Video Streamer</h1>
        <p>This application streams video from a file to a receiver and provides a local view.</p>
        <ul>
            <li><a href="/tx_video_feed" target="_blank">View the local video stream</a> (This will also send frames to the receiver)</li>
            <li><a href="/status" target="_blank">View network status and statistics</a></li>
            <li><a href="/start_stream" target="_blank">Get information about streaming</a></li>
        </ul>
        <p>Current video file: {}</p>
        <p>Resolution: {}x{}</p>
        <p>Sending frames to receiver at: <strong>http://{}:{}/receive_video</strong></p>
        <p>To view the received video, visit: <strong>http://{}:{}/rx_video_feed</strong> in a browser</p>
    </body>
    </html>
    """.format(video_path, frame_width, frame_height, receiver_ip, receiver_port, receiver_ip, receiver_port)

@app.route('/start_stream', methods=['GET'])
def start_stream():
    print("Route /start_stream triggered")  # Debugging line
    return "Visit /tx_video_feed to view the stream and send frames to the receiver."

@app.route('/set_fps/<int:fps>')
def set_fps(fps):
    global adaptive_fps
    if 1 <= fps <= 60:  # Limit FPS to reasonable range
        adaptive_fps = fps
        return f"FPS set to {fps}"
    else:
        return "FPS must be between 1 and 60", 400

@app.route('/status')
def status():
    return f"""
    <html>
    <head>
        <title>Video Streamer Status</title>
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
        <h1>Video Streamer Status</h1>
        
        <div class="status-box {'good' if network_error_count == 0 else 'warning' if network_error_count < 5 else 'error'}">
            <h2>Network Status</h2>
            <p>Error Count: {network_error_count}</p>
            <p>Current FPS: {adaptive_fps} (Target: 60)</p>
            <p>Frames Processed: {frame_count}</p>
            <p>Last Successful Send: {time.strftime('%H:%M:%S', time.localtime(last_successful_send))}</p>
            <p>Running for: {int(time.time() - last_successful_send)} seconds since last success</p>
        </div>
        
        <div class="status-box good">
            <h2>Video Information</h2>
            <p>Video File: {video_path}</p>
            <p>Resolution: {frame_width}x{frame_height}</p>
        </div>
        
        <div class="status-box good">
            <h2>Connection Information</h2>
            <p>Sending to: {receiver_ip}:{receiver_port}</p>
        </div>
        
        <div class="status-box">
            <h2>Manual Controls</h2>
            <p>Adjust Frame Rate:</p>
            <div style="display: flex; gap: 10px;">
                <a href="/set_fps/5" style="padding: 5px 10px; background: #eee; text-decoration: none;">5 FPS</a>
                <a href="/set_fps/10" style="padding: 5px 10px; background: #eee; text-decoration: none;">10 FPS</a>
                <a href="/set_fps/15" style="padding: 5px 10px; background: #eee; text-decoration: none;">15 FPS</a>
                <a href="/set_fps/20" style="padding: 5px 10px; background: #eee; text-decoration: none;">20 FPS</a>
                <a href="/set_fps/30" style="padding: 5px 10px; background: #eee; text-decoration: none;">30 FPS</a>
                <a href="/set_fps/60" style="padding: 5px 10px; background: #eee; text-decoration: none;">60 FPS</a>
            </div>
        </div>
        
        <p><a href="/tx_video_feed">View Video Stream</a></p>
        <p><small>This page refreshes automatically every 5 seconds</small></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Flask app...")  # Debugging line
    # Run the Flask app on port 5000, binding to all interfaces
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on port 5000, accessible from other machines



