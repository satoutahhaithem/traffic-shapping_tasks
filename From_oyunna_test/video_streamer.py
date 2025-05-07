import cv2
import base64
import time
import requests
from flask import Flask, Response, request
import threading

app = Flask(__name__)

# Configuration
receiver_ip = "127.0.0.1"  # Change this to the IP address of the machine running receive_video.py
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

# Function to send frames to receiver
def send_frame_to_receiver(jpeg_bytes):
    encoded_frame = base64.b64encode(jpeg_bytes).decode('utf-8')
    try:
        receiver_url = f"http://{receiver_ip}:{receiver_port}/receive_video"
        print(f"Sending frame to receiver at {receiver_url}...")
        response = requests.post(receiver_url, json={'frame': encoded_frame}, timeout=10)
        print(f"Response from receiver: {response.status_code} - {response.text}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to receiver: {e}")
        return False

# Function to encode and send frames
def generate():
    print("Entering generate function...")
    
    # Create a new capture object each time to avoid thread safety issues
    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        print("Error: Could not open video file in generate function.")
        return
    
    try:
        frame_count = 0
        while local_cap.isOpened():
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
                print(f"Processing frame #{frame_count}")
            
            # Encode the frame in JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Failed to encode frame.")
                continue
            
            jpeg_bytes = jpeg.tobytes()
            
            # Send frame to receiver (not in a separate thread to avoid thread safety issues)
            send_frame_to_receiver(jpeg_bytes)
            
            # Yield the frame to stream to browser
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            
            # Add a small delay between frames for a smoother stream and to reduce CPU usage
            time.sleep(1 / 15)  # 15 fps (reduced from 30 to lower CPU usage)
    
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

if __name__ == '__main__':
    print("Starting Flask app...")  # Debugging line
    # Run the Flask app on port 5000
    app.run(host='127.0.0.1', port=5000)  # Run the Flask app on port 5000



