import cv2
from flask import Flask, request, jsonify, Response
import base64
import numpy as np
import time

app = Flask(__name__)

# Initialize a variable to hold the current frame
current_frame = None

# Function to generate MJPEG stream from received frames
def generate():
    global current_frame
    print("MJPEG stream generator started - waiting for frames...")
    frame_count = 0
    while True:
        if current_frame is not None:
            frame_count += 1
            # Log the frame size for debugging
            print(f"Streaming frame #{frame_count} with shape: {current_frame.shape}")

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            if not ret:
                print("Failed to encode frame for streaming!")
                continue
            jpeg_bytes = jpeg.tobytes()
            print(f"Encoded JPEG size: {len(jpeg_bytes)} bytes")

            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
        else:
            print("Waiting for frames from streamer...")
            time.sleep(1)  # Wait a bit before checking again

@app.route('/')
def home():
    return "Welcome to the Video Receiver! Visit /rx_video_feed to view the stream."

@app.route('/receive_video', methods=['POST'])
def receive_video():
    global current_frame
    print("Received a frame from streamer!")

    # Get the base64-encoded frame from the POST request
    data = request.json
    frame_data = data['frame']
    print(f"Frame data length: {len(frame_data)}")

    # Decode the frame from base64 format
    img_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        print(f"Successfully decoded frame with shape: {frame.shape}")
        # Set the current frame to be used in the MJPEG stream
        current_frame = frame
        return jsonify({'status': 'success', 'message': 'Frame received and processed successfully'}), 200
    else:
        print("Failed to decode frame!")
        return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400

@app.route('/rx_video_feed')
def video_feed():
    # Return the MJPEG stream to the browser
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on all interfaces and port 5001
    app.run(host='0.0.0.0', port=8081)