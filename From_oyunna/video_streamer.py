import cv2
import base64
import time
import requests
from flask import Flask, Response, request
import threading

app = Flask(__name__)

# Replace with your video file
video_path = '/home/sattoutah/Bureau/git_mesurement_tc/Video_test/BigBuckBunny.mp4'

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


# Get the original resolution of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original resolution: {frame_width}x{frame_height}")

# Function to send frames to GCP asynchronously
def send_frame_to_gcp(jpeg):
    encoded_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    try:
        print("Sending frame to receiver...")
        response = requests.post('http://127.0.0.1:8081/receive_video', json={'frame': encoded_frame}, timeout=10)
        print(f"Response from receiver: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to receiver: {e}")

# Function to encode and send frames
def generate():
    print("Entering generat...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        else:
            print(f"Captured frame with shape: {frame.shape}, min pixel value: {frame.min()}, max pixel value: {frame.max()}")
        
        # Print the shape of the captured frame
        print(f"Captured frame with shape: {frame.shape}")
        
        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue
        
        # Convert to base64 string to send as JSON (for HTTP)
        #encoded_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        
        # Send frame to GCP server via POST request (example with HTTP)
        #response = requests.post('http://xx.xx.xx.xx:8080/receive_video', json={'frame': encoded_frame}, timeout=10)
        # Send frame to GCP asynchronously
        threading.Thread(target=send_frame_to_gcp, args=(jpeg,)).start()
        
         # Yield the frame to stream to VLC
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        #cv2.imshow("Transmitter Side Video", frame)  # Display the video locally
        #cv2.waitKey(1)  # 1 ms wait for updating the window
        
        # Check if the user pressed 'q' to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
        # Optionally, you can use WebSockets here for continuous streaming
        # client_socket.send(encoded_frame)

        # Simulate a small delay between frames for a smoother stream
        time.sleep(1 / 30)  # 30 fps (adjust if necessary)
# Flask route to trigger the video stream (this is the route that VLC will use)
@app.route('/tx_video_feed')
def video_feed():
    print("Entering video_feed route...")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return "Video Streamer: Visit /tx_video_feed to view the local stream or /start_stream to send video to the receiver."

@app.route('/start_stream', methods=['GET'])
def start_stream():
    print("Route /start_stream triggered")  # Debugging line
    # Start the video streaming in a new thread if it's not already running
    threading.Thread(target=run_generate, daemon=True).start()
    return "Video streaming started! Frames are being sent to the receiver."
def send_video_on_start():
    print("Starting automatic video streaming...")
    # Create a thread to run the generate function
    threading.Thread(target=run_generate, daemon=True).start()

def run_generate():
    # This function runs the generate function in a separate thread
    for frame in generate():
        # Just consume the frames to keep the generator running
        pass

if __name__ == '__main__':
    
    # Automatically send the video when the app starts
    send_video_on_start()
    print("Starting Flask app...")  # Debugging line
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on port 5000



