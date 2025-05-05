import cv2
import base64
import time
import requests
import socket
import subprocess
from flask import Flask, Response, request
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
video_path = './Video_test/BigBuckBunny.mp4'
# Adjust this value to simulate different network conditions (higher value = more delay)
NETWORK_DELAY = 0.033  # Default: ~30 fps (1/30 ≈ 0.033)
# Constant for normal playback speed on transmitter side
TRANSMITTER_DELAY = 0.033  # Fixed at ~30 fps

# Get the local IP address to use for communication
def get_local_ip():
    try:
        # Try to get the IP address using the network interface from tc_api.py
        result = subprocess.run("ip route | grep default | awk '{print $5}' | head -n 1",
                              shell=True, capture_output=True, text=True)
        interface = result.stdout.strip()
        
        if interface:
            # Get the IP address of this interface
            result = subprocess.run(f"ip addr show {interface} | grep 'inet ' | awk '{{print $2}}' | cut -d/ -f1",
                                  shell=True, capture_output=True, text=True)
            ip = result.stdout.strip()
            if ip:
                print(f"[INFO] Using network interface {interface} with IP {ip}")
                return ip
    except Exception as e:
        print(f"[ERROR] Error getting local IP: {e}")
    
    # Fallback: Get any non-localhost IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google DNS to determine local IP
        ip = s.getsockname()[0]
        s.close()
        print(f"[INFO] Using fallback IP detection: {ip}")
        return ip
    except Exception as e:
        print(f"[ERROR] Fallback IP detection failed: {e}")
        return "127.0.0.1"  # Last resort fallback

# Get the local IP address
LOCAL_IP = get_local_ip()
print(f"[INFO] Local IP address for communication: {LOCAL_IP}")

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
    # Simulate network delay before sending the frame
    # Ensure delay is never negative and cap it at a maximum value to prevent system from becoming unresponsive
    additional_delay = max(0, min(NETWORK_DELAY - TRANSMITTER_DELAY, 2.0))  # Cap at 2 seconds max

    print(f"[DEBUG] send_frame_to_gcp: NETWORK_DELAY={NETWORK_DELAY}, TRANSMITTER_DELAY={TRANSMITTER_DELAY}, additional_delay={additional_delay}")

    if additional_delay > 0:
        try:
            print(f"[DEBUG] send_frame_to_gcp: Sleeping for {additional_delay} seconds")
            time.sleep(additional_delay)
        except Exception as e:
            print(f"[ERROR] Error during sleep: {e}")

    encoded_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    try:
        # Use the detected local IP instead of localhost
        receiver_url = f"http://{LOCAL_IP}:5001/receive_video"
        print(f"[DEBUG] Sending frame to: {receiver_url}")
        
        # Add a shorter timeout to make packet loss more visible
        response = requests.post(receiver_url, json={'frame': encoded_frame}, timeout=0.5)
        
        print(f"[DEBUG] Response status: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout sending frame - likely due to packet loss")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error sending frame to receiver: {e}")

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

        # Use normal playback speed for transmitter side
        time.sleep(TRANSMITTER_DELAY)  # Fixed at ~30 fps
# Flask route to trigger the video stream (this is the route that VLC will use)
@app.route('/tx_video_feed')
def video_feed():
    print("Entering video_feed route...")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_stream', methods=['GET'])
def start_stream():
    print("[DEBUG] Route /start_stream triggered")
    print(f"[DEBUG] Request headers: {request.headers}")
    print(f"[DEBUG] Request remote address: {request.remote_addr}")
    return "Video streamer is running. Current delay: " + str(NETWORK_DELAY)

@app.route('/set_delay/<path:delay_value>', methods=['GET'])
def set_delay(delay_value):
    """
    Set the network delay to simulate different network conditions
    Examples:
    - /set_delay/0.033 (30 fps - normal)
    - /set_delay/0.1 (10 fps - slight delay)
    - /set_delay/0.5 (2 fps - significant delay)
    """
    global NETWORK_DELAY

    print(f"[DEBUG] Received request to set delay to: {delay_value}")
    print(f"[DEBUG] Current NETWORK_DELAY before update: {NETWORK_DELAY}")
    print(f"[DEBUG] Request headers: {request.headers}")
    print(f"[DEBUG] Request remote address: {request.remote_addr}")
    
    # Convert the delay_value to float, handling potential errors
    try:
        delay_value = float(delay_value)
        print(f"[DEBUG] Successfully converted delay_value to float: {delay_value}")
    except ValueError:
        print(f"[ERROR] Invalid delay value '{delay_value}' - not a valid number")
        return f"Error: Invalid delay value '{delay_value}'. Please provide a valid number.", 400
    except Exception as e:
        print(f"[ERROR] Unexpected error converting delay value: {str(e)}")
        return f"Error: Unexpected error processing delay value: {str(e)}", 500
    
    # Set minimum delay but allow higher values
    max_safe_delay = 5.0  # Warning threshold (0.2 fps)
    absolute_max_delay = 10.0  # Absolute maximum (0.1 fps)
    min_delay = 0.01  # 10ms minimum to prevent issues
    
    # Ensure delay is within acceptable range
    if delay_value > absolute_max_delay:
        NETWORK_DELAY = absolute_max_delay
        print(f"[WARNING] Delay value {delay_value} capped at {absolute_max_delay} seconds")
        return f"Warning: Delay value {delay_value} capped at {absolute_max_delay} seconds for system stability. Effective rate: {1/absolute_max_delay:.2f} fps. Transmitter remains at normal speed."
    elif delay_value > max_safe_delay:
        NETWORK_DELAY = delay_value
        print(f"[CAUTION] Delay value {delay_value} is very high")
        return f"Caution: Delay value {delay_value} is very high and may cause system instability. Effective rate: {1/delay_value:.2f} fps. Transmitter remains at normal speed."
    elif delay_value < min_delay:
        NETWORK_DELAY = min_delay
        print(f"[WARNING] Delay value {delay_value} increased to minimum {min_delay} seconds")
        return f"Warning: Delay value {delay_value} increased to minimum {min_delay} seconds. Effective rate: {1/min_delay:.2f} fps. Transmitter remains at normal speed."
    else:
        old_delay = NETWORK_DELAY
        NETWORK_DELAY = delay_value
        print(f"[INFO] NETWORK_DELAY updated from {old_delay} to {NETWORK_DELAY}")
        
        # Calculate quality level based on delay
        quality_level = ""
        if delay_value <= 0.033:
            quality_level = "Perfect"
        elif delay_value <= 0.066:
            quality_level = "Very Good"
        elif delay_value <= 0.1:
            quality_level = "Good"
        elif delay_value <= 0.2:
            quality_level = "Fair"
        elif delay_value <= 0.33:
            quality_level = "Poor"
        elif delay_value <= 0.5:
            quality_level = "Very Poor"
        elif delay_value <= 0.8:
            quality_level = "Bad"
        elif delay_value <= 1.0:
            quality_level = "Very Bad"
        else:
            quality_level = "Terrible"
            
        return f"Network delay set to {delay_value:.3f} seconds (approximately {1/delay_value:.2f} fps). Quality level: {quality_level}. Transmitter remains at normal speed."

#def send_video_on_start(): 
#    generate()

if __name__ == '__main__':
    
    # Automatically send the image when the app starts
    #send_video_on_start()
    print("[INFO] Starting Flask app on port 5000...")
    print(f"[INFO] Initial NETWORK_DELAY set to: {NETWORK_DELAY}")
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode for more detailed error messages


