import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import cv2
import time
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("webrtc_receiver")
pcs = set()
relay = MediaRelay()

# Class to handle received video frames
class VideoReceiver:
    def __init__(self):
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        
    def frame_received(self, frame):
        # Convert frame to numpy array for OpenCV
        self.current_frame = frame.to_ndarray(format="bgr24")
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            # Smooth FPS calculation
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f"Received frame #{self.frame_count}, FPS: {self.fps:.1f}")

# Global video receiver instance
video_receiver = VideoReceiver()

# OpenCV display thread
async def display_thread():
    cv2.namedWindow("WebRTC Receiver", cv2.WINDOW_NORMAL)
    
    while True:
        if video_receiver.current_frame is not None:
            # Add FPS and frame count text to the frame
            frame = video_receiver.current_frame.copy()
            cv2.putText(frame, f"Frame: {video_receiver.frame_count} FPS: {video_receiver.fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("WebRTC Receiver", frame)
        
        # Check for key press (q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # Short sleep to reduce CPU usage
        await asyncio.sleep(0.01)
    
    cv2.destroyAllWindows()

# Track class to receive video
class VideoTrackReceiver(MediaStreamTrack):
    kind = "video"
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        
    async def recv(self):
        frame = await self.track.recv()
        video_receiver.frame_received(frame)
        return frame

async def index(request):
    content = open(os.path.join(ROOT, "webrtc_receiver.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "webrtc.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # Handle cleanup when connection closes
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # Handle incoming tracks
    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        
        if track.kind == "video":
            video_track = VideoTrackReceiver(relay.subscribe(track))
            pc.addTrack(video_track)
            
        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def run_server(host, port):
    # Create web application
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/webrtc.js", javascript)
    app.router.add_post("/offer", offer)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    print(f"WebRTC receiver server running at http://{host}:{port}")
    print("Open this URL in your browser to establish the connection")
    print("Press Ctrl+C to exit")
    
    # Start display thread
    display_task = asyncio.create_task(display_thread())
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour
    except asyncio.CancelledError:
        pass
    finally:
        display_task.cancel()
        await runner.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC video receiver")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run the asyncio event loop
    asyncio.run(run_server(args.host, args.port))