import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import uuid
import cv2
import time
from fractions import Fraction
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("webrtc_sender")
pcs = set()
relay = MediaRelay()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that reads from a video file.
    """

    kind = "video"

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1 / self.fps
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        print(f"Video opened: {self.width}x{self.height} @ {self.fps} FPS")

    async def recv(self):
        # Control frame rate
        now = time.time()
        wait_time = max(0, self.last_frame_time + self.frame_time - now)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            print("End of video, restarting...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Failed to restart video")
        
        # Convert to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = Fraction(1, int(self.fps))  # Use Fraction for time_base
        
        self.frame_count += 1
        self.last_frame_time = time.time()
        
        if self.frame_count % 30 == 0:
            print(f"Streaming frame #{self.frame_count}")
        
        return video_frame

async def index(request):
    content = open(os.path.join(ROOT, "webrtc_sender.html"), "r").read()
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

    # Open the video file
    video_path = os.path.join(os.path.dirname(ROOT), "Video_test", "BigBuckBunny.mp4")
    if not os.path.exists(video_path):
        video_path = input("Enter the path to your video file: ")
    
    video = VideoTransformTrack(video_path=video_path)
    pc.addTrack(video)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC video streaming server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Port for HTTP server (default: 8090)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/webrtc.js", javascript)
    app.router.add_post("/offer", offer)

    print(f"Starting WebRTC video streaming server at http://{args.host}:{args.port}")
    print(f"Open this URL in your browser to view the stream")
    
    try:
        web.run_app(app, host=args.host, port=args.port)
    except OSError as e:
        if "address already in use" in str(e):
            print(f"\nERROR: Port {args.port} is already in use.")
            print(f"Try using a different port with the --port option:")
            print(f"python3 {os.path.basename(__file__)} --port {args.port + 1}")
            sys.exit(1)
        else:
            raise