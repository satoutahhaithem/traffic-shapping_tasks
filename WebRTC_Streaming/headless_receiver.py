#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("webrtc_headless_receiver")
pcs = set()
relay = MediaRelay()

# Class to handle received video frames without OpenCV display
class HeadlessVideoReceiver:
    def __init__(self):
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
    def frame_received(self, frame):
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Get frame dimensions
        if self.frame_width == 0:
            self.frame_width = frame.width
            self.frame_height = frame.height
            print(f"Received first frame with dimensions: {self.frame_width}x{self.frame_height}")
        
        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            # Smooth FPS calculation
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f"Received frame #{self.frame_count}, FPS: {self.fps:.1f}")

# Global video receiver instance
video_receiver = HeadlessVideoReceiver()

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

async def status(request):
    """Return current receiver status as JSON"""
    return web.json_response({
        "frames_received": video_receiver.frame_count,
        "fps": round(video_receiver.fps, 1),
        "resolution": f"{video_receiver.frame_width}x{video_receiver.frame_height}",
        "uptime": round(time.time() - start_time, 1),
        "connections": len(pcs)
    })

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
    app.router.add_get("/status", status)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    try:
        await site.start()
    except OSError as e:
        if "address already in use" in str(e):
            print(f"\nERROR: Port {port} is already in use.")
            print(f"Try using a different port with the --port option:")
            print(f"python3 {os.path.basename(sys.argv[0])} --port {port + 1}")
            return 1
        else:
            raise
    
    print(f"WebRTC headless receiver running at http://{host}:{port}")
    print("Open this URL in your browser to establish the connection")
    print("Check /status endpoint for receiver statistics")
    print("Press Ctrl+C to exit")
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    import uuid  # Import here to avoid issues with circular imports
    
    parser = argparse.ArgumentParser(description="WebRTC headless video receiver (no OpenCV display)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8091, help="Port for HTTP server (default: 8091)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Record start time for uptime calculation
    start_time = time.time()
    
    print("Starting WebRTC Headless Receiver")
    print("================================")
    print("This version doesn't use OpenCV for display, avoiding QT/Wayland issues")
    print("Video will only be visible in the browser, not in a separate window")
    
    # Run the asyncio event loop
    result = asyncio.run(run_server(args.host, args.port))
    if result == 1:
        sys.exit(1)