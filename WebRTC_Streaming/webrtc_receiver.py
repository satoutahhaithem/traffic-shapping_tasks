import asyncio
import json
import logging
import time
import cv2
import numpy as np
import uuid
import websockets
import argparse
import socket

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the latest frame
latest_frame = None

# Default signaling server URL (can be overridden by command line argument)
SIGNALING_SERVER = "ws://localhost:8765"

async def connect_to_signaling(pc, client_id):
    """Connect to the signaling server and handle signaling messages"""
    async with websockets.connect(SIGNALING_SERVER) as websocket:
        # Register with the signaling server
        await websocket.send(json.dumps({"type": "register", "id": client_id}))
        logger.info(f"Connected to signaling server as {client_id}")
        
        # Handle incoming messages
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "offer":
                logger.info("Received offer from signaling server")
                offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                await pc.setRemoteDescription(offer)
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                # Send answer to signaling server
                await websocket.send(json.dumps({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                }))
                logger.info("Answer sent to signaling server")
            
            elif data["type"] == "ice":
                logger.info("Received ICE candidate from signaling server")
                candidate = data["candidate"]
                await pc.addIceCandidate(candidate)

async def run_receiver(pc, client_id):
    # Set up event handlers
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        logger.info('ICE candidate: %s', candidate)
        # In a real implementation, we would send this to the signaling server
    
    pc.on("iceconnectionstatechange", lambda: logger.info('ICE connection state is %s', pc.iceConnectionState))
    pc.on("signalingstatechange", lambda: logger.info('Signaling state is %s', pc.signalingState))

    @pc.on("track")
    def ontrack(track):
        logger.info('Track %s received', track.kind)

        if track.kind == 'video':
            @track.on("frame")
            def onframe(frame):
                global latest_frame
                # Convert frame to OpenCV format
                img = frame.to_ndarray(format="bgr24")
                latest_frame = img
                # In a real application, you would display this frame

    # Connect to signaling server
    signaling_task = asyncio.create_task(connect_to_signaling(pc, client_id))
    
    # Keep the connection alive and display frames
    try:
        while True:
            if latest_frame is not None:
                cv2.imshow("WebRTC Receiver", latest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            await asyncio.sleep(0.01) # Process events
    finally:
        cv2.destroyAllWindows()
        signaling_task.cancel()

async def main():
    global SIGNALING_SERVER
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="WebRTC Receiver")
    parser.add_argument("--signaling-server",
                        default=SIGNALING_SERVER,
                        help="Signaling server URL (default: ws://localhost:8765)")
    parser.add_argument("--id",
                        default=f"receiver_{uuid.uuid4().hex[:8]}",
                        help="Client ID (default: randomly generated)")
    
    args = parser.parse_args()
    
    # Update signaling server URL
    SIGNALING_SERVER = args.signaling_server
    
    # Use the provided or generated client ID
    client_id = args.id
    logger.info(f"Starting receiver with ID: {client_id}")
    logger.info(f"Connecting to signaling server: {SIGNALING_SERVER}")
    
    pc = RTCPeerConnection()
    await run_receiver(pc, client_id)

if __name__ == "__main__":
    asyncio.run(main())