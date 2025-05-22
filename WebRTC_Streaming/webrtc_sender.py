import asyncio
import json
import logging
import time
import uuid
import websockets
import argparse
import socket

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default signaling server URL (can be overridden by command line argument)
SIGNALING_SERVER = "ws://localhost:8765"

async def connect_to_signaling(pc, client_id):
    """Connect to the signaling server and handle signaling messages"""
    async with websockets.connect(SIGNALING_SERVER) as websocket:
        # Register with the signaling server
        await websocket.send(json.dumps({"type": "register", "id": client_id}))
        logger.info(f"Connected to signaling server as {client_id}")
        
        # Create and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        await websocket.send(json.dumps({
            "type": "offer",
            "sdp": pc.localDescription.sdp
        }))
        logger.info("Offer sent to signaling server")
        
        # Handle incoming messages
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "answer":
                logger.info("Received answer from signaling server")
                answer = RTCSessionDescription(sdp=data["sdp"], type="answer")
                await pc.setRemoteDescription(answer)
                logger.info("Remote description set. WebRTC connection should now be establishing.")
            
            elif data["type"] == "ice":
                logger.info("Received ICE candidate from signaling server")
                candidate = data["candidate"]
                await pc.addIceCandidate(candidate)

async def run_sender(pc, player, client_id):
    # Set up event handlers
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        logger.info('ICE candidate: %s', candidate)
        # In a real implementation, we would send this to the signaling server
    
    pc.on("iceconnectionstatechange", lambda: logger.info('ICE connection state is %s', pc.iceConnectionState))
    pc.on("signalingstatechange", lambda: logger.info('Signaling state is %s', pc.signalingState))

    # Add video track from the player
    if player.video:
        pc.addTrack(player.video)

    # Connect to signaling server
    signaling_task = asyncio.create_task(connect_to_signaling(pc, client_id))
    
    # Keep the connection alive
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        signaling_task.cancel()

async def main():
    global SIGNALING_SERVER
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="WebRTC Sender")
    parser.add_argument("--signaling-server",
                        default=SIGNALING_SERVER,
                        help="Signaling server URL (default: ws://localhost:8765)")
    parser.add_argument("--id",
                        default=f"sender_{uuid.uuid4().hex[:8]}",
                        help="Client ID (default: randomly generated)")
    parser.add_argument("--video-path",
                        default="../video/zidane.mp4",
                        help="Path to the video file (default: ../video/zidane.mp4)")
    
    args = parser.parse_args()
    
    # Update signaling server URL
    SIGNALING_SERVER = args.signaling_server
    
    # Use the provided or generated client ID
    client_id = args.id
    logger.info(f"Starting sender with ID: {client_id}")
    logger.info(f"Connecting to signaling server: {SIGNALING_SERVER}")
    
    # Use the provided video path
    video_path = args.video_path
    logger.info(f"Using video file: {video_path}")
    player = MediaPlayer(video_path)
    
    pc = RTCPeerConnection()
    
    await run_sender(pc, player, client_id)

if __name__ == "__main__":
    asyncio.run(main())