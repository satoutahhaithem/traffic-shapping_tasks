import asyncio
import json
import logging
import time

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_sender(pc, player):
    @pc.signal
    def onicecandidate(candidate):
        logger.info('ICE candidate: %s', candidate)
        # In a real application, send this candidate to the receiver via signaling

    @pc.signal
    def oniceconnectionstatechange():
        logger.info('ICE connection state is %s', pc.iceConnectionState)

    @pc.signal
    def onsignalingstatechange():
        logger.info('Signaling state is %s', pc.signalingState)

    # Add video track from the player
    if player.video:
        pc.addTrack(player.video)

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    logger.info("Offer created. Please paste this offer into the receiver's signaling input:")
    print(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

    # Wait for answer from receiver (this part needs signaling implementation)
    # For now, we'll manually wait for input
    print("Paste the receiver's answer below and press Enter:")
    answer_sdp = input()
    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
    await pc.setRemoteDescription(answer)

    logger.info("Remote description set. WebRTC connection should now be establishing.")

    # Keep the connection alive
    while True:
        await asyncio.sleep(1)

async def main():
    # Replace with your video file path
    video_path = "video/zidane.mp4"
    player = MediaPlayer(video_path)

    pc = RTCPeerConnection()

    await run_sender(pc, player)

if __name__ == "__main__":
    asyncio.run(main())