import asyncio
import json
import logging
import time
import cv2
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the latest frame
latest_frame = None

async def run_receiver(pc):
    # Set up event handlers using the correct pc.on() method
    pc.on("icecandidate", lambda candidate: logger.info('ICE candidate: %s', candidate))
    pc.on("iceconnectionstatechange", lambda: logger.info('ICE connection state is %s', pc.iceConnectionState))
    pc.on("signalingstatechange", lambda: logger.info('Signaling state is %s', pc.signalingState))

    # Use the correct @pc.on decorator for track events
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

    # Wait for offer from sender (this part needs signaling implementation)
    # For now, we'll manually wait for input
    print("Paste the sender's offer below and press Enter:")
    offer_sdp = input()
    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
    await pc.setRemoteDescription(offer)

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("Answer created. Please paste this answer into the sender's signaling input:")
    print(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

    # Keep the connection alive and display frames
    while True:
        if latest_frame is not None:
            cv2.imshow("WebRTC Receiver", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        await asyncio.sleep(0.01) # Process events

    cv2.destroyAllWindows()


async def main():
    pc = RTCPeerConnection()
    await run_receiver(pc)

if __name__ == "__main__":
    asyncio.run(main())