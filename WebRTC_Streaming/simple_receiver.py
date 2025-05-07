#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("simple_receiver")
pcs = set()
relay = MediaRelay()

# Simple HTML page with video element and debugging info
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Simple WebRTC Receiver</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .video-container { margin: 20px 0; }
        video { max-width: 100%; border: 1px solid #ddd; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .connected { background-color: #d4edda; color: #155724; }
        .disconnected { background-color: #f8d7da; color: #721c24; }
        .connecting { background-color: #fff3cd; color: #856404; }
        .debug { font-family: monospace; white-space: pre-wrap; background: #f8f9fa; 
                padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto; }
        button { padding: 8px 16px; background: #4CAF50; color: white; border: none; 
                border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Simple WebRTC Receiver</h1>
    
    <div class="video-container">
        <h2>Video Stream</h2>
        <video id="video" autoplay playsinline controls></video>
    </div>
    
    <div id="status" class="status disconnected">
        Status: Disconnected
    </div>
    
    <div>
        <h2>Connection</h2>
        <p>Enter the sender's IP address and port:</p>
        <input type="text" id="sender-ip" placeholder="Sender IP" value="localhost">
        <input type="text" id="sender-port" placeholder="Sender Port" value="8090">
        <button id="connect" onclick="connect()">Connect</button>
        <button id="disconnect" onclick="disconnect()" disabled>Disconnect</button>
    </div>
    
    <div>
        <h2>Debug Information</h2>
        <div id="debug" class="debug">Waiting for connection...</div>
    </div>
    
    <script>
        // Global variables
        let pc = null;
        let debugInterval = null;
        
        // Connect to sender
        async function connect() {
            const video = document.getElementById('video');
            const connectBtn = document.getElementById('connect');
            const disconnectBtn = document.getElementById('disconnect');
            const statusDiv = document.getElementById('status');
            const debugDiv = document.getElementById('debug');
            const senderIp = document.getElementById('sender-ip').value;
            const senderPort = document.getElementById('sender-port').value;
            
            if (!senderIp) {
                alert('Please enter the sender IP address');
                return;
            }
            
            // Update UI
            connectBtn.disabled = true;
            disconnectBtn.disabled = false;
            statusDiv.className = 'status connecting';
            statusDiv.textContent = 'Status: Connecting...';
            
            try {
                // Create peer connection with STUN server
                pc = new RTCPeerConnection({
                    sdpSemantics: 'unified-plan',
                    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
                });
                
                // Add debugging
                debugDiv.textContent = 'Creating peer connection...';
                startDebug();
                
                // Handle ICE candidates
                pc.onicecandidate = (event) => {
                    if (event.candidate) {
                        console.log('ICE candidate:', event.candidate);
                    }
                };
                
                // Handle connection state changes
                pc.onconnectionstatechange = () => {
                    debugDiv.textContent += `\\nConnection state changed to: ${pc.connectionState}`;
                    console.log('Connection state:', pc.connectionState);
                    
                    if (pc.connectionState === 'connected') {
                        statusDiv.className = 'status connected';
                        statusDiv.textContent = 'Status: Connected';
                    } else if (pc.connectionState === 'disconnected' || 
                              pc.connectionState === 'failed' || 
                              pc.connectionState === 'closed') {
                        statusDiv.className = 'status disconnected';
                        statusDiv.textContent = 'Status: Disconnected';
                    }
                };
                
                // Handle ICE connection state changes
                pc.oniceconnectionstatechange = () => {
                    debugDiv.textContent += `\\nICE connection state changed to: ${pc.iceConnectionState}`;
                    console.log('ICE connection state:', pc.iceConnectionState);
                };
                
                // Handle signaling state changes
                pc.onsignalingstatechange = () => {
                    debugDiv.textContent += `\\nSignaling state changed to: ${pc.signalingState}`;
                    console.log('Signaling state:', pc.signalingState);
                };
                
                // Handle track events
                pc.ontrack = (event) => {
                    debugDiv.textContent += `\\nTrack received: ${event.track.kind}`;
                    console.log('Track received:', event.track.kind);
                    
                    if (event.track.kind === 'video') {
                        debugDiv.textContent += '\\nAssigning video track to video element';
                        video.srcObject = event.streams[0];
                        
                        // Add event listeners to video element
                        video.onloadedmetadata = () => {
                            debugDiv.textContent += `\\nVideo metadata loaded: ${video.videoWidth}x${video.videoHeight}`;
                        };
                        
                        video.onplay = () => {
                            debugDiv.textContent += '\\nVideo started playing';
                        };
                        
                        video.onerror = (e) => {
                            debugDiv.textContent += `\\nVideo error: ${video.error.message}`;
                        };
                    }
                };
                
                // Create offer
                debugDiv.textContent += '\\nCreating offer...';
                const offer = await pc.createOffer({
                    offerToReceiveVideo: true
                });
                await pc.setLocalDescription(offer);
                
                // Send offer to server
                debugDiv.textContent += '\\nSending offer to server...';
                const response = await fetch(`http://${senderIp}:${senderPort}/offer`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });
                
                // Get answer from server
                debugDiv.textContent += '\\nReceived answer from server';
                const answer = await response.json();
                await pc.setRemoteDescription(answer);
                debugDiv.textContent += '\\nSet remote description';
                
            } catch (error) {
                console.error('Error connecting:', error);
                debugDiv.textContent += `\\nError: ${error.message}`;
                statusDiv.className = 'status disconnected';
                statusDiv.textContent = 'Status: Error - ' + error.message;
                
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
                
                if (pc) {
                    pc.close();
                    pc = null;
                }
            }
        }
        
        // Disconnect
        function disconnect() {
            const video = document.getElementById('video');
            const connectBtn = document.getElementById('connect');
            const disconnectBtn = document.getElementById('disconnect');
            const statusDiv = document.getElementById('status');
            const debugDiv = document.getElementById('debug');
            
            // Stop debugging
            stopDebug();
            
            // Close peer connection
            if (pc) {
                pc.close();
                pc = null;
            }
            
            // Stop video
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            // Update UI
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            statusDiv.className = 'status disconnected';
            statusDiv.textContent = 'Status: Disconnected';
            debugDiv.textContent = 'Disconnected. Click Connect to try again.';
        }
        
        // Start debugging
        function startDebug() {
            const debugDiv = document.getElementById('debug');
            const video = document.getElementById('video');
            
            debugInterval = setInterval(() => {
                if (!pc) return;
                
                // Add video element status
                let videoStatus = '';
                if (video.srcObject) {
                    videoStatus = `Video element: ${video.videoWidth}x${video.videoHeight}, `;
                    videoStatus += video.paused ? 'paused' : 'playing';
                    if (video.error) videoStatus += `, error: ${video.error.message}`;
                } else {
                    videoStatus = 'Video element: No source';
                }
                
                // Add to debug div but keep last 20 lines only
                const lines = debugDiv.textContent.split('\\n');
                if (lines.length > 20) {
                    lines.shift();
                }
                lines.push(videoStatus);
                debugDiv.textContent = lines.join('\\n');
                
            }, 2000);
        }
        
        // Stop debugging
        function stopDebug() {
            if (debugInterval) {
                clearInterval(debugInterval);
                debugInterval = null;
            }
        }
        
        // Handle page unload
        window.onbeforeunload = () => {
            if (pc) {
                pc.close();
                pc = null;
            }
            if (debugInterval) {
                clearInterval(debugInterval);
            }
        };
    </script>
</body>
</html>
"""

async def index(request):
    return web.Response(content_type="text/html", text=HTML_CONTENT)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PeerConnection({id(pc)})"
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(f"{pc_id} {msg}", *args)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple WebRTC receiver")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8091, help="Port for HTTP server (default: 8091)")
    args = parser.parse_args()

    # Create web application
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    
    # Start server
    print(f"Simple WebRTC receiver running at http://{args.host}:{args.port}")
    print("Open this URL in your browser to connect to the sender")
    print("This version includes detailed debugging information")
    web.run_app(app, host=args.host, port=args.port, access_log=None)