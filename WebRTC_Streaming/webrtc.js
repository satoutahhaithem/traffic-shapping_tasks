// Global variables
let pc = null;
let statsInterval = null;

// Configuration for WebRTC
const config = {
    sdpSemantics: 'unified-plan',
    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
};

// Start streaming
async function start() {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start');
    const stopButton = document.getElementById('stop');
    const statusDiv = document.getElementById('status');
    
    // Disable start button and enable stop button
    startButton.disabled = true;
    stopButton.disabled = false;
    
    // Update status
    statusDiv.className = 'status connecting';
    statusDiv.textContent = 'Status: Connecting...';
    
    try {
        // Create peer connection
        pc = new RTCPeerConnection(config);
        
        // Handle ICE candidate events
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('ICE candidate:', event.candidate);
            }
        };
        
        // Handle connection state changes
        pc.onconnectionstatechange = () => {
            console.log('Connection state:', pc.connectionState);
            if (pc.connectionState === 'connected') {
                statusDiv.className = 'status connected';
                statusDiv.textContent = 'Status: Connected';
                
                // Start collecting stats
                startStats();
            } else if (pc.connectionState === 'disconnected' || 
                      pc.connectionState === 'failed' || 
                      pc.connectionState === 'closed') {
                statusDiv.className = 'status disconnected';
                statusDiv.textContent = 'Status: Disconnected';
                
                // Stop collecting stats
                stopStats();
            }
        };
        
        // Handle track events (receiving video)
        pc.ontrack = (event) => {
            console.log('Track received:', event.track.kind);
            if (event.track.kind === 'video') {
                video.srcObject = event.streams[0];
            }
        };
        
        // Create offer
        const offer = await pc.createOffer({
            offerToReceiveVideo: true
        });
        await pc.setLocalDescription(offer);
        
        // Send offer to server
        const response = await fetch('/offer', {
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
        const answer = await response.json();
        await pc.setRemoteDescription(answer);
        
    } catch (error) {
        console.error('Error starting stream:', error);
        statusDiv.className = 'status disconnected';
        statusDiv.textContent = 'Status: Error - ' + error.message;
        
        // Re-enable start button and disable stop button
        startButton.disabled = false;
        stopButton.disabled = true;
    }
}

// Stop streaming
function stop() {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start');
    const stopButton = document.getElementById('stop');
    const statusDiv = document.getElementById('status');
    
    // Stop collecting stats
    stopStats();
    
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
    startButton.disabled = false;
    stopButton.disabled = true;
    statusDiv.className = 'status disconnected';
    statusDiv.textContent = 'Status: Disconnected';
}

// Start collecting stats
function startStats() {
    const statsDiv = document.getElementById('stats');
    
    // Collect stats every second
    statsInterval = setInterval(async () => {
        if (!pc) return;
        
        try {
            const stats = await pc.getStats();
            let statsOutput = '';
            
            stats.forEach(report => {
                if (report.type === 'inbound-rtp' && report.kind === 'video') {
                    statsOutput += `Resolution: ${report.frameWidth}x${report.frameHeight}\n`;
                    statsOutput += `Frames Received: ${report.framesReceived}\n`;
                    statsOutput += `Frames Decoded: ${report.framesDecoded}\n`;
                    statsOutput += `Frames Per Second: ${report.framesPerSecond}\n`;
                    statsOutput += `Packets Lost: ${report.packetsLost}\n`;
                    statsOutput += `Jitter: ${report.jitter.toFixed(3)}s\n`;
                }
                
                if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                    statsOutput += `Current RTT: ${(report.currentRoundTripTime * 1000).toFixed(2)}ms\n`;
                    statsOutput += `Available Bandwidth: ${Math.round(report.availableOutgoingBitrate / 1000)} kbps\n`;
                }
            });
            
            statsDiv.textContent = statsOutput || 'No stats available yet';
        } catch (error) {
            console.error('Error getting stats:', error);
        }
    }, 1000);
}

// Stop collecting stats
function stopStats() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
}

// Handle page unload
window.onbeforeunload = () => {
    if (pc) {
        pc.close();
        pc = null;
    }
};