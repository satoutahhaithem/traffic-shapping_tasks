// Function to adjust video size
function adjustVideoSize(height) {
    console.log(`adjustVideoSize called with height: ${height}`);
    
    // Get video elements by ID
    const txVideo = document.getElementById('tx-video');
    const rxVideo = document.getElementById('rx-video');
    const currentHeightDisplay = document.getElementById('currentHeight');
    
    console.log(`tx-video element:`, txVideo);
    console.log(`rx-video element:`, rxVideo);
    console.log(`currentHeight element:`, currentHeightDisplay);
    
    if (!txVideo || !rxVideo) {
        console.error('Video elements not found!');
        alert('Video elements not found! Check the console for details.');
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = 'Status: Error - Video elements not found!';
        }
        return;
    }
    
    try {
        // Update iframe heights
        txVideo.height = height;
        rxVideo.height = height;
        
        // Update display
        if (currentHeightDisplay) {
            currentHeightDisplay.textContent = height;
        }
        
        // Add to debug log
        const debugContent = document.getElementById('debug-content');
        if (debugContent) {
            const timestamp = new Date().toISOString();
            const debugInfo = `[${timestamp}] Video size adjusted to ${height}px\n\n` +
                            debugContent.textContent;
            debugContent.textContent = debugInfo;
        }
        
        console.log(`Video size successfully adjusted to ${height}px`);
        alert(`Video size adjusted to ${height}px`);
    } catch (error) {
        console.error('Error adjusting video size:', error);
        alert(`Error adjusting video size: ${error.message}`);
    }
}