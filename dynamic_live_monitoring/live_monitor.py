#!/usr/bin/env python3

"""
Live monitoring script for dynamic quality testing.
This script continuously updates graphs as new data comes in during testing.
"""

import os
import json
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

# Configuration
RESULTS_DIR = "test_results"
LIVE_DATA_FILE = os.path.join(RESULTS_DIR, "live_data.json")
PORT = 8090  # Changed from 8080 to avoid conflicts
UPDATE_INTERVAL = 1  # seconds - reduced to 1 second for more frequent updates

# Global variables
live_data = []
last_update_time = 0

def format_bandwidth(x, pos):
    """Format bandwidth in Mbps."""
    return f"{x/1000000:.1f}"

def generate_live_graphs():
    """Generate graphs from the live data."""
    if not live_data:
        print("No live data available yet.")
        # Create a simple empty graph
        plt.figure(figsize=(15, 10))
        plt.figtext(0.5, 0.5, "No data available yet. Waiting for updates...",
                   ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'live_graphs.png'))
        plt.close()
        return
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Extract timestamps and metrics
    timestamps = []
    bandwidth_usage = []
    frame_delivery_time = []
    frame_drop_rate = []
    visual_quality = []
    smoothness = []
    
    for result in live_data:
        if 'timestamp' in result:
            try:
                timestamp = datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S")
                timestamps.append(timestamp)
            except ValueError:
                timestamps.append(None)
        else:
            timestamps.append(None)
        
        if 'metrics' in result:
            bandwidth_usage.append(result['metrics'].get('bandwidth_usage', 0) / 1000000)  # Convert to Mbps
            frame_delivery_time.append(result['metrics'].get('frame_delivery_time', 0))
            frame_drop_rate.append(result['metrics'].get('frame_drop_rate', 0))
            visual_quality.append(result['metrics'].get('visual_quality_score', 0))
            smoothness.append(result['metrics'].get('smoothness_score', 0))
        else:
            bandwidth_usage.append(0)
            frame_delivery_time.append(0)
            frame_drop_rate.append(0)
            visual_quality.append(0)
            smoothness.append(0)
    
    # Filter out None timestamps and corresponding metrics
    valid_data = [(t, b, f, d, v, s) for t, b, f, d, v, s in zip(timestamps, bandwidth_usage, frame_delivery_time, frame_drop_rate, visual_quality, smoothness) if t is not None]
    
    if not valid_data:
        print("No valid data found.")
        return
    
    # Sort by timestamp
    valid_data.sort(key=lambda x: x[0])
    
    # Extract sorted data
    sorted_timestamps = [d[0] for d in valid_data]
    sorted_bandwidth = [d[1] for d in valid_data]
    sorted_delivery_time = [d[2] for d in valid_data]
    sorted_drop_rate = [d[3] for d in valid_data]
    sorted_visual_quality = [d[4] for d in valid_data]
    sorted_smoothness = [d[5] for d in valid_data]
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Bandwidth Usage over time
    axs[0, 0].plot(sorted_timestamps, sorted_bandwidth, 'o-', linewidth=2, markersize=6, color='blue')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Bandwidth Usage (Mbps)')
    axs[0, 0].set_title('Bandwidth Usage Over Time')
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot Frame Delivery Time over time
    axs[0, 1].plot(sorted_timestamps, sorted_delivery_time, 'o-', linewidth=2, markersize=6, color='red')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Frame Delivery Time (ms)')
    axs[0, 1].set_title('Frame Delivery Time Over Time')
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot Frame Drop Rate over time
    axs[1, 0].plot(sorted_timestamps, sorted_drop_rate, 'o-', linewidth=2, markersize=6, color='orange')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frame Drop Rate (%)')
    axs[1, 0].set_title('Frame Drop Rate Over Time')
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot Visual Quality and Smoothness over time
    ax1 = axs[1, 1]
    ax1.plot(sorted_timestamps, sorted_visual_quality, 'o-', linewidth=2, markersize=6, color='green', label='Visual Quality')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Visual Quality Score')
    ax1.set_title('Quality Scores Over Time')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(sorted_timestamps, sorted_smoothness, 'o-', linewidth=2, markersize=6, color='purple', label='Smoothness')
    ax2.set_ylabel('Smoothness Score')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'live_graphs.png'))
    print(f"Saved live graphs to {os.path.join(RESULTS_DIR, 'live_graphs.png')}")

def generate_html():
    """Generate HTML page with auto-refresh for live monitoring."""
    html_file = os.path.join(RESULTS_DIR, "live_monitor.html")
    
    with open(html_file, "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>Live Quality Monitoring</title>
            <meta http-equiv="refresh" content="{UPDATE_INTERVAL}">
            <script>
                // Auto-refresh the image every {UPDATE_INTERVAL} seconds
                function refreshImage() {{
                    var img = document.getElementById('liveGraph');
                    if (img) {{
                        img.src = 'live_graphs.png?t=' + new Date().getTime();
                    }}
                    setTimeout(refreshImage, {UPDATE_INTERVAL * 1000});
                }}
                
                // Start refreshing when page loads
                window.onload = function() {{
                    setTimeout(refreshImage, {UPDATE_INTERVAL * 1000});
                }};
            </script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2 {{ color: #333; }}
                .live-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background-color: #32CD32;
                    border-radius: 50%;
                    margin-right: 8px;
                    animation: pulse 1.5s infinite;
                }}
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.3; }}
                    100% {{ opacity: 1; }}
                }}
                .timestamp {{
                    color: #666;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                .graph-container {{
                    margin-top: 20px;
                    text-align: center;
                }}
                .graph-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .note {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #007bff;
                    margin-bottom: 20px;
                }}
                .refresh-button {{
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 10px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1><div class="live-indicator"></div> Live Quality Monitoring</h1>
                <div class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                
                <div class="note">
                    <p>This page automatically refreshes every {UPDATE_INTERVAL} second to show the latest data.</p>
                    <p>The graphs show how performance metrics change in real-time as network conditions are modified.</p>
                </div>
                
                <button class="refresh-button" onclick="document.getElementById('liveGraph').src='live_graphs.png?t='+new Date().getTime();">
                    Refresh Graphs Now
                </button>
                
                <div class="graph-container">
                    <img id="liveGraph" src="live_graphs.png?t={time.time()}" alt="Live Graphs">
                </div>
                
                <div class="timestamp" style="margin-top: 20px;">
                    Data points: {len(live_data)}
                </div>
            </div>
        </body>
        </html>
        """)
    
    return html_file

def monitor_data_file():
    """Monitor the live data file for changes and update graphs."""
    global live_data, last_update_time
    
    while True:
        try:
            # Check if the live data file exists
            if os.path.exists(LIVE_DATA_FILE):
                file_mod_time = os.path.getmtime(LIVE_DATA_FILE)
                
                # Load the data if the file has been modified or it's time to refresh
                if file_mod_time > last_update_time:
                    # Load the updated data
                    with open(LIVE_DATA_FILE, 'r') as f:
                        live_data = json.load(f)
                    
                    print(f"Loaded {len(live_data)} data points from {LIVE_DATA_FILE}")
                    last_update_time = file_mod_time
                
                # Always generate updated graphs and HTML, even if the data hasn't changed
                # This ensures the timestamp in the HTML is updated
                generate_live_graphs()
                generate_html()
            
            # Wait before checking again
            time.sleep(UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"Error monitoring data file: {e}")
            time.sleep(UPDATE_INTERVAL)

def start_http_server():
    """Start a simple HTTP server to serve the live monitoring page."""
    # Change to the results directory
    os.chdir(RESULTS_DIR)
    
    # Try to start the HTTP server, if port is in use, try another port
    port = PORT
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            server_address = ('', port)
            httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"Port {port} is already in use, trying port {port + 1}...")
                port += 1
                if attempt == max_attempts - 1:
                    print(f"Could not find an available port after {max_attempts} attempts.")
                    print("Please close any running servers or specify a different port.")
                    return
            else:
                raise
    
    print(f"Starting server at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Open the browser
    webbrowser.open(f"http://localhost:{port}/live_monitor.html")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

def create_initial_data():
    """Create initial live data file and empty graph if they don't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    if not os.path.exists(LIVE_DATA_FILE):
        with open(LIVE_DATA_FILE, 'w') as f:
            json.dump([], f)
        print(f"Created initial live data file: {LIVE_DATA_FILE}")
    
    # Create an empty graph file if it doesn't exist
    graph_file = os.path.join(RESULTS_DIR, 'live_graphs.png')
    if not os.path.exists(graph_file):
        # Create a simple empty graph
        plt.figure(figsize=(15, 10))
        plt.figtext(0.5, 0.5, "No data available yet. Waiting for updates...",
                   ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(graph_file)
        plt.close()
        print(f"Created initial empty graph file: {graph_file}")

def main():
    """Main function."""
    print("Starting Live Quality Monitoring")
    print("===============================")
    
    # Create initial data file
    create_initial_data()
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_data_file)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Generate initial graphs
    generate_live_graphs()
    generate_html()
    
    # Start the HTTP server
    start_http_server()

if __name__ == "__main__":
    main()