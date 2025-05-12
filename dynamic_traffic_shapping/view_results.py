#!/usr/bin/env python3

import os
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

# Configuration
RESULTS_DIR = "test_results"
PORT = 8000

def list_test_results():
    """List all test results in the results directory."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' does not exist.")
        return []
    
    results = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".html"):
            timestamp = filename.replace("quality_test_report_", "").replace(".html", "")
            try:
                date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                results.append((filename, date_str, timestamp))
            except ValueError:
                # Skip files with invalid timestamp format
                continue
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x[2], reverse=True)
    return results

def generate_index_html():
    """Generate an index.html file that lists all test results."""
    results = list_test_results()
    
    if not results:
        print("No test results found.")
        return False
    
    with open(os.path.join(RESULTS_DIR, "index.html"), "w") as f:
        f.write("""
        <html>
        <head>
            <title>Dynamic Quality Testing Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .button { display: inline-block; padding: 8px 16px; background-color: #007bff;
                         color: white; text-decoration: none; border-radius: 4px; }
                .graph-thumbnail { transition: transform 0.3s ease; }
                .graph-thumbnail:hover { transform: scale(1.05); }
                .graph-container { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }
                .graph-item { flex: 1; min-width: 250px; text-align: center; }
                .graph-item img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
                .graph-item div { margin-top: 5px; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dynamic Quality Testing Results</h1>
                
                <div style="margin-bottom: 20px; padding: 15px; background-color: #f0f8ff; border-left: 4px solid #1e90ff;">
                    <h2>Network Analysis Graphs</h2>
                    <p>The following graphs show the relationship between controlled network parameters and measured metrics:</p>
                    <div class="graph-container">
                        <div class="graph-item">
                            <a href="bandwidth_graphs.png" target="_blank">
                                <img src="bandwidth_graphs.png" alt="Bandwidth Analysis" class="graph-thumbnail">
                                <div>Bandwidth Analysis</div>
                            </a>
                        </div>
                        <div class="graph-item">
                            <a href="delay_graphs.png" target="_blank">
                                <img src="delay_graphs.png" alt="Delay Analysis" class="graph-thumbnail">
                                <div>Delay Analysis</div>
                            </a>
                        </div>
                        <div class="graph-item">
                            <a href="loss_graphs.png" target="_blank">
                                <img src="loss_graphs.png" alt="Packet Loss Analysis" class="graph-thumbnail">
                                <div>Packet Loss Analysis</div>
                            </a>
                        </div>
                        <div class="graph-item">
                            <a href="combined_graph.png" target="_blank">
                                <img src="combined_graph.png" alt="Combined Analysis" class="graph-thumbnail">
                                <div>Combined Analysis (3D)</div>
                            </a>
                        </div>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px; padding: 15px; background-color: #fff0f5; border-left: 4px solid #ff69b4;">
                    <h2>Time Series Analysis</h2>
                    <p>The following graph shows how metrics change over time during testing:</p>
                    <div class="graph-container">
                        <div class="graph-item" style="flex: 1; min-width: 90%;">
                            <a href="time_series_graphs.png" target="_blank">
                                <img src="time_series_graphs.png" alt="Time Series Analysis" class="graph-thumbnail">
                                <div>Metrics Over Time</div>
                            </a>
                        </div>
                    </div>
                </div>
                
                <h2>Test Results</h2>
                <p>Select a test result to view:</p>
                
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Actions</th>
                    </tr>
        """)
        
        for filename, date_str, _ in results:
            f.write(f"""
                    <tr>
                        <td>{date_str}</td>
                        <td><a href="{filename}" class="button">View Results</a></td>
                    </tr>
            """)
        
        f.write("""
                </table>
                
                <p style="margin-top: 20px;">
                    <a href="javascript:window.location.reload()" class="button">Refresh List</a>
                </p>
            </div>
        </body>
        </html>
        """)
    
    return True

def start_server():
    """Start a simple HTTP server to serve the test results."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    # Change to the results directory
    os.chdir(RESULTS_DIR)
    
    # Create index.html if it doesn't exist or if there are no results
    if not os.path.exists("index.html") or os.path.getsize("index.html") == 0:
        if not generate_index_html():
            print("No test results to display.")
            return
    
    # Start the HTTP server
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    print(f"Starting server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    # Open the browser
    webbrowser.open(f"http://localhost:{PORT}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

def create_sample_results():
    """Create sample test results for demonstration purposes."""
    print("Creating sample test results for demonstration...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a sample HTML report
    html_report = os.path.join(RESULTS_DIR, f"quality_test_report_{timestamp}.html")
    with open(html_report, "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>Sample Quality Test Results - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2 {{ color: #333; }}
                .results-section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .note {{ background-color: #ffffcc; padding: 10px; border-left: 4px solid #ffeb3b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sample Quality Test Results</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="note">
                    <h3>Sample Data</h3>
                    <p>This is a sample test result created for demonstration purposes.
                    Run actual tests using <code>run_quality_tests.py</code> to see real results.</p>
                </div>
                
                <div class="results-section">
                    <h2>Test Parameters</h2>
                    <p>Resolution scales tested: 50%, 75%, 90%, 100%</p>
                    <p>JPEG qualities tested: 60%, 75%, 85%, 95%</p>
                    <p>Frame rates tested: 10, 15, 20, 30 FPS</p>
                    <p>Network conditions tested:</p>
                    <ul>
                        <li>Poor: 2mbit, 150ms, 3% loss</li>
                        <li>Fair: 4mbit, 80ms, 1% loss</li>
                        <li>Good: 6mbit, 40ms, 0.5% loss</li>
                        <li>Excellent: 10mbit, 20ms, 0% loss</li>
                    </ul>
                </div>
                
                <div class="results-section">
                    <h2>Sample Results</h2>
                    <table>
                        <tr>
                            <th>Test #</th>
                            <th>Network</th>
                            <th>Resolution</th>
                            <th>Quality</th>
                            <th>FPS</th>
                            <th>Bandwidth</th>
                            <th>Visual Quality</th>
                            <th>Smoothness</th>
                        </tr>
                        <tr>
                            <td>1</td>
                            <td>Excellent</td>
                            <td>100%</td>
                            <td>95%</td>
                            <td>30</td>
                            <td>8.50 Mbps</td>
                            <td>95.0</td>
                            <td>98.5</td>
                        </tr>
                        <tr>
                            <td>2</td>
                            <td>Good</td>
                            <td>90%</td>
                            <td>85%</td>
                            <td>30</td>
                            <td>6.20 Mbps</td>
                            <td>88.5</td>
                            <td>92.0</td>
                        </tr>
                        <tr>
                            <td>3</td>
                            <td>Fair</td>
                            <td>75%</td>
                            <td>75%</td>
                            <td>20</td>
                            <td>3.80 Mbps</td>
                            <td>75.0</td>
                            <td>78.5</td>
                        </tr>
                        <tr>
                            <td>4</td>
                            <td>Poor</td>
                            <td>50%</td>
                            <td>60%</td>
                            <td>15</td>
                            <td>1.90 Mbps</td>
                            <td>55.0</td>
                            <td>45.0</td>
                        </tr>
                    </table>
                </div>
                
                <div class="results-section">
                    <h2>Conclusions</h2>
                    <p>This sample data shows how different quality settings perform under various network conditions:</p>
                    <ul>
                        <li>Under excellent network conditions, high resolution and quality settings perform well</li>
                        <li>As network conditions degrade, lower resolution and quality settings become necessary</li>
                        <li>Frame rate has a significant impact on smoothness, especially under poor network conditions</li>
                    </ul>
                    <p>Run actual tests to see how your specific setup performs under different conditions.</p>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Sample HTML report saved to {html_report}")
    return True

def main():
    """Main function."""
    print("Dynamic Quality Testing Results Viewer")
    print("=====================================")
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' does not exist.")
        print("Creating sample results for demonstration...")
        create_sample_results()
    
    # Check if there are any results
    results = list_test_results()
    if not results:
        print("No test results found.")
        print("Creating sample results for demonstration...")
        create_sample_results()
    
    # Generate index.html
    generate_index_html()
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()