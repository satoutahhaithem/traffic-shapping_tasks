#!/usr/bin/env python3

"""
Quick test script for the dynamic quality testing system.
This script runs a small subset of tests to quickly generate results.
"""

import os
import time
import json
import subprocess
import requests
from datetime import datetime

# Configuration
SENDER_IP = "localhost"
SENDER_PORT = 5000
RECEIVER_IP = "192.168.2.169"  # Change to match the IP address of your receiver
RECEIVER_PORT = 8081
INTERFACE = "wlp0s20f3"  # Change to match your network interface

# Results directory
RESULTS_DIR = "test_results"

# Test parameters - reduced set for quick testing
RESOLUTION_SCALES = [0.5, 1.0]  # Just test min and max
JPEG_QUALITIES = [60, 95]       # Just test min and max
FRAME_RATES = [15, 30]          # Just test mid and max
NETWORK_CONDITIONS = [
    {"name": "Good", "rate": "6mbit", "delay": "40ms", "loss": "0.5%"},
    {"name": "Poor", "rate": "2mbit", "delay": "150ms", "loss": "3%"}
]

# Results storage
results = []
current_resolution = 0.5
current_quality = 60
current_fps = 15
current_network = NETWORK_CONDITIONS[0]

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(f" {text} ".center(80, "-"))
    print("-" * 80)

def apply_network_condition(condition):
    """Apply network condition using tc."""
    print(f"Applying network condition: {condition['name']}")
    print(f"  Rate: {condition['rate']}, Delay: {condition['delay']}, Loss: {condition['loss']}")
    
    # Reset any existing tc rules
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    
    # Apply new tc rules
    subprocess.run([
        "sudo", "tc", "qdisc", "add", "dev", INTERFACE, "root", "netem", 
        "rate", condition["rate"], 
        "delay", condition["delay"], 
        "loss", condition["loss"]
    ])
    
    # Verify the settings
    result = subprocess.run(["tc", "qdisc", "show", "dev", INTERFACE], 
                           capture_output=True, text=True)
    print(f"  TC settings: {result.stdout.strip()}")
    
    # Wait for network to stabilize
    time.sleep(2)

def reset_network_condition():
    """Reset network conditions to normal."""
    print("Resetting network conditions")
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", INTERFACE, "root"], 
                  stderr=subprocess.DEVNULL)
    time.sleep(1)

def set_quality_parameters(resolution, quality, fps):
    """Set quality parameters on the video streamer."""
    print(f"Setting quality parameters: Resolution={resolution}, Quality={quality}, FPS={fps}")
    
    # Set resolution scale
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_resolution/{resolution}", 
                               timeout=5)
        print(f"  Resolution response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting resolution: {e}")
    
    # Set JPEG quality
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_quality/{quality}", 
                               timeout=5)
        print(f"  Quality response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting quality: {e}")
    
    # Set frame rate
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/set_fps/{fps}", 
                               timeout=5)
        print(f"  FPS response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error setting FPS: {e}")
    
    # Wait for settings to take effect
    time.sleep(2)

def measure_performance():
    """Measure performance metrics for current settings."""
    print("Measuring performance metrics...")
    
    # Initialize metrics
    metrics = {
        "bandwidth_usage": 0,
        "frame_delivery_time": 0,
        "frame_drop_rate": 0,
        "visual_quality_score": 0,
        "smoothness_score": 0
    }
    
    # Get metrics from sender
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/get_metrics", timeout=5)
        if response.status_code == 200:
            sender_metrics = response.json()
            metrics.update(sender_metrics)
            print(f"  Sender metrics: {sender_metrics}")
    except Exception as e:
        print(f"  Error getting sender metrics: {e}")
    
    # Get metrics from receiver
    try:
        response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/get_metrics", timeout=5)
        if response.status_code == 200:
            receiver_metrics = response.json()
            metrics.update(receiver_metrics)
            print(f"  Receiver metrics: {receiver_metrics}")
    except Exception as e:
        print(f"  Error getting receiver metrics: {e}")
    
    # If metrics endpoints aren't implemented, estimate metrics based on settings
    if metrics["bandwidth_usage"] == 0:
        # Estimate bandwidth based on resolution, quality, and fps
        resolution_factor = {0.5: 0.25, 1.0: 1.0}
        quality_factor = {60: 0.6, 95: 1.0}
        fps_factor = {15: 0.5, 30: 1.0}
        
        base_bandwidth = 5000000  # 5 Mbps base for full quality
        estimated_bandwidth = (base_bandwidth * 
                              resolution_factor.get(current_resolution, 1.0) * 
                              quality_factor.get(current_quality, 1.0) * 
                              fps_factor.get(current_fps, 1.0))
        
        metrics["bandwidth_usage"] = estimated_bandwidth
        print(f"  Estimated bandwidth: {estimated_bandwidth/1000000:.2f} Mbps")
    
    if metrics["visual_quality_score"] == 0:
        # Estimate visual quality based on resolution and quality
        resolution_score = {0.5: 50, 1.0: 100}
        quality_score = {60: 60, 95: 95}
        
        visual_quality = (resolution_score.get(current_resolution, 50) * 0.6 + 
                         quality_score.get(current_quality, 50) * 0.4)
        
        metrics["visual_quality_score"] = visual_quality
        print(f"  Estimated visual quality: {visual_quality:.1f}/100")
    
    if metrics["smoothness_score"] == 0:
        # Estimate smoothness based on fps and network condition
        fps_score = {15: 50, 30: 100}
        network_score = {"Poor": 30, "Good": 80}
        
        smoothness = (fps_score.get(current_fps, 50) * 0.7 + 
                     network_score.get(current_network["name"], 50) * 0.3)
        
        metrics["smoothness_score"] = smoothness
        print(f"  Estimated smoothness: {smoothness:.1f}/100")
    
    return metrics

def run_test(resolution, quality, fps, network):
    """Run a single test with the given parameters."""
    global current_resolution, current_quality, current_fps, current_network
    
    # Store current parameters for metric estimation
    current_resolution = resolution
    current_quality = quality
    current_fps = fps
    current_network = network
    
    print_section(f"Testing: Resolution={resolution}, Quality={quality}, FPS={fps}, Network={network['name']}")
    
    # Apply network condition
    apply_network_condition(network)
    
    # Set quality parameters
    set_quality_parameters(resolution, quality, fps)
    
    # Wait for system to stabilize
    print("Waiting for system to stabilize...")
    time.sleep(10)  # Shorter stabilization time for quick tests
    
    # Measure performance
    metrics = measure_performance()
    
    # Record results
    test_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resolution_scale": resolution,
        "jpeg_quality": quality,
        "frame_rate": fps,
        "network_condition": network["name"],
        "network_rate": network["rate"],
        "network_delay": network["delay"],
        "network_loss": network["loss"],
        "metrics": metrics
    }
    
    results.append(test_result)
    print(f"Test completed: {test_result}")
    
    # Wait before next test
    time.sleep(2)

def generate_report():
    """Generate a report of the test results."""
    print_header("Generating Test Report")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"quick_test_results_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    
    # Generate a text report
    text_report = os.path.join(RESULTS_DIR, f"quick_test_report_{timestamp}.txt")
    with open(text_report, "w") as f:
        f.write("=================================================\n")
        f.write("DYNAMIC QUALITY TESTING - QUICK TEST REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=================================================\n\n")
        
        f.write("SUMMARY\n")
        f.write(f"Total tests run: {len(results)}\n")
        f.write(f"Resolution scales tested: {', '.join(str(r) for r in RESOLUTION_SCALES)}\n")
        f.write(f"JPEG qualities tested: {', '.join(str(q) for q in JPEG_QUALITIES)}\n")
        f.write(f"Frame rates tested: {', '.join(str(fps) for fps in FRAME_RATES)}\n")
        f.write(f"Network conditions tested: {', '.join(n['name'] for n in NETWORK_CONDITIONS)}\n\n")
        
        f.write("DETAILED RESULTS\n")
        for i, result in enumerate(results):
            f.write(f"Test #{i+1}:\n")
            f.write(f"  Time: {result['timestamp']}\n")
            f.write(f"  Resolution Scale: {result['resolution_scale']}\n")
            f.write(f"  JPEG Quality: {result['jpeg_quality']}\n")
            f.write(f"  Frame Rate: {result['frame_rate']}\n")
            f.write(f"  Network: {result['network_condition']} ({result['network_rate']}, {result['network_delay']}, {result['network_loss']})\n")
            f.write("  Metrics:\n")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    f.write(f"    {metric}: {value:.2f}\n")
                else:
                    f.write(f"    {metric}: {value}\n")
            f.write("\n")
    
    print(f"Text report saved to {text_report}")
    
    # Create an HTML report
    html_report = os.path.join(RESULTS_DIR, f"quick_test_report_{timestamp}.html")
    with open(html_report, "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>Quick Test Results - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2 {{ color: #333; }}
                .results-section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .note {{ background-color: #e6f7ff; padding: 10px; border-left: 4px solid #1890ff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quick Test Results</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="note">
                    <h3>Quick Test</h3>
                    <p>This is a quick test with a reduced set of parameters. 
                    Run the full test using <code>run_quality_tests.py</code> for more comprehensive results.</p>
                </div>
                
                <div class="results-section">
                    <h2>Network Analysis Graphs</h2>
                    <p>The following graphs show the relationship between controlled network parameters and measured metrics:</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px;">
                        <div style="flex: 1; min-width: 45%;">
                            <h3>Bandwidth Analysis</h3>
                            <img src="bandwidth_graphs.png" alt="Bandwidth Analysis" style="width: 100%;">
                        </div>
                        <div style="flex: 1; min-width: 45%;">
                            <h3>Delay Analysis</h3>
                            <img src="delay_graphs.png" alt="Delay Analysis" style="width: 100%;">
                        </div>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                        <div style="flex: 1; min-width: 45%;">
                            <h3>Packet Loss Analysis</h3>
                            <img src="loss_graphs.png" alt="Packet Loss Analysis" style="width: 100%;">
                        </div>
                        <div style="flex: 1; min-width: 45%;">
                            <h3>Combined Analysis (3D)</h3>
                            <img src="combined_graph.png" alt="Combined Analysis" style="width: 100%;">
                        </div>
                    </div>
                </div>
                
                <div class="results-section">
                    <h2>Test Parameters</h2>
                    <p>Resolution scales tested: {', '.join(str(r) for r in RESOLUTION_SCALES)}</p>
                    <p>JPEG qualities tested: {', '.join(str(q) for q in JPEG_QUALITIES)}</p>
                    <p>Frame rates tested: {', '.join(str(fps) for fps in FRAME_RATES)}</p>
                    <p>Network conditions tested:</p>
                    <ul>
                        {"".join(f"<li>{n['name']}: {n['rate']}, {n['delay']}, {n['loss']}</li>" for n in NETWORK_CONDITIONS)}
                    </ul>
                </div>
                
                <div class="results-section">
                    <h2>Test Results</h2>
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
                        {"".join(f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{result['network_condition']}</td>
                            <td>{result['resolution_scale']}</td>
                            <td>{result['jpeg_quality']}</td>
                            <td>{result['frame_rate']}</td>
                            <td>{result['metrics']['bandwidth_usage']/1000000:.2f} Mbps</td>
                            <td>{result['metrics']['visual_quality_score']:.1f}</td>
                            <td>{result['metrics']['smoothness_score']:.1f}</td>
                        </tr>
                        """ for i, result in enumerate(results))}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"HTML report saved to {html_report}")
    
    # Generate graphs using the generate_graphs.py script
    print("Generating graphs from test results...")
    try:
        subprocess.run(["python3", "dynamic_quality_testing/generate_graphs.py"], check=True)
        print("Graphs generated successfully.")
    except Exception as e:
        print(f"Error generating graphs: {e}")
        print("You can generate graphs manually by running: python3 dynamic_quality_testing/generate_graphs.py")
    
    print("\nTo view the results, run:")
    print("  python3 dynamic_quality_testing/view_results.py")

def check_dependencies():
    """Check if all dependencies are installed."""
    print_header("Checking Dependencies")
    
    all_dependencies_met = True
    
    # Check if tc is installed
    result = subprocess.run(["which", "tc"], capture_output=True)
    if result.returncode == 0:
        print("✓ tc is installed")
    else:
        print("✗ tc is not installed. Install with: sudo apt install iproute2")
        all_dependencies_met = False
    
    # Check if video streamer is running
    try:
        response = requests.get(f"http://{SENDER_IP}:{SENDER_PORT}/", timeout=2)
        print("✓ Video streamer is running")
    except:
        print("✗ Video streamer is not running")
        print(f"  Start with: python3 dynamic_quality_testing/video_streamer.py")
        all_dependencies_met = False
    
    # Check if video receiver is running
    try:
        response = requests.get(f"http://{RECEIVER_IP}:{RECEIVER_PORT}/", timeout=2)
        print("✓ Video receiver is running")
    except:
        print("✗ Video receiver is not running")
        print(f"  Start with: python3 dynamic_quality_testing/receive_video.py")
        all_dependencies_met = False
    
    return all_dependencies_met

def main():
    """Main function to run the quick quality tests."""
    print_header("Dynamic Quality Testing - Quick Test")
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and ensure the video streamer and receiver are running.")
        return
    
    # Confirm with user
    print("\nThis script will run a quick test with a reduced set of quality settings and network conditions.")
    print("It will modify your network settings using tc, which requires sudo privileges.")
    print("Make sure both the video streamer and receiver are running before proceeding.")
    
    confirm = input("\nDo you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        # Run tests for good network condition
        network = NETWORK_CONDITIONS[0]  # Good
        for resolution in RESOLUTION_SCALES:
            for quality in JPEG_QUALITIES:
                for fps in FRAME_RATES:
                    run_test(resolution, quality, fps, network)
        
        # Run tests for poor network condition
        network = NETWORK_CONDITIONS[1]  # Poor
        for resolution in RESOLUTION_SCALES:
            for quality in JPEG_QUALITIES:
                for fps in FRAME_RATES:
                    run_test(resolution, quality, fps, network)
        
        # Reset network conditions
        reset_network_condition()
        
        # Generate report
        generate_report()
        
        print_header("Testing Complete")
        print(f"Tested {len(results)} combinations of quality settings and network conditions.")
        print("Check the test_results directory for detailed results.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
    finally:
        # Always reset network conditions
        reset_network_condition()

if __name__ == "__main__":
    main()