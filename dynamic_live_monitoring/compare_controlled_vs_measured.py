#!/usr/bin/env python3

"""
Script to compare controlled vs measured metrics.
This script generates separate graphs for controlled and measured metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
RESULTS_DIR = "test_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "live_data.json")  # Use live_data.json instead
GRAPHS_DIR = os.path.join(RESULTS_DIR, "comparison_graphs")

# Network conditions
NETWORK_CONDITIONS = [
    {"name": "Excellent", "rate": "10mbit", "delay": "20ms", "loss": "0%"},
    {"name": "Good", "rate": "6mbit", "delay": "40ms", "loss": "0.5%"},
    {"name": "Fair", "rate": "4mbit", "delay": "80ms", "loss": "1%"},
    {"name": "Poor", "rate": "2mbit", "delay": "150ms", "loss": "3%"}
]

def load_test_results():
    """Load test results from the JSON file."""
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file not found: {RESULTS_FILE}")
        return []
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return []

def get_tc_command(condition):
    """Get the tc command for a network condition."""
    return f"sudo tc qdisc change dev wlp0s20f3 root netem rate {condition['rate']} delay {condition['delay']} loss {condition['loss']}"

def filter_results_by_condition(results, condition_name):
    """Filter results by network condition name."""
    return [r for r in results if r.get('network_condition') == condition_name]

def extract_metrics(results):
    """Extract controlled and measured metrics from results."""
    controlled_bandwidth = []
    measured_bandwidth = []
    controlled_delay = []
    measured_delay = []
    controlled_loss = []
    measured_loss = []
    
    for result in results:
        # Extract controlled metrics
        if 'network_rate' in result:
            # Convert rate to Mbps (e.g., "10mbit" -> 10)
            rate_str = result['network_rate']
            if 'mbit' in rate_str:
                rate = float(rate_str.replace('mbit', ''))
                controlled_bandwidth.append(rate)
        
        if 'network_delay' in result:
            # Convert delay to ms (e.g., "20ms" -> 20)
            delay_str = result['network_delay']
            if 'ms' in delay_str:
                delay = float(delay_str.replace('ms', ''))
                controlled_delay.append(delay)
        
        if 'network_loss' in result:
            # Convert loss to percentage (e.g., "3%" -> 3)
            loss_str = result['network_loss']
            if '%' in loss_str:
                loss = float(loss_str.replace('%', ''))
                controlled_loss.append(loss)
        
        # Extract measured metrics
        if 'metrics' in result:
            metrics = result['metrics']
            if 'bandwidth_usage' in metrics:
                # Convert to Mbps
                measured_bandwidth.append(metrics['bandwidth_usage'] / 1000000)
            
            if 'frame_delivery_time' in metrics:
                measured_delay.append(metrics['frame_delivery_time'])
            
            if 'frame_drop_rate' in metrics:
                measured_loss.append(metrics['frame_drop_rate'])
    
    return {
        'controlled_bandwidth': controlled_bandwidth,
        'measured_bandwidth': measured_bandwidth,
        'controlled_delay': controlled_delay,
        'measured_delay': measured_delay,
        'controlled_loss': controlled_loss,
        'measured_loss': measured_loss
    }

def generate_comparison_graphs(condition, metrics):
    """Generate comparison graphs for controlled vs measured metrics."""
    # Create directory if it doesn't exist
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
    
    # Generate bandwidth comparison
    plt.figure(figsize=(12, 5))
    
    # Controlled Bandwidth
    plt.subplot(1, 2, 1)
    plt.title(f"Controlled Bandwidth - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Bandwidth (Mbps)")
    if metrics['controlled_bandwidth']:
        plt.plot(metrics['controlled_bandwidth'], 'b-o', label="Controlled")
        plt.axhline(y=np.mean(metrics['controlled_bandwidth']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['controlled_bandwidth']):.2f} Mbps")
    plt.grid(True)
    plt.legend()
    
    # Measured Bandwidth
    plt.subplot(1, 2, 2)
    plt.title(f"Measured Bandwidth - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Bandwidth (Mbps)")
    if metrics['measured_bandwidth']:
        plt.plot(metrics['measured_bandwidth'], 'g-o', label="Measured")
        plt.axhline(y=np.mean(metrics['measured_bandwidth']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['measured_bandwidth']):.2f} Mbps")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{condition['name'].lower()}_bandwidth_comparison.png"))
    plt.close()
    
    # Generate delay comparison
    plt.figure(figsize=(12, 5))
    
    # Controlled Delay
    plt.subplot(1, 2, 1)
    plt.title(f"Controlled Delay - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Delay (ms)")
    if metrics['controlled_delay']:
        plt.plot(metrics['controlled_delay'], 'b-o', label="Controlled")
        plt.axhline(y=np.mean(metrics['controlled_delay']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['controlled_delay']):.2f} ms")
    plt.grid(True)
    plt.legend()
    
    # Measured Delay
    plt.subplot(1, 2, 2)
    plt.title(f"Measured Frame Delivery Time - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Frame Delivery Time (ms)")
    if metrics['measured_delay']:
        plt.plot(metrics['measured_delay'], 'g-o', label="Measured")
        plt.axhline(y=np.mean(metrics['measured_delay']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['measured_delay']):.2f} ms")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{condition['name'].lower()}_delay_comparison.png"))
    plt.close()
    
    # Generate loss comparison
    plt.figure(figsize=(12, 5))
    
    # Controlled Loss
    plt.subplot(1, 2, 1)
    plt.title(f"Controlled Packet Loss - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Packet Loss (%)")
    if metrics['controlled_loss']:
        plt.plot(metrics['controlled_loss'], 'b-o', label="Controlled")
        plt.axhline(y=np.mean(metrics['controlled_loss']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['controlled_loss']):.2f} %")
    plt.grid(True)
    plt.legend()
    
    # Measured Loss
    plt.subplot(1, 2, 2)
    plt.title(f"Measured Frame Drop Rate - {condition['name']}")
    plt.xlabel("Sample")
    plt.ylabel("Frame Drop Rate (%)")
    if metrics['measured_loss']:
        plt.plot(metrics['measured_loss'], 'g-o', label="Measured")
        plt.axhline(y=np.mean(metrics['measured_loss']), color='r', linestyle='--', label=f"Mean: {np.mean(metrics['measured_loss']):.2f} %")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{condition['name'].lower()}_loss_comparison.png"))
    plt.close()

def generate_report():
    """Generate a report with commands and graphs."""
    # Create directory if it doesn't exist
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
    
    # Create HTML report
    report_file = os.path.join(GRAPHS_DIR, "comparison_report.html")
    
    with open(report_file, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Controlled vs Measured Metrics Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .command {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-left: 4px solid #007bff;
                    font-family: monospace;
                    margin-bottom: 20px;
                }}
                .metrics-summary {{
                    margin: 20px 0;
                }}
                .metrics-summary table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                .metrics-summary th {{
                    background-color: #f0f0f0;
                    text-align: left;
                }}
                .metrics-summary td, .metrics-summary th {{
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                .graph-container {{
                    margin-top: 20px;
                    margin-bottom: 40px;
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
                    border-left: 4px solid #28a745;
                    margin-bottom: 20px;
                }}
                hr {{
                    margin: 30px 0;
                    border: 0;
                    border-top: 1px solid #eee;
                }}
                code {{
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Controlled vs Measured Metrics Comparison</h1>
                <div class="note">
                    <p>This report compares the controlled network parameters (set using tc) with the measured metrics.</p>
                    <p>For each network condition, you can see:</p>
                    <ul>
                        <li>The tc command used to set the network condition</li>
                        <li>Graphs comparing controlled and measured bandwidth</li>
                        <li>Graphs comparing controlled and measured delay</li>
                        <li>Graphs comparing controlled and measured packet loss</li>
                    </ul>
                </div>
        """)
        
        # Load test results
        results = load_test_results()
        
        if not results:
            f.write("<p>No test results found.</p>")
            f.write("</div></body></html>")
            return
        
        # Process each network condition
        for condition in NETWORK_CONDITIONS:
            # Filter results for this condition
            condition_results = filter_results_by_condition(results, condition['name'])
            
            if not condition_results:
                continue
            
            # Extract metrics
            metrics = extract_metrics(condition_results)
            
            # Generate graphs
            generate_comparison_graphs(condition, metrics)
            
            # Calculate average values for controlled and measured metrics
            avg_controlled_bw = np.mean(metrics['controlled_bandwidth']) if metrics['controlled_bandwidth'] else 0
            avg_measured_bw = np.mean(metrics['measured_bandwidth']) if metrics['measured_bandwidth'] else 0
            
            avg_controlled_delay = np.mean(metrics['controlled_delay']) if metrics['controlled_delay'] else 0
            avg_measured_delay = np.mean(metrics['measured_delay']) if metrics['measured_delay'] else 0
            
            avg_controlled_loss = np.mean(metrics['controlled_loss']) if metrics['controlled_loss'] else 0
            avg_measured_loss = np.mean(metrics['measured_loss']) if metrics['measured_loss'] else 0
            
            # Add to report
            f.write(f"""
                <h2>{condition['name']} Network Condition</h2>
                
                <div class="command">
                    <h3>Command Used:</h3>
                    <code>{get_tc_command(condition)}</code>
                </div>
                
                <div class="metrics-summary">
                    <h3>Metrics Summary:</h3>
                    <table border="1" cellpadding="5" cellspacing="0">
                        <tr>
                            <th>Metric</th>
                            <th>Controlled Value</th>
                            <th>Measured Value</th>
                            <th>Difference</th>
                        </tr>
                        <tr>
                            <td>Bandwidth</td>
                            <td>{avg_controlled_bw:.2f} Mbps</td>
                            <td>{avg_measured_bw:.2f} Mbps</td>
                            <td>{(avg_measured_bw - avg_controlled_bw):.2f} Mbps</td>
                        </tr>
                        <tr>
                            <td>Delay</td>
                            <td>{avg_controlled_delay:.2f} ms</td>
                            <td>{avg_measured_delay:.2f} ms</td>
                            <td>{(avg_measured_delay - avg_controlled_delay):.2f} ms</td>
                        </tr>
                        <tr>
                            <td>Packet Loss</td>
                            <td>{avg_controlled_loss:.2f}%</td>
                            <td>{avg_measured_loss:.2f}%</td>
                            <td>{(avg_measured_loss - avg_controlled_loss):.2f}%</td>
                        </tr>
                    </table>
                </div>
                
                <h3>Bandwidth Comparison</h3>
                <p>Command: <code>rate {condition['rate']}</code> | Controlled: <b>{avg_controlled_bw:.2f} Mbps</b> | Measured: <b>{avg_measured_bw:.2f} Mbps</b></p>
                <div class="graph-container">
                    <img src="{condition['name'].lower()}_bandwidth_comparison.png" alt="Bandwidth Comparison">
                </div>
                
                <h3>Delay Comparison</h3>
                <p>Command: <code>delay {condition['delay']}</code> | Controlled: <b>{avg_controlled_delay:.2f} ms</b> | Measured: <b>{avg_measured_delay:.2f} ms</b></p>
                <div class="graph-container">
                    <img src="{condition['name'].lower()}_delay_comparison.png" alt="Delay Comparison">
                </div>
                
                <h3>Packet Loss Comparison</h3>
                <p>Command: <code>loss {condition['loss']}</code> | Controlled: <b>{avg_controlled_loss:.2f}%</b> | Measured: <b>{avg_measured_loss:.2f}%</b></p>
                <div class="graph-container">
                    <img src="{condition['name'].lower()}_loss_comparison.png" alt="Loss Comparison">
                </div>
                
                <hr>
            """)
        
        f.write("</div></body></html>")
    
    print(f"Report generated: {report_file}")
    print(f"Open this file in a browser to view the report.")

def main():
    """Main function."""
    print("Generating Controlled vs Measured Metrics Comparison")
    print("===================================================")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Generate report
    generate_report()

if __name__ == "__main__":
    main()