#!/usr/bin/env python3
"""
QoS Controller for PC-to-PC video streaming.
This script provides a web interface for controlling Quality of Service parameters.
"""
import argparse
import json
import logging
import os
import subprocess
import threading
import time
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qos_controller.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default configuration (can be overridden via command line arguments)
DEFAULT_CONFIG = {
    'port': 5002,
    'sender_ip': '127.0.0.1',
    'sender_port': 5000,
    'receiver_ip': '127.0.0.1',
    'receiver_port': 5001,
    'use_tc': False,  # Whether to use Linux Traffic Control (tc)
    'interface': 'eth0',  # Network interface for tc
    'auto_open_browser': True,  # Whether to automatically open the browser
}

# Current configuration (will be updated by command line args)
config = DEFAULT_CONFIG.copy()

# Current QoS parameters
qos_params = {
    'bandwidth': 0,  # 0 means unlimited, otherwise in kbps
    'delay': 0,  # in milliseconds
    'jitter': 0,  # in milliseconds
    'packet_loss': 0,  # in percentage (0-100)
    'last_update': time.time(),
}

# Presets for common network conditions
NETWORK_PRESETS = {
    'perfect': {
        'name': 'Perfect Connection',
        'bandwidth': 0,  # unlimited
        'delay': 0,
        'jitter': 0,
        'packet_loss': 0
    },
    'good': {
        'name': 'Good Broadband',
        'bandwidth': 10000,  # 10 Mbps
        'delay': 20,
        'jitter': 5,
        'packet_loss': 0.1
    },
    'average': {
        'name': 'Average Broadband',
        'bandwidth': 5000,  # 5 Mbps
        'delay': 50,
        'jitter': 10,
        'packet_loss': 0.5
    },
    'mobile': {
        'name': 'Mobile 4G',
        'bandwidth': 2000,  # 2 Mbps
        'delay': 100,
        'jitter': 20,
        'packet_loss': 1
    },
    'poor': {
        'name': 'Poor Connection',
        'bandwidth': 500,  # 500 kbps
        'delay': 200,
        'jitter': 50,
        'packet_loss': 5
    },
    'satellite': {
        'name': 'Satellite',
        'bandwidth': 1000,  # 1 Mbps
        'delay': 500,
        'jitter': 100,
        'packet_loss': 2
    },
    'terrible': {
        'name': 'Terrible Connection',
        'bandwidth': 100,  # 100 kbps
        'delay': 1000,
        'jitter': 200,
        'packet_loss': 10
    }
}

def apply_tc_qos(params):
    """Apply QoS parameters using Linux Traffic Control (tc)."""
    if not config['use_tc']:
        logger.info("TC is disabled, not applying QoS parameters")
        return False
    
    interface = config['interface']
    
    # Check if the interface exists
    try:
        result = subprocess.run(
            f"ip link show {interface}",
            shell=True, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        logger.error(f"Network interface {interface} does not exist")
        return False
    
    # Remove any existing tc rules
    try:
        subprocess.run(
            f"sudo tc qdisc del dev {interface} root",
            shell=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError:
        # It's okay if this fails (e.g., if there are no existing rules)
        pass
    
    # If all parameters are zero, we're done (no rules needed)
    if params['bandwidth'] == 0 and params['delay'] == 0 and params['jitter'] == 0 and params['packet_loss'] == 0:
        logger.info("All QoS parameters are zero, no tc rules applied")
        return True
    
    # Build the tc command
    cmd = f"sudo tc qdisc add dev {interface} root netem"
    
    if params['delay'] > 0:
        cmd += f" delay {params['delay']}ms"
        if params['jitter'] > 0:
            cmd += f" {params['jitter']}ms distribution normal"
    
    if params['packet_loss'] > 0:
        cmd += f" loss {params['packet_loss']}%"
    
    if params['bandwidth'] > 0:
        # For bandwidth limiting, we need to use tbf (token bucket filter)
        # First, apply the netem rules
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply netem rules: {e}")
            return False
        
        # Then, apply the bandwidth limit
        # Convert kbps to bytes per second (for burst and limit)
        bytes_per_sec = params['bandwidth'] * 125  # 1 kbps = 125 bytes/sec
        
        # Apply tbf for bandwidth limiting
        tbf_cmd = (
            f"sudo tc qdisc add dev {interface} parent 1:1 handle 10: tbf "
            f"rate {params['bandwidth']}kbit burst {bytes_per_sec} limit {bytes_per_sec*2}"
        )
        try:
            subprocess.run(tbf_cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply tbf rules: {e}")
            return False
    else:
        # Just apply the netem rules
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply tc rules: {e}")
            return False
    
    logger.info(f"Applied tc QoS parameters: {params}")
    return True

def apply_sender_qos(params):
    """Apply QoS parameters to the sender."""
    try:
        # Convert parameters for the sender API
        sender_params = {
            'bandwidth_limit': params['bandwidth'] * 125 if params['bandwidth'] > 0 else 0,  # Convert kbps to bytes/sec
            'artificial_delay': params['delay'] / 1000.0,  # Convert ms to seconds
            'packet_loss': params['packet_loss']
        }
        
        # Send to the sender API
        response = requests.post(
            f"http://{config['sender_ip']}:{config['sender_port']}/config",
            json=sender_params,
            timeout=5
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to apply QoS to sender: {response.status_code} {response.text}")
            return False
        
        logger.info(f"Applied sender QoS parameters: {sender_params}")
        return True
    except requests.RequestException as e:
        logger.error(f"Error communicating with sender: {e}")
        return False

def apply_qos_parameters(params):
    """Apply QoS parameters to both tc (if enabled) and the sender."""
    success = True
    
    # Apply to tc if enabled
    if config['use_tc']:
        tc_success = apply_tc_qos(params)
        if not tc_success:
            logger.warning("Failed to apply tc QoS parameters")
            success = False
    
    # Always try to apply to the sender
    sender_success = apply_sender_qos(params)
    if not sender_success:
        logger.warning("Failed to apply sender QoS parameters")
        success = False
    
    return success

# API Routes

@app.route('/')
def home():
    """Redirect to the control panel."""
    return send_from_directory('.', 'control_panel.html')

@app.route('/control_panel.html')
def control_panel():
    """Serve the control panel HTML."""
    return send_from_directory('.', 'control_panel.html')

@app.route('/qos', methods=['GET', 'POST'])
def api_qos():
    """Get or update QoS parameters."""
    global qos_params
    
    if request.method == 'GET':
        return jsonify(qos_params)
    
    # Handle POST to update QoS parameters
    try:
        new_params = request.json
        if not new_params:
            return jsonify({'status': 'error', 'message': 'No parameters provided'}), 400
        
        # Update parameters with new values
        for key, value in new_params.items():
            if key in qos_params and key != 'last_update':
                # Convert to appropriate type and validate
                if key in ['bandwidth', 'delay', 'jitter']:
                    value = max(0, int(value))
                elif key == 'packet_loss':
                    value = max(0, min(100, float(value)))
                
                qos_params[key] = value
        
        # Update timestamp
        qos_params['last_update'] = time.time()
        
        # Apply the parameters
        success = apply_qos_parameters(qos_params)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'QoS parameters updated',
                'parameters': qos_params
            })
        else:
            return jsonify({
                'status': 'partial',
                'message': 'Some QoS parameters could not be applied',
                'parameters': qos_params
            }), 207
    except Exception as e:
        logger.error(f"Error updating QoS parameters: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

@app.route('/preset/<preset_name>', methods=['GET'])
def api_preset(preset_name):
    """Apply a preset network condition."""
    global qos_params
    
    if preset_name not in NETWORK_PRESETS:
        return jsonify({
            'status': 'error',
            'message': f'Unknown preset: {preset_name}',
            'available_presets': list(NETWORK_PRESETS.keys())
        }), 400
    
    # Get the preset parameters
    preset = NETWORK_PRESETS[preset_name]
    
    # Update QoS parameters
    for key, value in preset.items():
        if key in qos_params and key != 'name':
            qos_params[key] = value
    
    # Update timestamp
    qos_params['last_update'] = time.time()
    
    # Apply the parameters
    success = apply_qos_parameters(qos_params)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Applied preset: {preset["name"]}',
            'parameters': qos_params
        })
    else:
        return jsonify({
            'status': 'partial',
            'message': f'Preset {preset["name"]} could not be fully applied',
            'parameters': qos_params
        }), 207

@app.route('/presets', methods=['GET'])
def api_presets():
    """Get all available presets."""
    return jsonify(NETWORK_PRESETS)

@app.route('/status', methods=['GET'])
def api_status():
    """Get status of sender and receiver."""
    status = {
        'qos_controller': {
            'status': 'running',
            'config': config,
            'qos_params': qos_params
        },
        'sender': {
            'status': 'unknown',
            'details': None
        },
        'receiver': {
            'status': 'unknown',
            'details': None
        }
    }
    
    # Check sender status
    try:
        response = requests.get(
            f"http://{config['sender_ip']}:{config['sender_port']}/status",
            timeout=2
        )
        if response.status_code == 200:
            status['sender'] = {
                'status': 'running',
                'details': response.json()
            }
    except requests.RequestException:
        status['sender'] = {
            'status': 'not_running',
            'details': None
        }
    
    # Check receiver status
    try:
        response = requests.get(
            f"http://{config['receiver_ip']}:{config['receiver_port']}/stats",
            timeout=2
        )
        if response.status_code == 200:
            status['receiver'] = {
                'status': 'running',
                'details': response.json()
            }
    except requests.RequestException:
        status['receiver'] = {
            'status': 'not_running',
            'details': None
        }
    
    return jsonify(status)

@app.route('/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration."""
    global config
    
    if request.method == 'GET':
        return jsonify(config)
    
    # Handle POST to update config
    try:
        new_config = request.json
        if not new_config:
            return jsonify({'status': 'error', 'message': 'No configuration provided'}), 400
        
        # Update config with new values
        for key, value in new_config.items():
            if key in config:
                config[key] = value
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Ignoring unknown config parameter: {key}")
        
        return jsonify({'status': 'success', 'message': 'Configuration updated', 'config': config})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def parse_arguments():
    """Parse command line arguments and update configuration."""
    global config
    
    parser = argparse.ArgumentParser(description='QoS Controller for PC-to-PC Video Streaming')
    
    parser.add_argument('--port', type=int, help='Port to listen on')
    parser.add_argument('--sender-ip', type=str, help='IP address of the sender')
    parser.add_argument('--sender-port', type=int, help='Port of the sender')
    parser.add_argument('--receiver-ip', type=str, help='IP address of the receiver')
    parser.add_argument('--receiver-port', type=int, help='Port of the receiver')
    parser.add_argument('--use-tc', action='store_true', help='Use Linux Traffic Control (tc)')
    parser.add_argument('--interface', type=str, help='Network interface for tc')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.port:
        config['port'] = args.port
    
    if args.sender_ip:
        config['sender_ip'] = args.sender_ip
    
    if args.sender_port:
        config['sender_port'] = args.sender_port
    
    if args.receiver_ip:
        config['receiver_ip'] = args.receiver_ip
    
    if args.receiver_port:
        config['receiver_port'] = args.receiver_port
    
    if args.use_tc:
        config['use_tc'] = True
    
    if args.interface:
        config['interface'] = args.interface
    
    if args.no_browser:
        config['auto_open_browser'] = False
    
    return config

if __name__ == '__main__':
    # Parse command line arguments
    config = parse_arguments()
    
    # Log the configuration
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Open the control panel in the browser if auto_open_browser is enabled
    if config['auto_open_browser']:
        url = f"http://localhost:{config['port']}/control_panel.html"
        logger.info(f"Opening control panel in browser: {url}")
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    
    # Start the Flask app
    logger.info(f"Starting QoS controller on port {config['port']}")
    app.run(host='0.0.0.0', port=config['port'], debug=False, threaded=True)