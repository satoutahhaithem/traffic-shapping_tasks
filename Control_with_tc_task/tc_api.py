#!/usr/bin/env python3
"""
Simple API server to execute tc commands from the web interface
"""
from flask import Flask, request, jsonify
import subprocess
import os
import re
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('tc_api.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Network interface to use - auto-detect
def get_default_interface():
    try:
        # Try to get the default interface using ip route
        result = subprocess.run("ip route | grep default | awk '{print $5}' | head -n 1",
                               shell=True, capture_output=True, text=True, check=True)
        interface = result.stdout.strip()
        if interface:
            logger.info(f"Auto-detected default interface: {interface}")
            return interface
    except Exception as e:
        logger.error(f"Error auto-detecting interface: {str(e)}")
    
    # Fallback interfaces to try
    fallbacks = ["eth0", "wlan0", "ens33", "enp0s3", "wlp0s20f3"]
    for interface in fallbacks:
        try:
            # Check if interface exists
            result = subprocess.run(f"ip link show {interface}",
                                  shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Using fallback interface: {interface}")
                return interface
        except:
            pass
    
    # Last resort fallback
    logger.warning("Could not detect interface, using lo (loopback)")
    return "lo"

INTERFACE = get_default_interface()
logger.info(f"Using network interface: {INTERFACE}")

@app.route('/execute_tc_command', methods=['POST'])
def execute_tc_command():
    """Execute a tc command"""
    logger.info(f"Received execute_tc_command request from {request.remote_addr}")
    logger.debug(f"Request headers: {request.headers}")
    
    data = request.json
    
    if not data or 'command' not in data:
        logger.error("No command provided in request")
        return jsonify({'error': 'No command provided'}), 400
    
    command = data['command']
    logger.info(f"Received command: {command}")
    
    # Validate the command to ensure it's a tc command
    if not command.startswith('tc '):
        logger.error(f"Invalid command (not a tc command): {command}")
        return jsonify({'error': 'Only tc commands are allowed'}), 400
    
    # Extract parameters from the command
    delay_match = re.search(r'delay\s+(\d+)ms', command)
    rate_match = re.search(r'rate\s+(\d+\.?\d*)(\w+)', command)
    loss_match = re.search(r'loss\s+(\d+\.?\d*)%', command)
    
    delay = delay_match.group(1) if delay_match else "0"
    rate = rate_match.group(1) if rate_match else "10"
    rate_unit = rate_match.group(2) if rate_match else "mbit"
    loss = loss_match.group(1) if loss_match else "0"
    
    logger.info(f"Extracted parameters: delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%")
    
    # Validate parameters
    try:
        # Convert values but don't enforce strict limits
        delay = int(delay) if delay else 0
        rate = float(rate) if rate else 10
        loss = float(loss) if loss else 0
        
        # Just ensure values are not negative
        if delay < 0:
            logger.warning(f"Negative delay value ({delay}) corrected to 0")
            delay = 0
        
        if rate <= 0:
            logger.warning(f"Non-positive rate value ({rate}) corrected to 0.1")
            rate = 0.1
        
        if loss < 0:
            logger.warning(f"Negative loss value ({loss}) corrected to 0")
            loss = 0
        
        if rate_unit not in ['kbit', 'mbit', 'gbit']:
            logger.warning(f"Invalid rate unit ({rate_unit}) corrected to mbit")
            rate_unit = 'mbit'
        
        logger.info(f"Validated parameters: delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%")
    
    except ValueError as e:
        logger.error(f"Parameter validation error: {str(e)}")
        return jsonify({'error': f'Invalid parameter values: {str(e)}'}), 400
    
    # Check if the qdisc exists
    check_cmd = f"tc qdisc show dev {INTERFACE} | grep netem"
    logger.debug(f"Executing check command: {check_cmd}")
    
    try:
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        logger.debug(f"Check command result: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
        
        # If the qdisc doesn't exist, add it
        if result.returncode != 0:
            add_cmd = f"sudo tc qdisc add dev {INTERFACE} root netem delay {delay}ms rate {rate}{rate_unit} loss {loss}%"
            logger.info(f"Adding new qdisc with command: {add_cmd}")
            
            add_result = subprocess.run(add_cmd, shell=True, capture_output=True, text=True)
            if add_result.returncode != 0:
                logger.error(f"Add command failed: {add_result.stderr}")
                return jsonify({
                    'error': f'Add command failed: {add_result.stderr}',
                    'command': add_cmd,
                    'success': False
                }), 500
            
            logger.info(f"Successfully added netem qdisc with delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%")
            
            # Verify the addition was applied
            verify_cmd = f"tc -s qdisc show dev {INTERFACE}"
            verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
            logger.debug(f"Verification result after add: {verify_result.stdout}")
            
            return jsonify({
                'message': f'Added netem qdisc with delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%',
                'command': add_cmd,
                'verification': verify_result.stdout,
                'success': True
            })
        
        # If the qdisc exists, change it
        change_cmd = f"sudo tc qdisc change dev {INTERFACE} root netem delay {delay}ms rate {rate}{rate_unit} loss {loss}%"
        logger.info(f"Changing existing qdisc with command: {change_cmd}")
        
        change_result = subprocess.run(change_cmd, shell=True, capture_output=True, text=True)
        if change_result.returncode != 0:
            logger.error(f"Change command failed: {change_result.stderr}")
            return jsonify({
                'error': f'Change command failed: {change_result.stderr}',
                'command': change_cmd,
                'success': False
            }), 500
        
        # Verify the change was applied
        verify_cmd = f"tc -s qdisc show dev {INTERFACE}"
        verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
        logger.debug(f"Verification result after change: {verify_result.stdout}")
        
        logger.info(f"Successfully changed netem qdisc to delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%")
        return jsonify({
            'message': f'Changed netem qdisc to delay={delay}ms, rate={rate}{rate_unit}, loss={loss}%',
            'command': change_cmd,
            'verification': verify_result.stdout,
            'success': True
        })
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {str(e)}")
        return jsonify({
            'error': f'Command failed: {str(e)}',
            'details': str(e),
            'success': False
        }), 500

@app.route('/reset_tc', methods=['POST'])
def reset_tc():
    """Reset tc configuration"""
    reset_cmd = f"sudo tc qdisc del dev {INTERFACE} root"
    logger.info(f"Resetting tc configuration with command: {reset_cmd}")
    
    try:
        reset_result = subprocess.run(reset_cmd, shell=True, capture_output=True, text=True)
        
        if reset_result.returncode != 0:
            logger.error(f"Reset command failed: {reset_result.stderr}")
            return jsonify({
                'error': f'Reset command failed: {reset_result.stderr}',
                'command': reset_cmd,
                'success': False
            }), 500
        
        logger.info("Successfully reset tc configuration")
        return jsonify({
            'message': 'Reset tc configuration',
            'command': reset_cmd,
            'success': True
        })
    except subprocess.CalledProcessError as e:
        logger.error(f"Reset command failed with exception: {str(e)}")
        return jsonify({
            'error': f'Command failed: {str(e)}',
            'command': reset_cmd,
            'success': False
        }), 500

@app.route('/get_tc_status', methods=['GET'])
def get_tc_status():
    """Get current tc status"""
    status_cmd = f"tc -s qdisc show dev {INTERFACE}"
    logger.info(f"Getting tc status with command: {status_cmd}")
    
    try:
        status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
        
        if status_result.returncode != 0:
            logger.error(f"Status command failed: {status_result.stderr}")
            return jsonify({
                'error': f'Status command failed: {status_result.stderr}',
                'command': status_cmd,
                'success': False
            }), 500
        
        logger.info(f"Successfully retrieved tc status: {status_result.stdout}")
        return jsonify({
            'status': status_result.stdout,
            'command': status_cmd,
            'success': True
        })
    except subprocess.CalledProcessError as e:
        logger.error(f"Status command failed with exception: {str(e)}")
        return jsonify({
            'error': f'Command failed: {str(e)}',
            'command': status_cmd,
            'success': False
        }), 500

if __name__ == '__main__':
    print(f"Starting tc API server on port 5002...")
    logger.info(f"Using network interface: {INTERFACE}")
    logger.info(f"Starting tc API server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)  # Enable debug mode for more detailed error messages