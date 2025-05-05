#!/usr/bin/env python3
"""
Simple server to execute TC commands from the tc_buttons.html page
"""
from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the tc_buttons.html page"""
    return send_from_directory('.', 'tc_buttons.html')

@app.route('/execute_command', methods=['POST'])
def execute_command():
    """Execute the apply_tc_both.sh script with the specified parameters"""
    data = request.json
    
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided'}), 400
    
    command = data['command']
    
    # Validate the command to ensure it's only calling apply_tc_both.sh
    if not command.startswith('./apply_tc_both.sh'):
        return jsonify({'error': 'Only apply_tc_both.sh commands are allowed'}), 400
    
    try:
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({
                'error': f'Command failed: {result.stderr}',
                'command': command,
                'success': False
            }), 500
        
        return jsonify({
            'message': f'Command executed successfully: {command}',
            'output': result.stdout,
            'success': True
        })
    
    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': f'Command failed: {str(e)}',
            'details': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    print(f"Starting TC Button Server on port 8080...")
    print(f"Access the interface at http://localhost:8080/")
    app.run(host='0.0.0.0', port=8080, debug=True)