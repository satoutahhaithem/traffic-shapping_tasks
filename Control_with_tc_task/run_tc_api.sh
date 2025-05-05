#!/bin/bash

# This script runs tc_api.py with sudo while preserving the virtual environment

# Get the path to the virtual environment's Python
VENV_PYTHON=$(which python3)

# Run the tc_api.py script with sudo using the virtual environment's Python
sudo $VENV_PYTHON tc_api.py