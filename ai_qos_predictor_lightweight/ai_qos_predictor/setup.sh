#!/bin/bash
# Setup script for AI QoS Predictor

# Print colored messages
print_message() {
    echo -e "\e[1;34m>> $1\e[0m"
}

print_success() {
    echo -e "\e[1;32m✓ $1\e[0m"
}

print_error() {
    echo -e "\e[1;31m✗ $1\e[0m"
}

print_message "Setting up AI QoS Predictor..."

# Check Python version
print_message "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. Found: Python $python_version"
    exit 1
else
    print_success "Python $python_version detected"
fi

# Create virtual environment
print_message "Creating virtual environment..."
if [ -d "venv" ]; then
    print_message "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    else
        print_success "Virtual environment created"
    fi
fi

# Activate virtual environment
print_message "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
else
    print_success "Virtual environment activated"
fi

# Install dependencies
print_message "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    exit 1
else
    print_success "Dependencies installed"
fi

# Create directory structure
print_message "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/features
mkdir -p models/saved
mkdir -p models/results
mkdir -p evaluation
if [ $? -ne 0 ]; then
    print_error "Failed to create directory structure"
    exit 1
else
    print_success "Directory structure created"
fi

# Make main.py executable
print_message "Making scripts executable..."
chmod +x main.py
chmod +x models/train_model.py
if [ $? -ne 0 ]; then
    print_error "Failed to make scripts executable"
    exit 1
else
    print_success "Scripts are now executable"
fi

print_message "Setup complete! You can now use the AI QoS Predictor."
print_message "To get started, run: ./main.py process-data --dataset 5g_kpi"
print_message "For more information, see the QUICK_START.md and USER_GUIDE.md files."

# Deactivate virtual environment
deactivate