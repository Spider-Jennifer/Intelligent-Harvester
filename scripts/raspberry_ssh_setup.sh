#!/bin/bash
# Raspberry Pi setup script for SSH connection

echo "========================================"
echo "Raspberry Pi Remote Detection Setup"
echo "========================================"
echo

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This script should run on Raspberry Pi"
fi

# Get Raspberry Pi IP
RPI_IP=$(hostname -I | awk '{print $1}')
echo "Raspberry Pi IP: $RPI_IP"
echo "Computer Server IP: 192.168.45.217"
echo

# Create directory
echo "Creating project directory..."
mkdir -p ~/yolov8_remote
cd ~/yolov8_remote
echo "Current directory: $(pwd)"
echo

# Download files
echo "Downloading files from computer server..."
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget http://192.168.45.217:8000/raspberry_client_ultra_light.py
    wget http://192.168.45.217:8000/install_raspberry_pi.sh
    wget http://192.168.45.217:8000/start_detection.sh
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -O http://192.168.45.217:8000/raspberry_client_ultra_light.py
    curl -O http://192.168.45.217:8000/install_raspberry_pi.sh
    curl -O http://192.168.45.217:8000/start_detection.sh
else
    echo "Error: wget or curl not found"
    echo "Install with: sudo apt-get install wget curl"
    exit 1
fi

# Make scripts executable
chmod +x install_raspberry_pi.sh
chmod +x start_detection.sh

echo
echo "Files downloaded successfully!"
echo "List of files:"
ls -la
echo

# Ask user if they want to install dependencies
read -p "Do you want to install dependencies now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    sudo ./install_raspberry_pi.sh
    
    echo
    read -p "Do you want to start detection now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting detection..."
        python3 raspberry_client_ultra_light.py
    else
        echo "You can start detection later with:"
        echo "  cd ~/yolov8_remote"
        echo "  python3 raspberry_client_ultra_light.py"
    fi
else
    echo "You can install dependencies later with:"
    echo "  cd ~/yolov8_remote"
    echo "  sudo ./install_raspberry_pi.sh"
fi

echo
echo "Setup complete!"