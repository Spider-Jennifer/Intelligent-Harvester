#!/bin/bash
# Quick start script for Raspberry Pi
# Skip full installation and run directly

echo "Quick starting Raspberry Pi detection..."
echo "========================================"

# Check if already in directory
if [ ! -f "raspberry_client_ultra_light.py" ]; then
    echo "File not found in current directory"
    echo "Changing to ~/yolov8_remote..."
    cd ~/yolov8_remote
fi

# Check if Python packages are installed
echo "Checking Python packages..."
python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "OpenCV not found. Installing minimal packages..."
    sudo apt-get update
    sudo apt-get install -y python3-opencv
fi

python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Ultralytics not found. Installing..."
    pip3 install ultralytics numpy
fi

# Start detection
echo "Starting detection..."
echo "Press Ctrl+C to stop"
python3 raspberry_client_ultra_light.py