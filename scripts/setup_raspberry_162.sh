#!/bin/bash
# Raspberry Pi Setup Script for IP 192.168.45.162
# Computer Server IP: 192.168.45.217

echo "========================================"
echo "Raspberry Pi Setup (IP: 192.168.45.162)"
echo "Computer Server: 192.168.45.217:8000"
echo "========================================"
echo

# Step 1: Create directory
echo "Step 1: Creating project directory..."
mkdir -p ~/yolov8_remote
cd ~/yolov8_remote
echo "Directory: $(pwd)"
echo

# Step 2: Download files
echo "Step 2: Downloading files..."
echo "Downloading raspberry_client_ultra_light.py..."
wget http://192.168.45.217:8000/raspberry_client_ultra_light.py -O raspberry_client_ultra_light.py

echo "Downloading install_raspberry_pi.sh..."
wget http://192.168.45.217:8000/install_raspberry_pi.sh -O install_raspberry_pi.sh

echo "Downloading start_detection.sh..."
wget http://192.168.45.217:8000/start_detection.sh -O start_detection.sh
echo

# Step 3: Make scripts executable
echo "Step 3: Making scripts executable..."
chmod +x install_raspberry_pi.sh
chmod +x start_detection.sh
echo "Permissions set."
echo

# Step 4: List files
echo "Step 4: Files downloaded:"
ls -la
echo

# Step 5: Install dependencies
echo "Step 5: Installing dependencies..."
echo "This may take a few minutes..."
sudo ./install_raspberry_pi.sh
echo

# Step 6: Start detection
echo "Step 6: Starting detection..."
echo "Press Ctrl+C to stop"
python3 raspberry_client_ultra_light.py