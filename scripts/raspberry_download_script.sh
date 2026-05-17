#!/bin/bash
# Raspberry Pi download script
# Run this on Raspberry Pi to download all files

echo "Downloading files from computer server..."
echo "Computer IP: 192.168.45.217"

# Create directory
mkdir -p ~/yolov8_remote
cd ~/yolov8_remote

# Download files
echo "Downloading raspberry_client_ultra_light.py..."
wget http://192.168.45.217:8000/raspberry_client_ultra_light.py -O raspberry_client_ultra_light.py

echo "Downloading install_raspberry_pi.sh..."
wget http://192.168.45.217:8000/install_raspberry_pi.sh -O install_raspberry_pi.sh

echo "Downloading start_detection.sh..."
wget http://192.168.45.217:8000/start_detection.sh -O start_detection.sh

# Make scripts executable
chmod +x install_raspberry_pi.sh
chmod +x start_detection.sh

echo "Files downloaded successfully!"
echo "Run: sudo ./install_raspberry_pi.sh"