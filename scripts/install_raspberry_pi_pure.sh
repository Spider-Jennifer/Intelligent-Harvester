#!/bin/bash

echo "========================================"
echo "Raspberry Pi 4B Apple Detection - Dependencies"
echo "========================================"
echo

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-opencv

echo "Installing Python packages..."
pip3 install --no-cache-dir numpy ultralytics

echo "Creating startup script..."
cat > start_detection.sh << 'END'
#!/bin/bash

echo "Starting Raspberry Pi Apple Detection..."
echo "Press Ctrl+C to stop"

v4l2-ctl --set-fmt-video=width=320,height=240,pixelformat=YUYV
v4l2-ctl --set-ctrl=brightness=50
v4l2-ctl --set-ctrl=contrast=10

python3 raspberry_client_ultra_light.py
END

chmod +x start_detection.sh

echo
echo "Installation complete!"
echo "Run the following command to start detection:"
echo "  ./start_detection.sh"
echo
echo "Or double-click to run:"
echo "  run_ultra_light.bat (Windows)"
echo "  start_detection.sh (Linux/Raspberry Pi)"