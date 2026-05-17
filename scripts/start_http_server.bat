@echo off
echo Starting HTTP Server on port 8000...
echo Files available for download:
echo 1. raspberry_client_ultra_light.py
echo 2. install_raspberry_pi.sh
echo 3. start_detection.sh
echo.
echo On Raspberry Pi, run:
echo wget http://192.168.45.217:8000/raspberry_client_ultra_light.py
echo wget http://192.168.45.217:8000/install_raspberry_pi.sh
echo.
echo Press Ctrl+C to stop server
python -m http.server 8000