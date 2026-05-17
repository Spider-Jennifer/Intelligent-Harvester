@echo off
echo Testing connection to Raspberry Pi 192.168.45.162...
echo.

echo 1. Testing ping...
ping -n 3 192.168.45.162
echo.

echo 2. Testing HTTP server...
curl -I http://192.168.45.217:8000/
echo.

echo 3. SSH connection command:
echo    ssh pi@192.168.45.162
echo    Password: raspberry
echo.

echo 4. Quick setup command (copy to SSH):
echo    mkdir -p ~/yolov8_remote ^&^& cd ~/yolov8_remote ^&^& wget http://192.168.45.217:8000/raspberry_client_ultra_light.py ^&^& wget http://192.168.45.217:8000/install_raspberry_pi.sh ^&^& wget http://192.168.45.217:8000/start_detection.sh ^&^& chmod +x install_raspberry_pi.sh start_detection.sh ^&^& sudo ./install_raspberry_pi.sh ^&^& python3 raspberry_client_ultra_light.py
echo.

echo Press any key to exit...
pause > nul