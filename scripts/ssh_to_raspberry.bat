@echo off
echo SSH Connection to Raspberry Pi
echo ==============================
echo.
echo 1. Find Raspberry Pi IP address:
echo    On Raspberry Pi terminal: hostname -I
echo.
echo 2. Connect via SSH:
echo    ssh pi@RASPBERRY_IP
echo    Default password: raspberry
echo.
echo 3. Once connected, run these commands:
echo    mkdir -p ~/yolov8_remote
echo    cd ~/yolov8_remote
echo    wget http://192.168.45.217:8000/raspberry_client_ultra_light.py
echo    wget http://192.168.45.217:8000/install_raspberry_pi.sh
echo    chmod +x install_raspberry_pi.sh
echo    sudo ./install_raspberry_pi.sh
echo    python3 raspberry_client_ultra_light.py
echo.
echo Press any key to continue...
pause > nul