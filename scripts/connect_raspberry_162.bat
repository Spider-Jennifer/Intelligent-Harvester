@echo off
echo Connecting to Raspberry Pi at 192.168.45.162...
echo.
echo SSH Command:
echo ssh pi@192.168.45.162
echo.
echo Password: raspberry
echo.
echo After connecting, run these commands:
echo.
echo mkdir -p ~/yolov8_remote
echo cd ~/yolov8_remote
echo wget http://192.168.45.217:8000/raspberry_client_ultra_light.py
echo wget http://192.168.45.217:8000/install_raspberry_pi.sh
echo wget http://192.168.45.217:8000/start_detection.sh
echo chmod +x install_raspberry_pi.sh start_detection.sh
echo sudo ./install_raspberry_pi.sh
echo python3 raspberry_client_ultra_light.py
echo.
echo Press any key to open SSH connection...
pause > nul
ssh pi@192.168.45.162