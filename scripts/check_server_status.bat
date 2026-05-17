@echo off
echo Checking HTTP Server Status...
echo ==============================
echo.

echo 1. Server process:
tasklist | findstr python
echo.

echo 2. Port 8000 status:
netstat -an | findstr :8000
echo.

echo 3. Test file access:
curl -I http://localhost:8000/raspberry_client_ultra_light.py
echo.

echo 4. Current directory files:
dir raspberry_client_ultra_light.py install_raspberry_pi.sh start_detection.sh 2>nul
echo.

echo 5. If Raspberry Pi is stuck, try these troubleshooting steps:
echo    - Check if wget/curl is installed on Raspberry Pi
echo    - Check network connectivity: ping 192.168.45.217 from Raspberry Pi
echo    - Check firewall on computer allows port 8000
echo    - Try downloading manually with smaller steps
echo.

echo Press any key to exit...
pause > nul