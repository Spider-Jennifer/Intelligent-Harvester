@echo off
chcp 65001 >nul
echo ========================================
echo 苹果摄像头检测 - 直接运行
echo ========================================
echo.
echo 访问网址: http://localhost:8600
echo.
echo 正在启动应用...
echo.

REM 清理旧进程
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

REM 直接运行
python -m streamlit run fast_camera_app.py --server.port 8600 --server.headless true