@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测项目 - 立即运行
echo ========================================

echo.
echo [1] 清理旧进程...
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2] 启动Web应用...
echo 端口: 8520
echo 网址: http://localhost:8520
echo.
echo 应用启动中，请稍候...
echo 如果浏览器没有自动打开，请手动访问上述网址
echo 按 Ctrl+C 停止应用
echo.

cd /d "%~dp0YOLO-v8-app\YOLOv8-app-master"
start "" http://localhost:8520
python -m streamlit run app.py --server.port 8520 --server.headless true

pause