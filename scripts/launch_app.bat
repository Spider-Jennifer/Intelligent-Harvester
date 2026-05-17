@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测 Web 应用
echo ========================================

echo.
echo 正在启动应用...
echo 应用启动后，请访问以下网址：
echo.
echo http://localhost:8501
echo.
echo 按 Ctrl+C 停止应用
echo.

cd /d "%~dp0YOLO-v8-app\YOLOv8-app-master"
"C:\Users\李晨鑫\AppData\Roaming\Python\Python39\Scripts\streamlit.exe" run app.py --server.port 8501

pause