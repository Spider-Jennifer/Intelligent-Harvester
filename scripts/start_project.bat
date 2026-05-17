@echo off
chcp 65001 >nul

echo ========================================
echo 🚀 YOLOv8苹果检测项目 - 一键启动
echo ========================================
echo.
echo 请选择运行模式：
echo.
echo [1] 优化版Web应用 (推荐)
echo     特点：解决摄像头延迟问题，Web界面
echo     端口：8600
echo     访问：http://localhost:8600
echo.
echo [2] 最小延迟版本
echo     特点：最低延迟，纯OpenCV界面
echo     无Web界面，最高性能
echo.
echo [3] 原始Web应用
echo     特点：原始项目界面
echo     端口：8520
echo     访问：http://localhost:8520
echo.
echo [4] 退出
echo.
set /p choice="请选择 (1-4): "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" goto exit
echo 无效选择，请重新运行
pause
exit

:option1
echo.
echo 🚀 正在启动优化版Web应用...
echo 端口：8600
echo 访问地址：http://localhost:8600
echo 按 Ctrl+C 停止应用
echo.
timeout /t 2 /nobreak >nul
start "" http://localhost:8600
python -m streamlit run fast_camera_app.py --server.port 8600 --server.headless true
goto exit

:option2
echo.
echo 🚀 正在启动最小延迟版本...
echo 特点：纯OpenCV，最低延迟
echo 按 Q 或 ESC 退出
echo 按 C 切换置信度，S 切换跳帧
echo.
timeout /t 2 /nobreak >nul
python minimal_camera.py
goto exit

:option3
echo.
echo 🚀 正在启动原始Web应用...
echo 端口：8520
echo 访问地址：http://localhost:8520
echo 按 Ctrl+C 停止应用
echo.
timeout /t 2 /nobreak >nul
start "" http://localhost:8520
cd /d "YOLO-v8-app\YOLOv8-app-master"
python -m streamlit run app.py --server.port 8520 --server.headless true
goto exit

:exit
echo.
echo 程序已结束
pause