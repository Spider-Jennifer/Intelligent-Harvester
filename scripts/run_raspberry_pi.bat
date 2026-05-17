@echo off
chcp 65001 >nul

echo ========================================
echo 树莓派4B专用苹果检测 - 精简优化版
echo ========================================
echo.
echo 优化特点：
echo 1. 最低内存占用
echo 2. 最小推理尺寸 (160x120)
echo 3. 简化图像处理
echo 4. 移除所有调试打印
echo 5. 最低分辨率 (320x240)
echo.
echo 启动树莓派优化版摄像头检测...
echo 按 Q 或 ESC 退出
echo.

REM 清理旧进程
taskkill /f /im python.exe 2>nul
timeout /t 1 /nobreak >nul

REM 检查依赖
echo.
echo 检查依赖...
python -c "import cv2, numpy" 2>nul
if errorlevel 1 (
    echo 安装必要依赖...
    pip install opencv-python numpy -q
)

REM 检查ultralytics
python -c "import ultralytics" 2>nul
if errorlevel 1 (
    echo 安装ultralytics...
    pip install ultralytics -q
)

echo.
echo 启动检测程序...
echo.

REM 运行程序
python raspberry_pi_camera.py

pause