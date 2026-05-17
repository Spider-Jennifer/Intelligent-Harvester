@echo off
chcp 65001 >nul
echo ========================================
echo 🍎 最小延迟摄像头苹果检测
echo ========================================
echo.
echo 特点：
echo 1. 纯OpenCV，无Streamlit开销
echo 2. 最低延迟，最高响应速度
echo 3. 实时性能监控
echo 4. 可调节参数
echo.
echo 控制说明：
echo   [Q] 或 [ESC] - 退出程序
echo   [C] - 切换置信度阈值
echo   [S] - 切换跳帧设置
echo   [R] - 重置性能统计
echo   [+] - 增加置信度
echo   [-] - 降低置信度
echo.

REM 清理旧进程
taskkill /f /im python.exe 2>nul
timeout /t 1 /nobreak >nul

REM 检查依赖
echo [1] 检查Python依赖...
python -c "import cv2; print('✅ OpenCV已安装')" 2>nul
if errorlevel 1 (
    echo ⚠️ 安装OpenCV...
    pip install opencv-python -q
)

python -c "import numpy; print('✅ NumPy已安装')" 2>nul
if errorlevel 1 (
    echo ⚠️ 安装NumPy...
    pip install numpy -q
)

REM 检查ultralytics
python -c "import ultralytics; print('✅ Ultralytics已安装')" 2>nul
if errorlevel 1 (
    echo ⚠️ 安装Ultralytics...
    pip install ultralytics -q
)

echo.
echo [2] 启动最小延迟摄像头检测...
echo 按 Q 或 ESC 键退出窗口
echo.

REM 运行最小延迟检测
python minimal_camera.py

echo.
echo 检测已结束
pause