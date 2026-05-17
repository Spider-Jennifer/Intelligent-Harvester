@echo off
chcp 65001 >nul
title YOLOv8 Web应用测试

echo ========================================
echo 🧪 YOLOv8 Web应用快速测试
echo ========================================
echo.

echo [1] 测试Python环境...
python --version
if errorlevel 1 (
    echo ❌ Python未找到
    pause
    exit /b 1
)

echo.
echo [2] 测试依赖包...
echo 正在测试必要的Python包...

echo.
echo 测试streamlit...
python -c "import streamlit; print('✅ Streamlit版本:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit未安装
)

echo.
echo 测试opencv...
python -c "import cv2; print('✅ OpenCV版本:', cv2.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ OpenCV未安装
)

echo.
echo 测试ultralytics...
python -c "import ultralytics; print('✅ Ultralytics版本:', ultralytics.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ Ultralytics未安装
)

echo.
echo [3] 测试模型文件...
if exist apple_sensitive.pt (
    echo ✅ 找到 apple_sensitive.pt (推荐模型)
) else (
    echo ⚠️  未找到 apple_sensitive.pt
)

if exist apple_best.pt (
    echo ✅ 找到 apple_best.pt
) else (
    echo ⚠️  未找到 apple_best.pt
)

if exist yolov8n.pt (
    echo ✅ 找到 yolov8n.pt (预训练模型)
) else (
    echo ⚠️  未找到 yolov8n.pt
)

echo.
echo [4] 测试摄像头...
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ 摄像头可用')
    cap.release()
else:
    print('❌ 摄像头不可用')
" 2>nul

echo.
echo [5] 测试Web应用启动...
echo 正在测试快速启动...
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo 📋 测试结果总结
echo ========================================
echo.
echo 如果所有测试都通过，可以：
echo 1. 双击 一键启动Web应用.bat
echo 2. 访问 http://localhost:8600
echo.
echo 如果有❌错误标记，请：
echo 1. 安装缺失的依赖包
echo 2. 确保摄像头连接正常
echo 3. 参考 Web应用使用指南.txt
echo.
echo 按任意键退出...
pause >nul