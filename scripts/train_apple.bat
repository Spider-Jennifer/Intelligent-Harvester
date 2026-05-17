@echo off
chcp 65001 >nul
echo ========================================
echo 苹果检测模型训练脚本
echo ========================================
echo.
echo 正在检查环境...
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查ultralytics库
python -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo 警告: ultralytics库未安装，正在尝试安装...
    pip install ultralytics
    if errorlevel 1 (
        echo 错误: 安装ultralytics失败
        pause
        exit /b 1
    )
    echo ultralytics库安装成功
)

echo.
echo 环境检查通过，开始训练...
echo.
echo 训练参数:
echo   - 训练轮数: 50
echo   - 批次大小: 8
echo   - 图像尺寸: 320
echo   - 设备: CPU
echo.
echo 训练过程可能需要10-30分钟，请耐心等待...
echo.

REM 运行训练脚本
python train_apple_simple.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 下一步操作:
echo 1. 测试训练好的模型: python train_apple_simple.py test
echo 2. 复制模型到项目目录: copy runs\detect\apple_detection_simple\weights\best.pt apple_best.pt
echo 3. 使用新模型运行摄像头检测: python fast_camera_app.py
echo.
pause