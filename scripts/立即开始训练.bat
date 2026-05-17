@echo off
chcp 65001 >nul
title YOLOv8 苹果检测模型 - 立即开始训练

echo ========================================
echo YOLOv8 苹果检测模型准确率提升
echo ========================================
echo.

echo 正在检查环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    echo 请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo 选择训练模式:
echo.
echo 1. 快速训练 (30-60分钟) - 立即开始
echo 2. 查看当前模型性能
echo 3. 查看详细指南
echo 4. 退出
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 启动快速准确率提升训练...
    echo 预计时间: 30-60分钟
    echo 按 Ctrl+C 可以中断训练
    echo.
    echo 正在检查依赖包...
    python -c "import ultralytics" 2>nul
    if errorlevel 1 (
        echo 安装ultralytics...
        pip install ultralytics -q
    )
    echo.
    echo 开始训练...
    python improve_accuracy_now.py
) else if "%choice%"=="2" (
    echo.
    echo 检查当前模型性能...
    python test_model_accuracy.py
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    echo 打开详细指南...
    start 提高模型准确率指南.txt
    echo 指南已打开，请查看文件内容
    echo.
    pause
) else (
    echo 退出...
    exit /b 0
)

echo.
echo 操作完成!
echo.
pause