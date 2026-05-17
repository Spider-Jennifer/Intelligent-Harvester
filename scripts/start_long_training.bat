@echo off
chcp 65001 >nul
title YOLOv8 Long Training - Apple Detection

echo ========================================
echo YOLOv8 Long Training - Apple Detection
echo ========================================
echo.

echo Checking environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not installed or not in PATH
    echo Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Checking dataset...
if not exist "apple_dataset\data.yaml" (
    echo Error: Dataset not found
    echo Please prepare the dataset first
    pause
    exit /b 1
)

echo.
echo ========================================
echo IMPORTANT: Long Training Information
echo ========================================
echo.
echo This training will:
echo - Run for 200 epochs
echo - Take several hours to complete
echo - Save checkpoints every 10 epochs
echo - Generate training charts
echo - Save best model as apple_long_trained.pt
echo.
echo Do NOT interrupt the training!
echo.
echo ========================================
echo.

set /p confirm="Start long training? (y/n): "
if /i not "%confirm%"=="y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting long training...
echo This will take several hours. Please wait...
echo.

python long_training.py

echo.
echo ========================================
echo Training completed!
echo ========================================
echo.
echo Check the results:
echo 1. Model: apple_long_trained.pt
echo 2. Training logs: runs\long_training\
echo 3. Charts and metrics in the runs folder
echo.
pause