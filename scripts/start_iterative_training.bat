@echo off
chcp 65001 >nul
title YOLOv8 Iterative Training

echo ========================================
echo YOLOv8 Iterative Training
echo ========================================
echo.

echo Checking environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not installed
    pause
    exit /b 1
)

if not exist "apple_dataset\data.yaml" (
    echo Error: Dataset not found
    pause
    exit /b 1
)

echo.
echo ========================================
echo Iterative Training Options
echo ========================================
echo.
echo This will train the model multiple times,
echo each time using the previous best model.
echo.
echo Recommended: Option 1 (3 iterations)
echo.
echo ========================================
echo.

python iterative_training.py

echo.
pause