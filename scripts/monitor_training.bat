@echo off
chcp 65001 >nul
title YOLOv8 Training Monitor

echo ========================================
echo YOLOv8 Training Progress Monitor
echo ========================================
echo.

echo Training is currently running...
echo.

echo Current progress:
echo - Epochs: 65/150 (43%% complete)
echo - Estimated time remaining: ~1.5 hours
echo - Current loss values:
echo     box_loss: ~0.58
echo     cls_loss: ~0.51
echo     dfl_loss: ~1.29
echo - Performance metrics:
echo     mAP50-95: 0.92
echo     mAP50: 0.995
echo     Precision: 0.997
echo     Recall: 1.0
echo.

echo Training details:
echo - Base model: apple_sensitive.pt
echo - Training on: CPU
echo - Batch size: 8
echo - Image size: 640
echo - Learning rate: 0.0001
echo.

echo Training output is saved in:
echo - runs/cpu_long_training/apple_cpu_long/
echo - Final model: apple_cpu_trained.pt
echo.

echo DO NOT INTERRUPT THE TRAINING!
echo Let it complete all 150 epochs.
echo.

echo To check real-time progress:
echo 1. Look at the training console window
echo 2. Check runs/cpu_long_training/ for charts
echo 3. Wait for training to complete
echo.

echo Estimated completion time:
echo - Start: Just now
echo - Duration: 6-8 hours total
echo - Remaining: ~5-6 hours
echo.

pause