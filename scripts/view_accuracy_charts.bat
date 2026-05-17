@echo off
chcp 65001 >nul
title YOLOv8 Accuracy Charts Viewer

echo ========================================
echo YOLOv8 Apple Detection Accuracy Charts
echo ========================================
echo.

echo Available charts and reports:
echo.

echo 1. Detection Accuracy Chart (折线图)
echo    - File: detection_accuracy_chart.png
echo    - Shows: mAP50 and mAP50-95 trends
echo.

echo 2. Training Metrics Summary (综合指标)
echo    - File: training_metrics_summary.png
echo    - Shows: All training metrics in 4 charts
echo.

echo 3. HTML Training Report (HTML报告)
echo    - File: training_report.html
echo    - Shows: Detailed statistics and tables
echo.

echo 4. Current Accuracy Report (当前识别率)
echo    - File: accuracy_report_final.md
echo    - Shows: Current performance analysis
echo.

echo 5. Show Current Accuracy (命令行显示)
echo    - Command: python show_current_accuracy.py
echo.

echo ========================================
echo.

set /p choice="Select option (1-5, or 0 to exit): "

if "%choice%"=="1" (
    echo Opening detection_accuracy_chart.png...
    start detection_accuracy_chart.png
) else if "%choice%"=="2" (
    echo Opening training_metrics_summary.png...
    start training_metrics_summary.png
) else if "%choice%"=="3" (
    echo Opening training_report.html in browser...
    start training_report.html
) else if "%choice%"=="4" (
    echo Opening accuracy_report_final.md...
    start accuracy_report_final.md
) else if "%choice%"=="5" (
    echo Running accuracy check...
    python show_current_accuracy.py
    echo.
    pause
) else if "%choice%"=="0" (
    echo Exiting...
) else (
    echo Invalid choice
)

echo.
echo ========================================
echo Note: Training is still in progress
echo Current epoch: 64/150 (42.7%% complete)
echo ========================================
echo.

if not "%choice%"=="0" (
    pause
)