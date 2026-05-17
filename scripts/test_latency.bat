@echo off
chcp 65001 >nul
echo ========================================
echo 模型推理延迟测试
echo ========================================
echo.

echo 正在检查Python环境...
python --version
echo.

echo 正在检查ultralytics库...
python -c "import ultralytics; print(f'ultralytics版本: {ultralytics.__version__}')"
echo.

echo 正在检查苹果图片目录...
if exist "apple_photos" (
    dir /b "apple_photos\*.jpg" | find /c /v "" >nul
    for /f %%i in ('dir /b "apple_photos\*.jpg" ^| find /c /v ""') do set jpg_count=%%i
    for /f %%i in ('dir /b "apple_photos\*.png" ^| find /c /v ""') do set png_count=%%i
    set /a total_count=jpg_count+png_count
    echo 找到 %total_count% 张苹果图片
) else (
    echo 错误: apple_photos目录不存在!
    pause
    exit /b 1
)
echo.

echo 正在检查模型文件...
if exist "apple_sensitive.pt" (
    for %%F in ("apple_sensitive.pt") do set size=%%~zF
    set /a size_mb=size/1024/1024
    echo [OK] apple_sensitive.pt (%size_mb% MB)
)
if exist "apple_improved.pt" (
    for %%F in ("apple_improved.pt") do set size=%%~zF
    set /a size_mb=size/1024/1024
    echo [OK] apple_improved.pt (%size_mb% MB)
)
if exist "apple_best.pt" (
    for %%F in ("apple_best.pt") do set size=%%~zF
    set /a size_mb=size/1024/1024
    echo [OK] apple_best.pt (%size_mb% MB)
)
if exist "runs\cpu_long_training\apple_cpu_long\weights\best.pt" (
    for %%F in ("runs\cpu_long_training\apple_cpu_long\weights\best.pt") do set size=%%~zF
    set /a size_mb=size/1024/1024
    echo [OK] 最新训练模型 (%size_mb% MB)
)
if exist "yolov8n.pt" (
    for %%F in ("yolov8n.pt") do set size=%%~zF
    set /a size_mb=size/1024/1024
    echo [OK] yolov8n.pt (%size_mb% MB)
)
echo.

echo 正在运行延迟测试...
echo 这可能需要几分钟时间，请耐心等待...
echo.

python test_inference_latency.py

echo.
echo ========================================
echo 测试完成!
echo ========================================
echo.
echo 结果已保存到 latency_results 目录
echo 包括:
echo   1. 模型对比图
echo   2. 延迟分布图
echo   3. 延迟趋势图
echo   4. 检测数量关系图
echo   5. 详细报告
echo.
pause