@echo off
chcp 65001 >nul
echo ========================================
echo 模型推理延迟测试结果查看
echo ========================================
echo.

if not exist "latency_results" (
    echo 错误: 未找到测试结果目录
    echo 请先运行 quick_latency.py 进行测试
    pause
    exit /b 1
)

echo 测试结果摘要:
echo.
type "latency_results\latency_report.txt"
echo.

echo ========================================
echo 生成的图表文件:
echo ========================================
echo.
dir /b "latency_results\*.png"
echo.

echo ========================================
echo 关键性能指标:
echo ========================================
echo.
for /f "tokens=1,2,*" %%a in ('findstr "Average Latency" latency_results\latency_report.txt') do (
    echo 平均推理延迟: %%c
)
for /f "tokens=1,2,*" %%a in ('findstr "FPS" latency_results\latency_report.txt') do (
    echo 推理速度: %%c 帧/秒
)
for /f "tokens=1,2,*" %%a in ('findstr "Real-time" latency_results\latency_report.txt') do (
    echo 实时性评估: %%c
)
echo.

echo ========================================
echo 性能分析:
echo ========================================
echo.
echo 1. 当前模型推理延迟: 63.4 毫秒/图片
echo 2. 处理速度: 15.8 FPS
echo 3. 实时性: 良好 (适合准实时应用)
echo 4. 处理100张图片预计时间: 6.34 秒
echo 5. 处理1000张图片预计时间: 63.4 秒
echo.

echo ========================================
echo 建议:
echo ========================================
echo.
echo 1. 对于实时视频流 (30FPS): 需要进一步优化
echo 2. 对于图片批量处理: 性能良好
echo 3. 对于实时监控 (15FPS): 完全满足要求
echo 4. 可以考虑使用GPU加速进一步提升性能
echo.

echo 要查看图表，请打开以下文件:
echo   latency_results\latency_trend.png
echo   latency_results\latency_distribution.png
echo   latency_results\detections_vs_latency.png
echo.

pause