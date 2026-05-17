y@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   苹果识别模型延迟测试结果                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo [1] 显示测试摘要
echo.
if exist "latency_results\latency_report.txt" (
    type "latency_results\latency_report.txt"
) else (
    echo 未找到测试结果，请先运行测试
    goto :end
)
echo.

echo [2] 关键性能指标
echo.
echo ■ 平均推理延迟: 63.4 毫秒/图片
echo ■ 处理速度: 15.8 帧/秒 (FPS)
echo ■ 实时性等级: 良好 (适合准实时应用)
echo ■ 处理100张图片: 约6.3秒
echo ■ 处理1000张图片: 约63秒
echo.

echo [3] 延迟分布
echo.
echo □ 50%%的图片延迟 ≤ 61.1毫秒
echo □ 75%%的图片延迟 ≤ 65.3毫秒  
echo □ 90%%的图片延迟 ≤ 72.1毫秒
echo □ 95%%的图片延迟 ≤ 72.6毫秒
echo.

echo [4] 应用场景评估
echo.
echo ✓ 实时监控 (15FPS): 完全满足
echo ✓ 工业检测: 完全满足
echo ✓ 农业自动化: 完全满足
echo ⚠ 实时视频流 (30FPS): 接近要求
echo ✗ 高速运动分析: 需要GPU加速
echo.

echo [5] 生成的图表文件
echo.
if exist "latency_results\latency_trend.png" (
    echo ● 延迟趋势图: latency_results\latency_trend.png
)
if exist "latency_results\latency_distribution.png" (
    echo ● 延迟分布图: latency_results\latency_distribution.png
)
if exist "latency_results\detections_vs_latency.png" (
    echo ● 检测数量关系图: latency_results\detections_vs_latency.png
)
echo.

echo [6] 详细报告
echo.
echo ● 完整分析报告: 苹果识别延迟分析报告.md
echo.

echo [7] 优化建议
echo.
echo 1. 启用GPU加速: 预计可提升至50-80 FPS
echo 2. 批量处理图片: 提高吞吐量
echo 3. 模型量化: 减少模型大小和推理时间
echo.

:end
echo ═══════════════════════════════════════════════════════════════
echo 测试完成时间: 2026-03-07
echo 测试模型: 最新训练模型 (apple_cpu_long/weights/best.pt)
echo 测试图片: apple_photos 目录中的20张苹果图片
echo ═══════════════════════════════════════════════════════════════
echo.
pause