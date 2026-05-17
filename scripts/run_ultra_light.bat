@echo off
chcp 65001 >nul

echo ========================================
echo 树莓派4B苹果检测 - 极致精简版
echo ========================================
echo.
echo 优化特点：
echo 1. 绝对最小内存占用
echo 2. 硬编码优化参数
echo 3. 移除所有对象创建开销
echo 4. 最小推理尺寸 (160x120)
echo 5. 极致简化逻辑
echo.
echo 启动极致精简版...
echo 按 Q 或 ESC 退出
echo.

REM 清理进程
taskkill /f /im python.exe 2>nul
timeout /t 1 /nobreak >nul

REM 运行程序
python raspberry_pi_ultra_light.py

pause