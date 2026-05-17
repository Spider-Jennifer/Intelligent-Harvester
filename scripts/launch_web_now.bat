@echo off
chcp 65001 >nul

echo ========================================
echo 🚀 立即启动Web摄像头检测应用
echo ========================================
echo.

echo 📊 当前状态检查...
echo.

REM 检查端口8600
netstat -ano | findstr :8600 >nul
if %errorlevel% equ 0 (
    echo ✅ 端口8600 (优化版) 正在运行
    echo   访问地址: http://localhost:8600
) else (
    echo ❌ 端口8600 (优化版) 未运行
)

echo.

REM 检查端口8520
netstat -ano | findstr :8520 >nul
if %errorlevel% equ 0 (
    echo ✅ 端口8520 (原始版) 正在运行
    echo   访问地址: http://localhost:8520
) else (
    echo ❌ 端口8520 (原始版) 未运行
)

echo.
echo 🔧 解决方案:
echo 1. 如果端口已运行但无法访问:
echo    - 尝试使用: http://127.0.0.1:8520
echo    - 尝试使用: http://127.0.0.1:8600
echo    - 检查浏览器是否阻止了localhost访问
echo.
echo 2. 如果端口未运行:
echo    - 按任意键启动优化版Web应用 (端口8600)
echo    - 或按 Ctrl+C 取消
echo.

pause

echo.
echo 🚀 正在启动优化版Web应用...
echo 端口: 8600
echo 访问地址: http://localhost:8600
echo 按 Ctrl+C 停止应用
echo.

REM 清理旧进程
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

REM 在浏览器中打开
start "" http://localhost:8600

REM 启动应用
python -m streamlit run fast_camera_app.py --server.port 8600 --server.headless true

pause