@echo off
chcp 65001 >nul

echo ========================================
echo 远程推理系统 - 一键启动
echo ========================================
echo.
echo 请选择操作：
echo 1. 启动电脑服务器（PC端）
echo 2. 启动树莓派客户端（树莓派端）
echo 3. 检查依赖
echo 4. 退出
echo.

set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" goto start_server
if "%choice%"=="2" goto start_client
if "%choice%"=="3" goto check_deps
if "%choice%"=="4" goto exit

echo 无效选择
pause
exit

:start_server
echo.
echo 正在启动电脑服务器...
echo 请确保：
echo 1. 电脑和树莓派在同一网络
echo 2. 防火墙允许9999端口
echo 3. 模型文件存在
echo.
python remote_inference_server.py
pause
exit

:start_client
echo.
echo 请选择客户端版本：
echo 1. 完整版（功能全面）
echo 2. 极致精简版（推荐树莓派4B）
echo.

set /p client_choice="请选择 (1-2): "

if "%client_choice%"=="1" (
    echo 启动完整版客户端...
    python raspberry_client.py
) else if "%client_choice%"=="2" (
    echo 启动极致精简版客户端...
    python raspberry_client_ultra_light.py
) else (
    echo 无效选择
)
pause
exit

:check_deps
echo.
echo 检查依赖...
python check_deps.py
echo.
echo 如需安装缺失依赖，请运行：
echo pip install ultralytics opencv-python numpy
pause
exit

:exit
echo 退出
pause