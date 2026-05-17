@echo off
chcp 65001 >nul
echo.
echo ================================================
echo   树莓派摄像头问题修复工具 - Windows辅助脚本
echo ================================================
echo.

echo 此脚本将帮助您准备修复树莓派摄像头问题的文件
echo.

REM 检查Python文件
if exist "test_camera_simple.py" (
    echo ✓ test_camera_simple.py 已存在
) else (
    echo ✗ test_camera_simple.py 未找到
)

if exist "raspberry_client_camera_fixed.py" (
    echo ✓ raspberry_client_camera_fixed.py 已存在
) else (
    echo ✗ raspberry_client_camera_fixed.py 未找到
)

if exist "fix_camera_issue.py" (
    echo ✓ fix_camera_issue.py 已存在
) else (
    echo ✗ fix_camera_issue.py 未找到
)

echo.
echo ================================================
echo   文件传输指南
echo ================================================
echo.

echo 请将以下文件传输到树莓派:
echo.
echo 1. test_camera_simple.py
echo 2. raspberry_client_camera_fixed.py
echo 3. fix_camera_issue.py
echo 4. raspberry_client_ultra_light.py (已修复版本)
echo.

echo 传输方法:
echo.
echo 方法1: 使用SCP命令 (在Windows PowerShell或CMD中运行):
echo    scp *.py pi@192.168.45.162:~/yolov8_remote/
echo.
echo 方法2: 使用FileZilla等FTP工具
echo.
echo 方法3: 使用U盘复制
echo.

echo ================================================
echo   树莓派操作步骤
echo ================================================
echo.

echo 在树莓派上执行以下步骤:
echo.
echo 1. 测试摄像头:
echo    cd ~/yolov8_remote
echo    python3 test_camera_simple.py
echo.
echo 2. 如果测试失败，运行修复脚本:
echo    sudo python3 fix_camera_issue.py
echo.
echo 3. 启用摄像头并重启:
echo    sudo raspi-config
echo    # 选择 Interface Options -> Camera -> Yes
echo    sudo reboot
echo.
echo 4. 使用修复版客户端:
echo    python3 raspberry_client_camera_fixed.py
echo.

echo ================================================
echo   快速修复命令 (复制到树莓派SSH)
echo ================================================
echo.

echo 复制以下命令到树莓派SSH会话中执行:
echo.
echo # 1. 进入项目目录
echo cd ~/yolov8_remote
echo.
echo # 2. 备份原客户端
echo cp raspberry_client_ultra_light.py raspberry_client_ultra_light.py.backup
echo.
echo # 3. 测试摄像头
echo python3 test_camera_simple.py
echo.
echo # 4. 如果第3步失败，启用摄像头
echo sudo raspi-config
echo # 手动操作: Interface Options -> Camera -> Yes -> Finish
echo sudo reboot
echo.
echo # 5. 重启后测试
echo python3 test_camera_simple.py
echo.
echo # 6. 运行修复版客户端
echo python3 raspberry_client_camera_fixed.py
echo.

echo ================================================
echo   紧急联系方式
echo ================================================
echo.

echo 如果问题仍未解决，请提供:
echo 1. 树莓派型号和摄像头型号
echo 2. 完整的错误信息
echo 3. 测试命令的输出
echo.

pause