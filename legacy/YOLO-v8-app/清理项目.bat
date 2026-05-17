@echo off
chcp 65001 >nul
echo ========================================
echo    YOLOv8 项目清理脚本
echo ========================================
echo.

echo [1] 清理重复启动脚本...
del "简单启动.bat" 2>nul
del "快速诊断.bat" 2>nul
del "桌面启动.bat" 2>nul
del "测试启动.py" 2>nul
del "非交互式启动.py" 2>nul

echo [2] 清理重复验证脚本...
del "verify_project.py" 2>nul
del "backup_project.py" 2>nul

echo [3] 清理旧备份文件（保留最新）...
for /f "skip=1 delims=" %%f in ('dir /b /o-d "YOLOv8_apple_detection_backup_*.zip"') do del "%%f"
for /f "skip=1 delims=" %%f in ('dir /b /o-d "YOLOv8_apple_detection_deploy_*.zip"') do del "%%f"

echo [4] 清理多余虚拟环境...
rmdir /s /q "apple_env" 2>nul
rmdir /s /q "new_venv" 2>nul

echo [5] 清理其他冗余文件...
del "emergency_restore.py" 2>nul
del "main.py" 2>nul

echo.
echo ========================================
echo    清理完成！
echo ========================================
echo.
echo 保留的核心文件：
echo - YOLOv8-app-master\ (主应用)
echo - .venv\ (虚拟环境)
echo - restore_and_run.bat (启动脚本)
echo - simple_verify.py (验证脚本)
echo - 最新备份文件
echo.
pause