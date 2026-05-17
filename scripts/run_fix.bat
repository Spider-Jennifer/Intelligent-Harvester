@echo off
chcp 65001 >nul
echo 启动苹果检测模型误识别修复脚本...
echo 脚本目录: %CD%
python fix_apple_misdetection.py
pause