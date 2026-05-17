@echo off
chcp 65001 >nul
echo ========================================
echo 苹果检测模型训练 - 立即开始
echo ========================================
echo.
echo 此脚本将使用迁移学习方法训练苹果检测模型
echo 即使标签文件较少，也能利用预训练模型的知识
echo.
echo 训练过程可能需要15-45分钟，请耐心等待...
echo.
echo 按任意键开始训练，或按 Ctrl+C 取消...
pause >nul

echo.
echo 正在检查环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

echo 正在安装必要库...
pip install ultralytics --quiet
if errorlevel 1 (
    echo 警告: 安装ultralytics失败，尝试继续...
)

echo.
echo 开始训练...
echo ========================================
python train_with_few_labels.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 生成的模型文件:
if exist "apple_transfer_best.pt" (
    echo - apple_transfer_best.pt (迁移学习模型)
)
if exist "yolov8n.pt" (
    echo - yolov8n.pt (预训练模型)
)
echo.
echo 使用方法:
echo 1. 测试模型: python test_trained_model.py
echo 2. 在摄像头上测试: python test_trained_model.py camera
echo 3. 更新应用: 修改 fast_camera_app.py 中的模型路径
echo.
pause