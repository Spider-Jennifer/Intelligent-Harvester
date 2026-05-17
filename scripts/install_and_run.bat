@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测项目 - 安装依赖并运行
echo ========================================

echo.
echo 1. 安装依赖...
pip install torch torchvision ultralytics opencv-python streamlit

echo.
echo 2. 运行测试...
python yolov8-apple-detection-master/test_apple_model.py

echo.
echo 3. 启动Web应用...
echo 请在新窗口中运行: streamlit run YOLO-v8-app/YOLOv8-app-master/app.py

pause