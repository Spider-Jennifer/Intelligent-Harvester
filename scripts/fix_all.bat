@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测项目 - 完整修复
echo ========================================

echo.
echo [步骤1] 检查Python环境
python --version
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    echo 请安装Python 3.8+ 并添加到PATH
    pause
    exit /b 1
)

echo.
echo [步骤2] 安装所有依赖
echo 正在安装依赖，请稍候...
pip install torch torchvision ultralytics opencv-python streamlit pillow numpy matplotlib

echo.
echo [步骤3] 修复Streamlit路径问题
set STREAMLIT_PATH=%USERPROFILE%\AppData\Roaming\Python\Python39\Scripts\streamlit.exe
if exist "%STREAMLIT_PATH%" (
    echo Streamlit路径: %STREAMLIT_PATH%
    set PATH=%STREAMLIT_PATH%;%PATH%
) else (
    echo 警告: Streamlit可执行文件未找到
    echo 尝试重新安装...
    pip install --force-reinstall streamlit
)

echo.
echo [步骤4] 运行命令行测试
echo 运行模型测试...
cd /d "%~dp0"
python yolov8-apple-detection-master/test_apple_model.py --num_images 3

echo.
echo [步骤5] 启动Web应用（使用不同端口）
echo 启动Web应用，请访问: http://localhost:8503
echo 按 Ctrl+C 停止应用
echo.
cd /d "%~dp0YOLO-v8-app\YOLOv8-app-master"
python -m streamlit run app.py --server.port 8503 --server.headless true

pause