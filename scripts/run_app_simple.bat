@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测应用 - 直接启动
echo ========================================

echo.
echo 步骤1: 检查Python环境
python --version
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 步骤2: 检查Streamlit
python -c "import streamlit; print('Streamlit已安装:', streamlit.__version__)"
if errorlevel 1 (
    echo Streamlit未安装，正在安装...
    pip install streamlit
)

echo.
echo 步骤3: 启动应用
echo 请访问: http://localhost:8501
echo 按 Ctrl+C 停止应用
echo.

cd /d "%~dp0YOLO-v8-app\YOLOv8-app-master"
python -m streamlit run app.py --server.port 8501

pause