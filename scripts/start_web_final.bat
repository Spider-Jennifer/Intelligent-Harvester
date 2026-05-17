@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv8 苹果检测 Web 应用 - 最终启动
echo ========================================

echo.
echo [1] 清理可能占用的端口...
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul

echo.
echo [2] 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: Python未安装
    pause
    exit /b 1
)

echo.
echo [3] 检查Streamlit...
python -c "import streamlit; print('Streamlit版本:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo Streamlit未安装，正在安装...
    pip install streamlit
)

echo.
echo [4] 启动应用...
echo 使用端口: 8520
echo 访问网址: http://localhost:8520
echo.
echo 如果浏览器没有自动打开，请手动访问上述网址
echo 按 Ctrl+C 停止应用
echo.

cd /d "%~dp0YOLO-v8-app\YOLOv8-app-master"
start "" http://localhost:8520
python -m streamlit run app.py --server.port 8520

pause