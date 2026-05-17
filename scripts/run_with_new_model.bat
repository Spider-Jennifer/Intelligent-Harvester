@echo off
chcp 65001 >nul
echo ========================================
echo 苹果检测 - 使用新训练模型
echo ========================================
echo.
echo 正在启动应用...
echo 默认使用新训练的 apple_best.pt 模型
echo.
echo 访问网址: http://localhost:8600
echo.
echo 按 Ctrl+C 停止应用
echo.

REM 停止可能正在运行的Python和Streamlit进程
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

REM 运行应用
python -m streamlit run fast_camera_app.py --server.port 8600 --server.headless true