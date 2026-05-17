@echo off
chcp 65001 >nul
echo ========================================
echo    YOLOv8 苹果检测应用 - 完整恢复脚本
echo ========================================
echo.

set PROJECT_ROOT=%~dp0
set APP_DIR=%PROJECT_ROOT%YOLOv8-app-master
set VENV_DIR=%PROJECT_ROOT%.venv

echo [1/6] 检查项目文件...
if not exist "%APP_DIR%\app.py" (
    echo [错误] 找不到 app.py 文件
    pause
    exit /b 1
)
if not exist "%APP_DIR%\config.py" (
    echo [错误] 找不到 config.py 文件
    pause
    exit /b 1
)
if not exist "%APP_DIR%\utils.py" (
    echo [错误] 找不到 utils.py 文件
    pause
    exit /b 1
)
echo [OK] 项目文件检查通过

echo [2/6] 检查模型文件...
if not exist "%APP_DIR%\weights\best.pt" (
    echo [错误] 找不到模型文件 weights/best.pt
    echo 请确保模型文件存在
    pause
    exit /b 1
)
echo [OK] 模型文件检查通过

echo [3/6] 检查虚拟环境...
if not exist "%VENV_DIR%" (
    echo [警告] 虚拟环境不存在，正在创建...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
)
echo [OK] 虚拟环境检查通过

echo [4/6] 激活虚拟环境...
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)
echo [OK] 虚拟环境激活成功

echo [5/6] 检查并安装依赖...
python -c "import streamlit, ultralytics, torch, cv2, PIL" 2>nul
if errorlevel 1 (
    echo [信息] 正在安装依赖包...
    pip install -r "%APP_DIR%\requirements_final.txt"
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)
echo [OK] 依赖包检查通过

echo [6/6] 启动应用...
echo.
echo ========================================
echo    应用即将启动...
echo    访问地址: http://localhost:8501
echo    按 Ctrl+C 停止应用
echo ========================================
echo.

cd /d "%APP_DIR%"
streamlit run app.py --server.headless=false --server.port=8501

echo.
echo 应用已停止
pause