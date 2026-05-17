@echo off
echo ========================================
echo YOLOv8 苹果检测微调训练
echo ========================================
echo.

cd /d "c:\Users\李晨鑫\Desktop\YOLO-v8-app"
call .venv\Scripts\activate

echo 请输入苹果图片文件夹路径，然后按回车：
set /p images_path=

echo 开始微调训练...
echo C:\Users\李晨鑫\Desktop\苹果照片 | python start_training.py

echo.
echo 训练完成！按任意键退出...
pause >nul