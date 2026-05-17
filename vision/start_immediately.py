import subprocess
import sys
import time
import webbrowser

print("=" * 60)
print("YOLOv8苹果检测项目 - 立即启动")
print("=" * 60)
print()
print("正在启动优化版Web摄像头检测应用...")
print("端口: 8600")
print("访问地址: http://localhost:8600")
print("按 Ctrl+C 停止应用")
print()

# 清理旧进程
print("清理旧进程...")
try:
    subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

time.sleep(2)

# 在浏览器中打开
print("在浏览器中打开应用...")
try:
    webbrowser.open("http://localhost:8600")
    print("已尝试在浏览器中打开应用")
except:
    print("无法自动打开浏览器，请手动访问: http://localhost:8600")

print()
print("启动Streamlit应用中...")
print("-" * 40)
print()

# 启动Streamlit应用
try:
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "fast_camera_app.py", 
        "--server.port", "8600",
        "--server.headless", "true"
    ])
except KeyboardInterrupt:
    print("\n应用已停止")
except Exception as e:
    print(f"启动失败: {e}")
    print("\n请尝试以下方法:")
    print("1. 双击运行 run_fast_camera.bat")
    print("2. 检查端口8600是否被占用")
    print("3. 尝试其他端口: python -m streamlit run fast_camera_app.py --server.port 8605")