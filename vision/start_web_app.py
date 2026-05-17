#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动Web界面摄像头检测应用
"""

import subprocess
import sys
import time
import webbrowser

def main():
    print("=" * 50)
    print("启动Web界面苹果摄像头检测")
    print("=" * 50)
    print()
    print("优化版本特点：")
    print("1. 降低分辨率 (640x480) - 减少计算量")
    print("2. 跳帧处理 (每2帧处理1次) - 提高FPS")
    print("3. 异步架构 - 摄像头读取和模型推理分离")
    print("4. 实时性能监控 - 显示FPS和延迟")
    print()
    
    # 选择端口
    port = 8600
    url = f"http://localhost:{port}"
    
    print(f"应用将在以下地址启动：")
    print(f"   {url}")
    print()
    
    # 尝试在浏览器中打开
    try:
        webbrowser.open(url)
        print("[√] 已尝试在浏览器中打开应用")
    except:
        print("[!] 无法自动打开浏览器，请手动访问上述网址")
    
    print()
    print("[>] 正在启动Streamlit应用...")
    print("   按 Ctrl+C 停止应用")
    print("=" * 50)
    print()
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "fast_camera_app.py", 
            "--server.port", str(port),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n[~] 应用已停止")
    except Exception as e:
        print(f"[X] 启动失败: {e}")
        print("\n请尝试以下方法：")
        print("1. 检查是否安装了Streamlit: pip install streamlit")
        print("2. 检查端口8600是否被占用")
        print("3. 双击运行 run_fast_camera.bat")

if __name__ == "__main__":
    main()