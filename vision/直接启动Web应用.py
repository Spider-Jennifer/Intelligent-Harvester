#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接启动Web应用
"""

import subprocess
import sys
import webbrowser

def main():
    print("=" * 50)
    print("启动YOLOv8苹果检测Web应用")
    print("=" * 50)
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
        print("已尝试在浏览器中打开应用")
    except:
        print("无法自动打开浏览器，请手动访问上述网址")
    
    print()
    print("正在启动Streamlit应用...")
    print("按 Ctrl+C 停止应用")
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
        print("\n\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n请尝试以下方法：")
        print("1. 检查是否安装了Streamlit: pip install streamlit")
        print("2. 检查端口8600是否被占用")
        print("3. 直接运行: streamlit run fast_camera_app.py")

if __name__ == "__main__":
    main()