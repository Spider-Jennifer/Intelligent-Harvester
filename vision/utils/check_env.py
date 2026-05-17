#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查环境并启动应用
"""

import os
import sys
import subprocess

def check_environment():
    print("=" * 60)
    print("检查运行环境")
    print("=" * 60)
    
    # 检查Python
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 检查Streamlit
    try:
        import streamlit
        print(f"Streamlit版本: {streamlit.__version__}")
        return True
    except ImportError:
        print("Streamlit未安装")
        return False

def install_streamlit():
    print("\n正在安装Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Streamlit安装成功！")
        return True
    except Exception as e:
        print(f"安装失败: {e}")
        return False

def start_app():
    print("\n" + "=" * 60)
    print("启动YOLOv8苹果检测应用")
    print("=" * 60)
    
    app_dir = "YOLO-v8-app/YOLOv8-app-master"
    if not os.path.exists(app_dir):
        print(f"错误：应用目录不存在: {app_dir}")
        return
    
    print(f"应用目录: {app_dir}")
    
    # 检查关键文件
    required_files = [
        "app.py",
        "config.py", 
        "utils.py",
        "weights/best.pt"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(app_dir, file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"[OK] {file_path} ({size:,} bytes)")
        else:
            print(f"[ERROR] {file_path} - 文件不存在")
            return
    
    print("\n正在启动应用...")
    print("访问网址: http://localhost:8501")
    print("按 Ctrl+C 停止应用")
    
    # 启动应用
    try:
        os.chdir(app_dir)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")

def main():
    if not check_environment():
        if not install_streamlit():
            print("\n请手动安装依赖:")
            print("pip install streamlit")
            return
    
    start_app()

if __name__ == "__main__":
    main()