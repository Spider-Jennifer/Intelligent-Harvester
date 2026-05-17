#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动 YOLOv8 苹果检测 Web 应用
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("启动 YOLOv8 苹果检测 Web 应用")
    print("=" * 60)
    
    # 检查应用目录
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
    
    print("\n正在启动 Streamlit 应用...")
    print("请稍候，应用启动后将在浏览器中打开")
    print("访问地址: http://localhost:8501")
    print("\n按 Ctrl+C 停止应用")
    
    # 启动 Streamlit 应用
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", 
                       os.path.join(app_dir, "app.py"), "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动应用时出错: {e}")

if __name__ == "__main__":
    main()