#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试应用
"""

import os
import sys

def main():
    print("=" * 60)
    print("YOLOv8 苹果检测 - 快速测试")
    print("=" * 60)
    
    # 测试1: 运行命令行测试
    print("\n[测试1] 运行命令行测试...")
    test_script = "yolov8-apple-detection-master/test_apple_model.py"
    if os.path.exists(test_script):
        print(f"运行: {test_script}")
        os.system(f"python {test_script}")
    else:
        print(f"脚本不存在: {test_script}")
    
    # 测试2: 检查Web应用
    print("\n[测试2] 检查Web应用...")
    app_dir = "YOLO-v8-app/YOLOv8-app-master"
    if os.path.exists(app_dir):
        print(f"应用目录存在: {app_dir}")
        
        # 检查关键文件
        files = ["app.py", "config.py", "utils.py", "weights/best.pt"]
        for f in files:
            path = os.path.join(app_dir, f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  [OK] {f} ({size:,} bytes)")
            else:
                print(f"  [ERROR] {f} - 文件不存在")
    else:
        print(f"应用目录不存在: {app_dir}")
    
    print("\n" + "=" * 60)
    print("解决方案:")
    print("1. 如果Streamlit启动失败，请尝试:")
    print("   python -m streamlit run YOLO-v8-app/YOLOv8-app-master/app.py")
    print("\n2. 或使用命令行测试:")
    print("   python yolov8-apple-detection-master/test_apple_model.py")
    print("\n3. 检查端口占用:")
    print("   netstat -ano | findstr :8501")
    print("\n4. 手动启动应用:")
    print("   cd YOLO-v8-app/YOLOv8-app-master")
    print("   streamlit run app.py --server.port 8501")

if __name__ == "__main__":
    main()