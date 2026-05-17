#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单环境测试脚本
"""

import sys

print("=" * 50)
print("YOLOv8 Web应用环境测试")
print("=" * 50)
print()

# 1. Python版本
print("[1] Python版本")
print("    Python版本:", sys.version[:50])
print("    Python路径:", sys.executable)
print()

# 2. 测试依赖包
print("[2] 依赖包测试")

# Streamlit
try:
    import streamlit
    print("    [OK] Streamlit版本:", streamlit.__version__)
except:
    print("    [ERROR] Streamlit未安装")

# OpenCV
try:
    import cv2
    print("    [OK] OpenCV版本:", cv2.__version__)
except:
    print("    [ERROR] OpenCV未安装")

# Ultralytics
try:
    import ultralytics
    print("    [OK] Ultralytics版本:", ultralytics.__version__)
except:
    print("    [ERROR] Ultralytics未安装")

# NumPy
try:
    import numpy as np
    print("    [OK] NumPy版本:", np.__version__)
except:
    print("    [ERROR] NumPy未安装")

print()

# 3. 模型文件
print("[3] 模型文件")
import os

models_to_check = [
    "apple_sensitive.pt",
    "apple_best.pt", 
    "apple_improved.pt",
    "apple_quick.pt",
    "yolov8n.pt"
]

for model in models_to_check:
    if os.path.exists(model):
        size = os.path.getsize(model) / (1024 * 1024)
        print(f"    [OK] {model} ({size:.1f} MB)")
    else:
        print(f"    [WARN] {model} (未找到)")

print()

# 4. 摄像头测试（简单测试）
print("[4] 摄像头测试")
try:
    import cv2
    # 简单测试，不显示错误
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows使用DSHOW
    if cap.isOpened():
        print("    [OK] 摄像头可用")
        cap.release()
    else:
        print("    [WARN] 摄像头不可用")
except Exception as e:
    print("    [ERROR] 摄像头测试失败")

print()

# 5. 启动建议
print("[5] 启动建议")
print("    要启动Web应用，请执行以下操作：")
print("    1. 双击 '一键启动Web应用.bat'")
print("    2. 或运行命令: python start_web_app.py")
print("    3. 在浏览器中访问: http://localhost:8600")
print()
print("    如果遇到问题，请查看 'Web应用使用指南.txt'")
print()

print("=" * 50)
print("测试完成")
print("=" * 50)
print()
input("按Enter键退出...")