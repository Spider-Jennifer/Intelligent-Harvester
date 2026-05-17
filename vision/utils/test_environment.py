#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试YOLOv8 Web应用环境
"""

import sys

print("=" * 50)
print("YOLOv8 Web应用环境测试")
print("=" * 50)
print()

# 1. 测试Python版本
print("[1] Python版本测试")
print(f"    Python版本: {sys.version}")
print(f"    Python路径: {sys.executable}")
print()

# 2. 测试依赖包
print("[2] 依赖包测试")

try:
    import streamlit
    print(f"    [OK] Streamlit版本: {streamlit.__version__}")
except ImportError:
    print("    ❌ Streamlit未安装")

try:
    import cv2
    print(f"    ✅ OpenCV版本: {cv2.__version__}")
except ImportError:
    print("    ❌ OpenCV未安装")

try:
    import ultralytics
    print(f"    ✅ Ultralytics版本: {ultralytics.__version__}")
except ImportError:
    print("    ❌ Ultralytics未安装")

try:
    import numpy as np
    print(f"    ✅ NumPy版本: {np.__version__}")
except ImportError:
    print("    ❌ NumPy未安装")

try:
    from PIL import Image
    print("    ✅ PIL/Pillow已安装")
except ImportError:
    print("    ❌ PIL/Pillow未安装")

print()

# 3. 测试模型文件
print("[3] 模型文件测试")
import os

models = [
    ("apple_sensitive.pt", "高灵敏度模型（推荐）"),
    ("apple_best.pt", "最佳模型"),
    ("apple_improved.pt", "改进模型"),
    ("apple_quick.pt", "快速模型"),
    ("yolov8n.pt", "预训练模型")
]

for model_file, description in models:
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024 * 1024)
        print(f"    ✅ {model_file} - {description} ({size:.1f} MB)")
    else:
        print(f"    ⚠️  {model_file} - {description} (未找到)")

print()

# 4. 测试摄像头
print("[4] 摄像头测试")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("    ✅ 摄像头可用")
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print(f"    ✅ 摄像头分辨率: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("    ⚠️  摄像头可以打开但无法读取画面")
        cap.release()
    else:
        print("    ❌ 摄像头不可用")
except Exception as e:
    print(f"    ❌ 摄像头测试失败: {e}")

print()

# 5. 测试Web应用启动
print("[5] Web应用启动测试")
print("    要启动Web应用，请执行以下操作：")
print("    1. 双击 '一键启动Web应用.bat'")
print("    2. 或运行: python start_web_app.py")
print("    3. 访问: http://localhost:8600")
print()

# 6. 总结
print("=" * 50)
print("📋 测试结果总结")
print("=" * 50)
print()
print("如果所有测试都通过 ✅:")
print("  1. 双击 '一键启动Web应用.bat'")
print("  2. 等待浏览器打开 http://localhost:8600")
print("  3. 点击'启动摄像头'开始检测")
print()
print("如果有 ❌ 错误标记:")
print("  1. 安装缺失的依赖包:")
print("     pip install streamlit opencv-python ultralytics numpy pillow")
print("  2. 确保摄像头连接正常")
print("  3. 参考 'Web应用使用指南.txt'")
print()
print("按 Enter 键退出...")
input()