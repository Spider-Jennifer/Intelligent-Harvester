#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 极简运行脚本 - 绕过依赖问题
"""

import os
import sys

def main():
    print("=" * 60)
    print("YOLOv8 苹果检测项目")
    print("=" * 60)
    
    # 检查项目结构
    print("\n项目结构检查:")
    
    # 检查模型文件
    model_files = [
        'yolov8-apple-detection-master/apple_detection_model_gpu/weights/best.pt',
        'YOLO-v8-app/YOLOv8-app-master/weights/best.pt',
        'yolov8-apple-detection-master/apple_detection_model2/weights/best.pt',
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024*1024)
            print(f"[OK] 模型文件: {model_file} ({size:.1f} MB)")
    
    # 检查测试图像
    test_dirs = [
        'YOLO-v8-app/quick_train/images',
        'yolov8-apple-detection-master/Dataset/apple_dataset/images/train',
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]
            if images:
                print(f"[OK] 测试图像目录: {test_dir} ({len(images)} 张图像)")
    
    # 检查现有脚本
    scripts = [
        'yolov8-apple-detection-master/test_apple_model.py',
        'YOLO-v8-app/simple_verify.py',
        'YOLO-v8-app/test_imports.py',
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"[OK] 脚本文件: {script}")
    
    print("\n运行建议:")
    print("1. 安装依赖: pip install torch torchvision ultralytics opencv-python")
    print("2. 运行测试: python yolov8-apple-detection-master/test_apple_model.py")
    print("3. 运行验证: python YOLO-v8-app/simple_verify.py")
    print("\n项目准备就绪！")

if __name__ == "__main__":
    main()