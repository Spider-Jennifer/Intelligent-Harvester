#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

print("测试导入...")
try:
    from ultralytics import YOLO
    print("✅ ultralytics 导入成功")
    
    # 测试路径
    test_path = r"C:\Users\李晨鑫\Desktop\yolov8\yolov8n.pt"
    print(f"测试路径: {test_path}")
    print(f"文件存在: {os.path.exists(test_path)}")
    
    # 检查图片目录
    apple_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    print(f"苹果照片目录: {apple_dir}")
    print(f"目录存在: {os.path.exists(apple_dir)}")
    
    if os.path.exists(apple_dir):
        import glob
        images = glob.glob(os.path.join(apple_dir, "*.jpg"))
        print(f"找到 {len(images)} 张JPG图片")
        for img in images[:3]:
            print(f"  - {os.path.basename(img)}")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 错误: {e}")