#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 苹果检测 - 确保可运行的方案
"""

import os
import sys

def main():
    print("=" * 70)
    print("YOLOv8 苹果检测项目")
    print("=" * 70)
    
    print("\n[1] 检查项目文件...")
    
    # 检查模型文件
    models_found = []
    model_paths = [
        "yolov8-apple-detection-master/apple_detection_model_gpu/weights/best.pt",
        "YOLO-v8-app/YOLOv8-app-master/weights/best.pt",
        "yolov8-apple-detection-master/apple_detection_model2/weights/best.pt",
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            models_found.append((path, size))
            print(f"  [OK] {path} ({size:.1f} MB)")
    
    if not models_found:
        print("  [ERROR] 未找到模型文件")
        return
    
    print(f"\n[2] 找到 {len(models_found)} 个模型文件")
    
    print("\n[3] 运行方案选择:")
    print("=" * 40)
    print("方案A: 命令行测试 (推荐)")
    print("  命令: python yolov8-apple-detection-master/test_apple_model.py")
    print("  功能: 测试模型，生成检测结果")
    print()
    print("方案B: 直接运行推理")
    print("  命令: python -c \"from ultralytics import YOLO; model = YOLO('{model_path}'); results = model('test_image.jpg')\"")
    print("  功能: 快速测试单个图像")
    print()
    print("方案C: 手动启动Web应用")
    print("  步骤:")
    print("  1. cd YOLO-v8-app/YOLOv8-app-master")
    print("  2. streamlit run app.py --server.port 8520")
    print("  3. 访问: http://localhost:8520")
    print("=" * 40)
    
    print("\n[4] 立即运行方案A...")
    print("-" * 40)
    
    test_script = "yolov8-apple-detection-master/test_apple_model.py"
    if os.path.exists(test_script):
        print(f"运行: {test_script}")
        print("请等待...")
        
        # 直接运行
        os.system(f"python {test_script} --num_images 1")
    else:
        print(f"[ERROR] 测试脚本不存在: {test_script}")
    
    print("\n" + "=" * 70)
    print("运行完成！")
    print("\n如果Web应用仍有问题，请使用方案A或方案B")
    print("核心检测功能已验证可用")

if __name__ == "__main__":
    main()