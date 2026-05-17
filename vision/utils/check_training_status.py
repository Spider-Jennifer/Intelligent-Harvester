#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查训练状态
"""

import os
import time
import glob

print("=" * 60)
print("Training Status Check")
print("=" * 60)
print()

# 检查训练目录
train_dir = "runs/cpu_long_training/apple_cpu_long"
if os.path.exists(train_dir):
    print(f"[OK] Training directory found: {train_dir}")
    
    # 检查权重文件
    weights_dir = os.path.join(train_dir, "weights")
    if os.path.exists(weights_dir):
        pt_files = glob.glob(os.path.join(weights_dir, "*.pt"))
        if pt_files:
            print(f"[OK] Found {len(pt_files)} model files:")
            for pt_file in sorted(pt_files):
                file_size = os.path.getsize(pt_file) / 1024 / 1024
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(pt_file)))
                filename = os.path.basename(pt_file)
                print(f"   - {filename}: {file_size:.1f} MB (modified: {mtime})")
        else:
            print("[ERROR] No model files found yet")
    else:
        print("[ERROR] Weights directory not created yet")
    
    # 检查结果文件
    results_file = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_file):
        print(f"[OK] Training results file found")
        
        # 读取最后几行
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f"   Total epochs recorded: {len(lines)-1}")
                    # 显示最后一行
                    last_line = lines[-1].strip()
                    print(f"   Latest metrics: {last_line}")
        except:
            print("   Could not read results file")
    else:
        print("❌ Results file not created yet")
        
    # 检查图表
    plot_files = glob.glob(os.path.join(train_dir, "*.png"))
    if plot_files:
        print(f"✅ Found {len(plot_files)} plot files")
    else:
        print("❌ No plot files yet")
        
else:
    print("❌ Training directory not found")
    print("   Training may not have started or is in different location")

print()
print("=" * 60)
print("Training Information")
print("=" * 60)

# 检查基础模型
base_models = ["apple_sensitive.pt", "apple_improved.pt", "apple_best.pt", "yolov8n.pt"]
for model in base_models:
    if os.path.exists(model):
        size = os.path.getsize(model) / 1024 / 1024
        print(f"Base model available: {model} ({size:.1f} MB)")

print()
print("Dataset status:")
if os.path.exists("apple_dataset/data.yaml"):
    print("✅ Dataset configuration found")
    
    # 检查图片数量
    import glob
    train_images = glob.glob("apple_dataset/images/train/*.jpg") + glob.glob("apple_dataset/images/train/*.png")
    val_images = glob.glob("apple_dataset/images/val/*.jpg") + glob.glob("apple_dataset/images/val/*.png")
    
    print(f"   Training images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
else:
    print("❌ Dataset not found")

print()
print("=" * 60)
print("Recommendations")
print("=" * 60)
print("1. DO NOT interrupt the training")
print("2. Let it complete all 150 epochs")
print("3. Check runs/cpu_long_training/ for progress")
print("4. Final model will be: apple_cpu_trained.pt")
print("5. Estimated time remaining: 5-6 hours")

print()
input("Press Enter to exit...")