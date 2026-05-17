#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
立即开始训练 - 长时间训练提高准确率
"""

import os
import time
from ultralytics import YOLO

print("=" * 70)
print("STARTING LONG TRAINING NOW")
print("=" * 70)
print()

# 检查数据集
if not os.path.exists("apple_dataset/data.yaml"):
    print("ERROR: Dataset not found")
    exit(1)

print("Dataset found: apple_dataset/data.yaml")

# 选择最新模型作为基础
models = ["apple_sensitive.pt", "apple_improved.pt", "apple_best.pt", "yolov8n.pt"]
base_model = None

for model in models:
    if os.path.exists(model):
        base_model = model
        print(f"Using base model: {base_model}")
        break

if not base_model:
    print("ERROR: No model found")
    exit(1)

# 训练配置
print()
print("Training Configuration:")
print("-" * 40)
print("Epochs: 200")
print("Batch size: 16")
print("Image size: 640")
print("Learning rate: 0.0001")
print("Optimizer: AdamW")
print("Save period: Every 10 epochs")
print("-" * 40)
print()

print("IMPORTANT: This will take several hours!")
print("Do NOT interrupt the training!")
print()

# 确认开始
response = input("Start training now? (y/n): ").strip().lower()
if response != 'y':
    print("Training cancelled")
    exit(0)

print()
print("=" * 70)
print("STARTING TRAINING...")
print("=" * 70)
print()

start_time = time.time()

try:
    # 加载模型
    model = YOLO(base_model)
    
    # 开始训练
    results = model.train(
        data='apple_dataset/data.yaml',
        epochs=200,
        patience=50,
        batch=16,
        imgsz=640,
        device='0',  # 使用GPU
        project='runs/long_training_now',
        name='apple_long',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,
        cos_lr=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    # 训练完成
    end_time = time.time()
    training_hours = (end_time - start_time) / 3600
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Training time: {training_hours:.2f} hours")
    
    # 保存最佳模型
    best_model_path = "runs/long_training_now/apple_long/weights/best.pt"
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "apple_long_trained.pt")
        print(f"Best model saved as: apple_long_trained.pt")
    
    print()
    print("Next steps:")
    print("1. Use apple_long_trained.pt in your web app")
    print("2. Test with confidence 0.15-0.35")
    print("3. Check runs/long_training_now/ for results")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
input("Press Enter to exit...")