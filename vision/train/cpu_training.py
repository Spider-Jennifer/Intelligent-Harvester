#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU训练脚本 - 长时间训练提高准确率
"""

import os
import time
from ultralytics import YOLO

print("=" * 70)
print("CPU LONG TRAINING STARTED")
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

# 训练配置 - 针对CPU优化
print()
print("Training Configuration (CPU Optimized):")
print("-" * 50)
print("Epochs: 150")  # CPU训练减少轮数
print("Batch size: 8")  # 更小的批次大小
print("Image size: 640")
print("Learning rate: 0.0001")
print("Optimizer: AdamW")
print("Device: CPU")
print("Workers: 2")  # 减少工作线程
print("Save period: Every 20 epochs")
print("-" * 50)
print()

print("Starting CPU training...")
print("This will take 6-8 hours. PLEASE DO NOT INTERRUPT!")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

start_time = time.time()

try:
    # 加载模型
    model = YOLO(base_model)
    
    # 开始训练 - CPU版本
    results = model.train(
        data='apple_dataset/data.yaml',
        epochs=150,
        patience=40,
        batch=8,  # 更小的批次
        imgsz=640,
        device='cpu',  # 使用CPU
        workers=2,     # 减少工作线程
        project='runs/cpu_long_training',
        name='apple_cpu_long',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,
        cos_lr=True,
        save_period=20,  # 每20轮保存
        plots=True,
        verbose=True,
        augment=True,
        mosaic=0.3,      # 减少增强强度
        degrees=5.0,     # 减少旋转角度
        translate=0.05,  # 减少平移
        scale=0.3,       # 减少缩放
        fliplr=0.3,      # 减少翻转概率
        cache=False,     # 禁用缓存以减少内存
        single_cls=True, # 单类别训练
        rect=False,      # 禁用矩形训练
        amp=False        # 禁用混合精度
    )
    
    # 训练完成
    end_time = time.time()
    training_hours = (end_time - start_time) / 3600
    
    print()
    print("=" * 70)
    print("CPU TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Training time: {training_hours:.2f} hours")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存最佳模型
    best_model_path = "runs/cpu_long_training/apple_cpu_long/weights/best.pt"
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "apple_cpu_trained.pt")
        print(f"Best model saved as: apple_cpu_trained.pt")
        
        # 评估模型
        print("\nEvaluating model...")
        val_results = model.val()
        print(f"Performance metrics:")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
        print(f"  mAP50: {val_results.box.map50:.4f}")
        print(f"  Precision: {val_results.box.p:.4f}")
        print(f"  Recall: {val_results.box.r:.4f}")
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE - NEXT STEPS")
    print("=" * 70)
    print("1. Use apple_cpu_trained.pt in your web application")
    print("2. Test with confidence threshold: 0.15-0.30")
    print("3. Check training results in: runs/cpu_long_training/")
    print("4. For better results, consider:")
    print("   - Adding more training images")
    print("   - Using GPU for faster training")
    print("   - Training for more epochs")
    
except Exception as e:
    print(f"ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试保存部分结果
    print("\nTrying to save partial results...")
    try:
        import glob
        checkpoints = glob.glob("runs/cpu_long_training/apple_cpu_long/weights/*.pt")
        if checkpoints:
            import shutil
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            shutil.copy(latest_checkpoint, "apple_cpu_partial.pt")
            print(f"Partial model saved as: apple_cpu_partial.pt")
    except:
        print("Could not save partial results")

print()
print("=" * 70)
print("Training process completed")
print("=" * 70)

# 等待用户确认
input("\nPress Enter to exit...")