#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动开始训练 - 无需确认
"""

import os
import time
from ultralytics import YOLO

print("=" * 70)
print("AUTO STARTING LONG TRAINING")
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
print("Epochs: 250")  # 增加到250轮
print("Batch size: 16")
print("Image size: 640")
print("Learning rate: 0.00008")  # 更小的学习率
print("Optimizer: AdamW")
print("Save period: Every 15 epochs")
print("-" * 40)
print()

print("Starting training automatically...")
print("This will take 4-6 hours. DO NOT INTERRUPT!")
print()

start_time = time.time()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    # 加载模型
    model = YOLO(base_model)
    
    # 开始训练 - 使用更长的训练配置
    results = model.train(
        data='apple_dataset/data.yaml',
        epochs=250,  # 更多轮数
        patience=60,  # 更长的早停耐心
        batch=16,
        imgsz=640,
        device='0',  # 使用GPU
        project='runs/auto_long_training',
        name='apple_auto_long',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.00008,  # 更小的学习率
        lrf=0.0005,   # 最终学习率
        cos_lr=True,
        save_period=15,  # 每15轮保存
        plots=True,
        verbose=True,
        augment=True,    # 数据增强
        mosaic=0.5,      # 马赛克增强
        mixup=0.1,       # MixUp增强
        copy_paste=0.1,  # 复制粘贴增强
        hsv_h=0.015,     # 色调增强
        hsv_s=0.7,       # 饱和度增强
        hsv_v=0.4,       # 明度增强
        degrees=10.0,    # 旋转角度
        translate=0.1,   # 平移
        scale=0.5,       # 缩放
        shear=2.0,       # 剪切
        perspective=0.0, # 透视
        flipud=0.0,      # 上下翻转
        fliplr=0.5,      # 左右翻转
        bgr=False,       # BGR转换
        auto_augment='randaugment',  # 自动增强
        erasing=0.4,     # 随机擦除
        crop_fraction=1.0 # 裁剪比例
    )
    
    # 训练完成
    end_time = time.time()
    training_hours = (end_time - start_time) / 3600
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Training time: {training_hours:.2f} hours")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存最佳模型
    best_model_path = "runs/auto_long_training/apple_auto_long/weights/best.pt"
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "apple_auto_long_trained.pt")
        print(f"Best model saved as: apple_auto_long_trained.pt")
        
        # 评估模型
        print("\nEvaluating model...")
        val_results = model.val()
        print(f"mAP50-95: {val_results.box.map:.4f}")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"Precision: {val_results.box.p:.4f}")
        print(f"Recall: {val_results.box.r:.4f}")
    
    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs: 250")
    print(f"Base model: {base_model}")
    print(f"Final model: apple_auto_long_trained.pt")
    print(f"Results saved in: runs/auto_long_training/")
    print()
    print("Next steps:")
    print("1. Use apple_auto_long_trained.pt in your web app")
    print("2. Test with confidence thresholds 0.15-0.35")
    print("3. Check training charts in runs/ folder")
    
except Exception as e:
    print(f"ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试保存部分结果
    print("\nTrying to save partial results...")
    try:
        import glob
        checkpoints = glob.glob("runs/auto_long_training/apple_auto_long/weights/*.pt")
        if checkpoints:
            import shutil
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            shutil.copy(latest_checkpoint, "apple_partial_trained.pt")
            print(f"Partial model saved as: apple_partial_trained.pt")
    except:
        print("Could not save partial results")

print()
print("=" * 70)
print("Training process finished")
print("=" * 70)