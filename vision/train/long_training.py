#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长时间训练脚本 - 专门用于反复训练提高准确率
训练周期长，次数多，反复训练
"""

import os
import time
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

def main():
    print("=" * 70)
    print("长时间训练 - 提高苹果检测模型准确率")
    print("=" * 70)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: Config file not found: {data_yaml}")
        print("Please prepare the dataset first")
        return
    
    print(f"Using dataset config: {data_yaml}")
    
    # 检查数据集图片数量
    train_dir = "apple_dataset/images/train"
    val_dir = "apple_dataset/images/val"
    
    train_images = list(Path(train_dir).glob("*.jpg")) + list(Path(train_dir).glob("*.png"))
    val_images = list(Path(val_dir).glob("*.jpg")) + list(Path(val_dir).glob("*.png"))
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    if len(train_images) < 20:
        print("Warning: Training set has few images, may affect results")
    
    # 选择基础模型
    base_models = ["yolov8n.pt", "apple_best.pt", "apple_sensitive.pt", "apple_improved.pt"]
    available_models = []
    
    for model in base_models:
        if os.path.exists(model):
            available_models.append(model)
            print(f"Found base model: {model}")
    
    if not available_models:
        print("Error: No base models found")
        print("Using default yolov8n.pt")
        base_model = "yolov8n.pt"
    else:
        # 选择最新的模型
        base_model = available_models[-1]
        print(f"Selected base model: {base_model}")
    
    # 训练参数 - 长时间训练配置
    print("\n" + "=" * 70)
    print("Training Configuration - Long Training")
    print("=" * 70)
    
    training_config = {
        'data': data_yaml,
        'epochs': 200,  # 更多轮数
        'patience': 50,  # 更长的早停耐心
        'batch': 16,     # 合适的批次大小
        'imgsz': 640,    # 标准图像大小
        'workers': 4,    # 数据加载工作线程
        'device': '0' if os.path.exists('check_gpu.py') else 'cpu',  # 自动检测GPU
        'project': 'runs/long_training',
        'name': 'apple_long_train',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # 更好的优化器
        'lr0': 0.0001,  # 更小的初始学习率
        'lrf': 0.001,   # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,     # 框损失权重
        'cls': 0.5,     # 分类损失权重
        'dfl': 1.5,     # DFL损失权重
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'save': True,
        'save_period': 10,  # 每10轮保存检查点
        'cache': False,
        'image_weights': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,  # 余弦学习率调度
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # 自动混合精度
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'seed': 42,
        'deterministic': True,
        'verbose': True,
        'plots': True,  # 生成训练图表
        'save_txt': False,
        'save_json': False,
        'save_hybrid': False,
        'conf': 0.25,
        'iou': 0.45,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'vid_stride': 1,
        'line_width': None,
        'visualize': False,
        'augment': True,  # 启用数据增强
        'agnostic_nms': False,
        'retina_masks': False,
        'embed': None,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'save_crop': False,
        'save_dir': None,
        'exist_ok': True
    }
    
    print("Training Parameters:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch']}")
    print(f"  Image size: {training_config['imgsz']}")
    print(f"  Learning rate: {training_config['lr0']}")
    print(f"  Optimizer: {training_config['optimizer']}")
    print(f"  Device: {training_config['device']}")
    print(f"  Save period: Every {training_config['save_period']} epochs")
    
    # 开始训练
    print("\n" + "=" * 70)
    print("Starting Long Training...")
    print("=" * 70)
    print("Note: This will take several hours. Please do not interrupt.")
    print("Training progress will be saved in runs/long_training/")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # 加载模型
        print(f"Loading model: {base_model}")
        model = YOLO(base_model)
        
        # 开始训练
        results = model.train(
            **training_config
        )
        
        # 训练完成
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # 转换为小时
        
        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Total training time: {training_time:.2f} hours")
        
        # 保存最佳模型
        best_model_path = "runs/long_training/apple_long_train/weights/best.pt"
        if os.path.exists(best_model_path):
            # 复制到主目录
            shutil.copy(best_model_path, "apple_long_trained.pt")
            print(f"Best model saved as: apple_long_trained.pt")
            
            # 评估模型
            print("\nEvaluating model...")
            val_results = model.val()
            print(f"mAP50-95: {val_results.box.map:.4f}")
            print(f"mAP50: {val_results.box.map50:.4f}")
            print(f"Precision: {val_results.box.p:.4f}")
            print(f"Recall: {val_results.box.r:.4f}")
        
        print("\n" + "=" * 70)
        print("Next Steps:")
        print("1. Use apple_long_trained.pt in your web app")
        print("2. Test with different confidence thresholds (0.15-0.35)")
        print("3. Check runs/long_training/ for training charts")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 建议后续操作
    print("\n" + "=" * 70)
    print("For even better results:")
    print("1. Add more training images")
    print("2. Use more diverse backgrounds")
    print("3. Consider training for 300+ epochs")
    print("4. Try different model architectures (yolov8m, yolov8l)")
    print("=" * 70)

if __name__ == "__main__":
    main()