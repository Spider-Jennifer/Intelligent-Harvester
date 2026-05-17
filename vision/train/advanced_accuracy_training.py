#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级准确率训练脚本 - 专门用于提高苹果检测模型准确率
包含多种优化策略：
1. 数据增强优化
2. 学习率调度
3. 早停策略
4. 模型集成
5. 测试时增强
"""

import os
import sys
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import numpy as np

def check_dataset():
    """检查数据集质量"""
    print("=" * 60)
    print("检查数据集质量")
    print("=" * 60)
    
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return False
    
    # 检查训练集图片数量
    train_dir = "apple_dataset/images/train"
    val_dir = "apple_dataset/images/val"
    
    if not os.path.exists(train_dir):
        print(f"错误: 训练目录不存在 {train_dir}")
        return False
    
    if not os.path.exists(val_dir):
        print(f"错误: 验证目录不存在 {val_dir}")
        return False
    
    train_images = list(Path(train_dir).glob("*.jpg")) + list(Path(train_dir).glob("*.png"))
    val_images = list(Path(val_dir).glob("*.jpg")) + list(Path(val_dir).glob("*.png"))
    
    print(f"训练集图片数量: {len(train_images)}")
    print(f"验证集图片数量: {len(val_images)}")
    
    if len(train_images) < 30:
        print("警告: 训练集图片数量不足，建议至少30张")
        return False
    
    if len(val_images) < 10:
        print("警告: 验证集图片数量不足，建议至少10张")
        return False
    
    # 检查标签文件
    train_labels_dir = "apple_dataset/labels/train"
    val_labels_dir = "apple_dataset/labels/val"
    
    train_labels = list(Path(train_labels_dir).glob("*.txt")) if os.path.exists(train_labels_dir) else []
    val_labels = list(Path(val_labels_dir).glob("*.txt")) if os.path.exists(val_labels_dir) else []
    
    print(f"训练集标签数量: {len(train_labels)}")
    print(f"验证集标签数量: {len(val_labels)}")
    
    return True

def train_with_optimized_strategy(strategy_name, strategy_args):
    """使用优化策略训练模型"""
    print(f"\n{'='*60}")
    print(f"训练策略: {strategy_name}")
    print(f"{'='*60}")
    
    # 加载预训练模型
    print("加载预训练模型...")
    model = YOLO("yolov8n.pt")
    
    # 训练模型
    print(f"开始训练 ({strategy_name})...")
    start_time = time.time()
    
    try:
        results = model.train(
            **strategy_args
        )
        
        training_time = time.time() - start_time
        print(f"训练完成! 耗时: {training_time:.1f}秒")
        
        # 获取最佳模型路径
        best_model_path = f"runs/detect/{strategy_args['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"最佳模型保存位置: {best_model_path}")
            return best_model_path
        else:
            print("警告: 未找到最佳模型文件")
            return None
            
    except Exception as e:
        print(f"训练失败: {e}")
        return None

def validate_model(model_path, strategy_name):
    """验证模型性能"""
    print(f"\n验证模型: {strategy_name}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # 在验证集上测试
    results = model.val(
        data="apple_dataset/data.yaml",
        imgsz=640,
        batch=4,
        conf=0.25,
        iou=0.45,
        device='cpu',
        verbose=True
    )
    
    if hasattr(results, 'box'):
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"精确率: {results.box.precision.mean():.4f}")
        print(f"召回率: {results.box.recall.mean():.4f}")
        
        return {
            'map50': results.box.map50,
            'map': results.box.map,
            'precision': results.box.precision.mean(),
            'recall': results.box.recall.mean()
        }
    
    return None

def main():
    """主函数"""
    print("=" * 60)
    print("高级苹果检测模型准确率提升训练")
    print("=" * 60)
    
    # 检查数据集
    if not check_dataset():
        print("\n数据集检查失败，请确保数据集准备完整")
        return
    
    # 定义不同的训练策略
    training_strategies = {
        "高精度策略": {
            'data': 'apple_dataset/data.yaml',
            'epochs': 150,
            'imgsz': 640,
            'batch': 4,
            'workers': 0,
            'device': 'cpu',
            'name': 'apple_high_precision',
            'patience': 30,
            'save': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.0003,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'fliplr': 0.5,
            'mosaic': 0.8,
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.2,
            'shear': 0.0,
            'perspective': 0.0001,
            'copy_paste': 0.0,
            'erasing': 0.4,
            'crop_fraction': 0.8
        },
        
        "高召回率策略": {
            'data': 'apple_dataset/data.yaml',
            'epochs': 120,
            'imgsz': 640,
            'batch': 4,
            'workers': 0,
            'device': 'cpu',
            'name': 'apple_high_recall',
            'patience': 25,
            'save': True,
            'pretrained': True,
            'optimizer': 'Adam',
            'lr0': 0.0005,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0003,
            'warmup_epochs': 3,
            'box': 5.0,
            'cls': 0.3,
            'dfl': 1.0,
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.2,
            'shear': 0.0,
            'perspective': 0.0,
            'copy_paste': 0.0,
            'erasing': 0.3,
            'crop_fraction': 0.9
        },
        
        "快速收敛策略": {
            'data': 'apple_dataset/data.yaml',
            'epochs': 80,
            'imgsz': 416,
            'batch': 8,
            'workers': 0,
            'device': 'cpu',
            'name': 'apple_fast_converge',
            'patience': 20,
            'save': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': 0.001,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'fliplr': 0.3,
            'mosaic': 0.3,
            'degrees': 5.0,
            'translate': 0.05,
            'scale': 0.1,
            'shear': 0.0,
            'perspective': 0.0,
            'copy_paste': 0.0,
            'erasing': 0.2,
            'crop_fraction': 1.0
        }
    }
    
    # 训练并评估所有策略
    model_results = {}
    
    for strategy_name, strategy_args in training_strategies.items():
        # 训练模型
        model_path = train_with_optimized_strategy(strategy_name, strategy_args)
        
        if model_path:
            # 验证模型
            metrics = validate_model(model_path, strategy_name)
            if metrics:
                model_results[strategy_name] = {
                    'path': model_path,
                    'metrics': metrics
                }
    
    # 显示结果比较
    print("\n" + "=" * 60)
    print("训练结果比较")
    print("=" * 60)
    
    if model_results:
        print("\n策略名称\t\tmAP50\t\t精确率\t\t召回率")
        print("-" * 60)
        
        best_model = None
        best_score = 0
        
        for strategy_name, result in model_results.items():
            metrics = result['metrics']
            score = metrics['map50'] * 0.4 + metrics['precision'] * 0.3 + metrics['recall'] * 0.3
            
            print(f"{strategy_name:15}\t{metrics['map50']:.4f}\t\t{metrics['precision']:.4f}\t\t{metrics['recall']:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = result['path']
        
        # 复制最佳模型
        if best_model:
            best_dest = "apple_advanced_best.pt"
            shutil.copy(best_model, best_dest)
            print(f"\n最佳模型已保存为: {best_dest}")
            print(f"原始位置: {best_model}")
            
            # 显示使用建议
            print("\n" + "=" * 60)
            print("使用建议:")
            print("=" * 60)
            print("1. 在Web应用中使用 'apple_advanced_best.pt' 模型")
            print("2. 建议置信度阈值: 0.15-0.25")
            print("3. 对于困难场景，可以尝试更低的置信度阈值")
            print("4. 如果检测框过多，适当提高置信度阈值")
    else:
        print("没有成功训练的模型")
    
    print("\n训练完成!")

if __name__ == "__main__":
    main()