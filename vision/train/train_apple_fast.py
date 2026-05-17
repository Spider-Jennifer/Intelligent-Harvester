#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速苹果检测模型训练（10个epoch）
使用重新标注的数据集，重点改进特征学习
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("快速苹果检测模型训练（10个epoch）")
    print("使用重新标注的数据集，改进苹果特征学习")
    print("=" * 80)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    # 训练参数 - 快速训练
    training_args = {
        'data': data_yaml,
        'epochs': 10,           # 10个epoch快速训练
        'imgsz': 416,           # 中等分辨率
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_fast_improved',
        'patience': 5,          # 早停
        'save': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.0008,
        'lrf': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0002,
        'warmup_epochs': 1,
        'box': 7.5,
        'cls': 0.6,             # 适中分类损失
        'dfl': 1.5,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 2.0,
        'translate': 0.05,
        'scale': 0.1,
        'shear': 0.2,
        'perspective': 0.0001,
        'fliplr': 0.2,
        'mosaic': 0.1,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'val': True,
        'plots': False,         # 关闭绘图以加快速度
        'verbose': True,
    }
    
    print("\n训练配置:")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    
    # 加载预训练模型
    print("\n加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    print("\n开始快速训练（10个epoch）...")
    print("注意: 训练可能需要15-20分钟")
    print("-" * 80)
    
    try:
        results = model.train(**training_args)
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练错误: {e}")
        return False
    
    # 复制最佳模型
    best_model_path = "runs/detect/apple_fast_improved/weights/best.pt"
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "apple_fast_improved.pt")
        print(f"\n快速改进模型已保存: apple_fast_improved.pt")
    else:
        print(f"警告: 最佳模型文件不存在 {best_model_path}")
    
    print("\n" + "=" * 80)
    print("快速训练完成!")
    print("模型文件: apple_fast_improved.pt")
    print("下一步: 测试新模型性能")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()