#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于过滤数据集的训练脚本
"""

from ultralytics import YOLO
import shutil
from pathlib import Path

def train_with_filtered_data():
    """使用过滤后的数据集进行训练"""
    
    print("=== 使用过滤数据集进行微调训练 ===")
    
    # 检查数据集是否存在
    if not Path('filtered_dataset/data.yaml').exists():
        print("错误：过滤数据集不存在，请先运行 smart_dataset_filter.py")
        return False
    
    # 加载现有权重
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    print("加载现有权重进行微调...")
    
    # 训练
    results = model.train(
        data='filtered_dataset/data.yaml',
        epochs=60,              # 过滤后的数据质量高，可以多训练几轮
        imgsz=640,
        batch=8,
        name='apple_filtered_fine_tune',
        save=True,
        plots=True,
        device='cpu',
        patience=25,
        lr0=0.001,
        optimizer='Adam',
        augment=True,
        pretrained=True,
    )
    
    # 验证
    print("\n=== 验证模型 ===")
    results = model.val()
    
    # 保存新权重
    best_model_path = 'runs/detect/apple_filtered_fine_tune/weights/best.pt'
    if Path(best_model_path).exists():
        # 备份原权重
        shutil.copy2('YOLOv8-app-master/weights/best.pt', 
                    'YOLOv8-app-master/weights/best_before_filter_backup.pt')
        
        # 替换为新权重
        shutil.copy2(best_model_path, 'YOLOv8-app-master/weights/best.pt')
        
        print(f"\n✅ 训练完成！")
        print(f"新权重：YOLOv8-app-master/weights/best.pt")
        print(f"备份：YOLOv8-app-master/weights/best_before_filter_backup.pt")
    
    return True

if __name__ == "__main__":
    train_with_filtered_data()