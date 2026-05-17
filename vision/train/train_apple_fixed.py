#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
苹果检测模型训练 - 修复版
使用绝对路径避免路径问题
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_apple_model():
    """训练苹果检测模型"""
    print("=" * 60)
    print("苹果检测模型训练 - 开始")
    print("=" * 60)
    
    # 使用修复的配置文件
    data_yaml = "dataset_fixed.yaml"
    
    # 检查预训练模型
    pretrained_model = "yolov8n.pt"
    
    print(f"数据集配置文件: {data_yaml}")
    print(f"预训练模型: {pretrained_model}")
    print()
    
    try:
        # 加载预训练模型
        print("正在加载YOLOv8n预训练模型...")
        model = YOLO(pretrained_model)
        
        # 训练参数配置 - 简化版，适合快速训练
        print("开始训练模型...")
        print("训练参数:")
        print("  - 训练轮数: 20 (快速训练)")
        print("  - 批次大小: 4 (适合CPU训练)")
        print("  - 图像尺寸: 320 (降低分辨率，加快训练)")
        print("  - 设备: CPU")
        print()
        
        # 开始训练
        results = model.train(
            data=data_yaml,           # 数据集配置文件
            epochs=20,                # 训练轮数
            imgsz=320,                # 图像尺寸
            batch=4,                  # 批次大小
            name='apple_detection',   # 实验名称
            save=True,                # 保存检查点
            plots=True,               # 生成图表
            device='cpu',             # 使用CPU训练
            patience=10,              # 早停耐心值
            lr0=0.01,                 # 学习率
            verbose=True              # 显示训练进度
        )
        
        print()
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        # 显示训练结果保存位置
        runs_dir = Path("runs/detect/apple_detection")
        if runs_dir.exists():
            print(f"训练结果保存在: {runs_dir}")
            
            # 检查生成的模型文件
            weights_dir = runs_dir / "weights"
            if weights_dir.exists():
                best_model = weights_dir / "best.pt"
                last_model = weights_dir / "last.pt"
                
                if best_model.exists():
                    print(f"最佳模型: {best_model}")
                    print(f"文件大小: {best_model.stat().st_size / 1024 / 1024:.2f} MB")
                    
                    # 复制到项目目录
                    import shutil
                    target_path = "apple_best.pt"
                    shutil.copy2(best_model, target_path)
                    print(f"已复制到: {target_path}")
        
        print()
        print("训练说明:")
        print("由于标签文件较少，模型主要学习苹果的一般特征")
        print("实际检测效果可能需要进一步优化")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        print()
        print("备选方案: 直接使用预训练模型")
        print("YOLOv8n模型已经具备物体检测能力，可以直接使用")

if __name__ == "__main__":
    train_apple_model()