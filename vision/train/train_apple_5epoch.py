#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超快速苹果检测模型训练（5个epoch）
用于验证重新标注数据的有效性
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("超快速苹果检测模型训练（5个epoch）")
    print("验证重新标注数据的有效性")
    print("=" * 80)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    # 超快速训练参数
    training_args = {
        'data': data_yaml,
        'epochs': 5,            # 仅5个epoch
        'imgsz': 320,           # 低分辨率，加快训练
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_5epoch_test',
        'patience': 3,          # 早停
        'save': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.05,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'warmup_epochs': 0,     # 无预热
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.0,           # 无增强
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'val': True,
        'plots': False,
        'verbose': True,
    }
    
    print("\n超快速训练配置（验证用途）")
    
    # 加载预训练模型
    print("\n加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    print("\n开始超快速训练（5个epoch）...")
    print("注意: 训练可能需要5-10分钟")
    print("-" * 80)
    
    try:
        results = model.train(**training_args)
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练错误: {e}")
        return False
    
    # 复制最佳模型
    best_model_path = "runs/detect/apple_5epoch_test/weights/best.pt"
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "apple_5epoch_test.pt")
        print(f"\n测试模型已保存: apple_5epoch_test.pt")
        
        # 快速验证
        print("\n快速验证新模型...")
        test_model = YOLO("apple_5epoch_test.pt")
        
        # 测试一张图片
        test_images = []
        if os.path.exists("apple_dataset/images/val"):
            test_images = [os.path.join("apple_dataset/images/val", f) 
                          for f in os.listdir("apple_dataset/images/val") 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:2]
        
        for img_path in test_images:
            print(f"\n测试图片: {os.path.basename(img_path)}")
            for conf in [0.1, 0.2, 0.3]:
                results = test_model(img_path, conf=conf, verbose=False)
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        print(f"  置信度 {conf}: 检测到 {len(result.boxes)} 个苹果")
                    else:
                        print(f"  置信度 {conf}: 未检测到苹果")
                else:
                    print(f"  置信度 {conf}: 检测失败")
    else:
        print(f"警告: 最佳模型文件不存在 {best_model_path}")
    
    print("\n" + "=" * 80)
    print("超快速训练完成!")
    print("此模型仅用于验证重新标注数据的有效性")
    print("对于生产使用，建议训练更多epoch（至少30个）")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()