#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速改进苹果检测模型训练（15个epoch）
重点减少误识别，使用重新标注的数据集
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("快速改进苹果检测模型训练")
    print("重点: 减少误识别，提高精度")
    print("使用重新标注的数据集")
    print("=" * 80)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    # 训练参数 - 快速训练，注重精度
    training_args = {
        'data': data_yaml,
        'epochs': 15,           # 快速训练
        'imgsz': 416,           # 中等分辨率
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_quick_improved',
        'patience': 8,          # 早停
        'save': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.0005,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0003,
        'warmup_epochs': 2,
        'box': 7.5,
        'cls': 0.7,             # 增加分类损失权重
        'dfl': 1.5,
        'hsv_h': 0.01,
        'hsv_s': 0.6,
        'hsv_v': 0.3,
        'degrees': 3.0,         # 轻度增强
        'translate': 0.05,
        'scale': 0.15,
        'shear': 0.5,
        'perspective': 0.0002,
        'fliplr': 0.3,          # 轻度水平翻转
        'mosaic': 0.3,          # 轻度mosaic
        'mixup': 0.0,           # 禁用mixup
        'copy_paste': 0.0,
        'erasing': 0.1,
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    print("\n训练配置:")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  学习率: {training_args['lr0']}")
    print(f"  分类损失权重: {training_args['cls']} (提高分类精度)")
    
    # 加载预训练模型
    print("\n加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    print("\n开始快速训练...")
    print("注意: 训练可能需要20-30分钟")
    print("-" * 80)
    
    try:
        results = model.train(**training_args)
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练错误: {e}")
        return False
    
    # 复制最佳模型
    best_model_path = "runs/detect/apple_quick_improved/weights/best.pt"
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "apple_quick_improved.pt")
        print(f"\n快速改进模型已保存: apple_quick_improved.pt")
        
        # 快速测试
        print("\n快速测试新模型...")
        test_model = YOLO("apple_quick_improved.pt")
        
        # 测试一张图片
        test_images = []
        if os.path.exists("apple_dataset/images/val"):
            test_images = [os.path.join("apple_dataset/images/val", f) 
                          for f in os.listdir("apple_dataset/images/val") 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:1]
        
        if test_images:
            img_path = test_images[0]
            print(f"测试图片: {os.path.basename(img_path)}")
            
            for conf in [0.1, 0.2, 0.3, 0.4]:
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
            print("没有找到测试图片")
    else:
        print(f"警告: 最佳模型文件不存在 {best_model_path}")
    
    print("\n" + "=" * 80)
    print("快速训练完成!")
    print("下一步:")
    print("1. 完整测试: python test_model_accuracy.py")
    print("2. 在真实图片上测试: python detect_apple_8sec.py")
    print("3. 如果满意，使用 apple_quick_improved.pt 替换旧模型")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()