#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终改进的苹果检测模型训练
重点减少误识别，提高苹果特征学习
使用重新标注的数据集（包含空标签作为负样本）
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("最终改进的苹果检测模型训练")
    print("重点: 减少误识别，提高苹果特征学习")
    print("使用重新标注的数据集（包含空标签作为负样本）")
    print("=" * 80)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    # 统计训练集标签
    train_labels_dir = "apple_dataset/labels/train"
    if os.path.exists(train_labels_dir):
        label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        empty_labels = 0
        non_empty_labels = 0
        
        for label_file in label_files:
            with open(os.path.join(train_labels_dir, label_file), 'r') as f:
                content = f.read().strip()
                if content == "":
                    empty_labels += 1
                else:
                    non_empty_labels += 1
        
        print(f"\n训练集标签统计:")
        print(f"  总标签文件: {len(label_files)}")
        print(f"  非空标签（有苹果）: {non_empty_labels}")
        print(f"  空标签（无苹果，负样本）: {empty_labels}")
        print(f"  负样本比例: {empty_labels/len(label_files)*100:.1f}%")
    
    # 训练参数 - 针对低误识别率优化
    training_args = {
        'data': data_yaml,
        'epochs': 30,           # 足够多的epoch
        'imgsz': 640,           # 高分辨率，更好特征学习
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_final_improved',
        'patience': 12,         # 早停耐心
        'save': True,
        'save_period': 10,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.0003,          # 低学习率，稳定训练
        'lrf': 0.001,           # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1.0,             # 提高分类损失权重，减少误识别
        'dfl': 1.5,
        'hsv_h': 0.01,          # 适度颜色增强
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 3.0,         # 适度旋转
        'translate': 0.05,
        'scale': 0.15,
        'shear': 1.0,
        'perspective': 0.0005,
        'flipud': 0.0,
        'fliplr': 0.3,          # 适度水平翻转
        'mosaic': 0.3,          # 适度mosaic
        'mixup': 0.0,           # 禁用mixup，避免类别混淆
        'copy_paste': 0.0,
        'erasing': 0.2,         # 随机擦除，提高鲁棒性
        'close_mosaic': 10,     # 最后10个epoch关闭mosaic
        'val': True,
        'split': 'val',
        'conf': 0.3,            # 验证置信度阈值（较高）
        'iou': 0.5,             # 验证IoU阈值
        'max_det': 50,          # 最大检测数，限制过度检测
        'plots': True,
        'verbose': True,
    }
    
    print("\n训练配置（针对低误识别率优化）:")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  学习率: {training_args['lr0']} (较低)")
    print(f"  分类损失权重: {training_args['cls']} (较高，减少误识别)")
    print(f"  验证置信度阈值: {training_args['conf']} (较高)")
    print(f"  最大检测数: {training_args['max_det']} (限制过度检测)")
    
    # 加载预训练模型
    print("\n加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    print("\n开始最终改进训练...")
    print("注意: 训练可能需要60-90分钟，请耐心等待")
    print("-" * 80)
    
    try:
        results = model.train(**training_args)
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        return False
    
    # 复制最佳模型
    best_model_path = "runs/detect/apple_final_improved/weights/best.pt"
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "apple_final_improved.pt")
        print(f"\n最终改进模型已保存: apple_final_improved.pt")
        
        # 快速测试
        print("\n快速测试最终模型...")
        test_model = YOLO("apple_final_improved.pt")
        
        # 在验证集上测试
        val_images_dir = "apple_dataset/images/val"
        if os.path.exists(val_images_dir):
            val_images = [f for f in os.listdir(val_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
            
            print(f"\n在验证集上测试:")
            for img_name in val_images:
                img_path = os.path.join(val_images_dir, img_name)
                print(f"\n图片: {img_name}")
                
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
        print(f"警告: 最佳模型文件不存在 {best_model_path}")
    
    print("\n" + "=" * 80)
    print("最终改进训练完成!")
    print("模型文件: apple_final_improved.pt")
    print("\n下一步:")
    print("1. 测试模型误识别: python test_misdetection.py")
    print("2. 在苹果照片检测素材上测试: 运行检测脚本")
    print("3. 如果满意，更新应用中使用此模型")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()