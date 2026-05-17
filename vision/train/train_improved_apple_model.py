#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的苹果检测模型训练脚本
重点提高精度，减少误识别（如将人识别为苹果）
使用重新标注的数据集
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("改进的苹果检测模型训练")
    print("重点: 提高检测精度，减少误识别")
    print("=" * 80)
    
    # 检查数据集配置
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    # 检查标签文件是否存在
    train_labels_dir = "apple_dataset/labels/train"
    if not os.path.exists(train_labels_dir):
        print(f"错误: 标签目录不存在 {train_labels_dir}")
        print("请先运行 relabel_apple_dataset.py 进行自动标注")
        return
    
    # 统计标签文件
    label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    print(f"训练集标签文件数量: {len(label_files)}")
    
    # 检查标签内容
    empty_labels = 0
    non_empty_labels = 0
    for label_file in label_files[:5]:  # 检查前5个文件
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            content = f.read().strip()
            if content == "":
                empty_labels += 1
            else:
                non_empty_labels += 1
    
    print(f"样本检查: 空标签 {empty_labels}, 非空标签 {non_empty_labels}")
    
    # 训练参数 - 针对高精度优化
    training_args = {
        'data': data_yaml,
        'epochs': 50,           # 更多epoch以获得更好精度
        'imgsz': 640,           # 更高分辨率，提高特征提取能力
        'batch': 4,             # 较小批次，在CPU上更稳定
        'workers': 0,           # CPU训练，避免多线程问题
        'device': 'cpu',        # 使用CPU训练
        'name': 'apple_high_precision',  # 新模型名称
        'patience': 15,         # 早停耐心，防止过拟合
        'save': True,
        'save_period': 10,      # 每10个epoch保存一次
        'pretrained': True,     # 使用预训练权重
        'optimizer': 'AdamW',   # 更好的优化器
        'lr0': 0.0005,          # 较低学习率，更稳定训练
        'lrf': 0.005,           # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,             # 框损失权重
        'cls': 0.8,             # 增加分类损失权重，提高分类精度
        'dfl': 1.5,
        'hsv_h': 0.015,         # 适度的颜色增强
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,         # 适度的旋转增强
        'translate': 0.1,
        'scale': 0.2,           # 适度的缩放增强
        'shear': 1.0,
        'perspective': 0.0005,
        'flipud': 0.0,
        'fliplr': 0.5,          # 水平翻转
        'mosaic': 0.5,          # 降低mosaic概率，避免复杂背景
        'mixup': 0.0,           # 禁用mixup，避免类别混淆
        'copy_paste': 0.0,      # 禁用copy-paste
        'erasing': 0.2,         # 随机擦除，提高鲁棒性
        'crop_fraction': 0.9,   # 保留大部分图像内容
        'close_mosaic': 10,     # 最后10个epoch关闭mosaic
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'conf': 0.25,           # 验证置信度阈值
        'iou': 0.6,             # 验证IoU阈值
        'max_det': 100,
        'half': False,
        'dnn': False,
        'plots': True,          # 生成训练曲线图
        'verbose': True,        # 显示详细训练信息
    }
    
    print("\n训练配置:")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  学习率: {training_args['lr0']}")
    print(f"  分类损失权重: {training_args['cls']} (提高以增强分类精度)")
    print(f"  早停耐心: {training_args['patience']}")
    print(f"  数据增强: 适度增强，避免过度混淆")
    
    # 加载预训练模型
    print("\n加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    print("\n开始训练改进的苹果检测模型...")
    print("注意: 训练可能需要60-90分钟，请耐心等待")
    print("-" * 80)
    
    try:
        results = model.train(**training_args)
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("尝试使用简化配置重新训练...")
        
        # 简化配置
        simplified_args = training_args.copy()
        simplified_args['epochs'] = 30
        simplified_args['imgsz'] = 416
        simplified_args['mosaic'] = 0.0
        simplified_args['fliplr'] = 0.0
        
        print("使用简化配置重新训练...")
        results = model.train(**simplified_args)
    
    # 复制最佳模型到根目录
    best_model_path = "runs/detect/apple_high_precision/weights/best.pt"
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "apple_high_precision.pt")
        print(f"\n改进模型已保存: apple_high_precision.pt")
        
        # 测试新模型
        print("\n测试改进模型...")
        test_model = YOLO("apple_high_precision.pt")
        
        # 使用验证集图片测试
        val_images_dir = "apple_dataset/images/val"
        if os.path.exists(val_images_dir):
            val_images = [f for f in os.listdir(val_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]  # 测试前3张
            
            print(f"\n在 {len(val_images)} 张验证集图片上测试:")
            
            for img_name in val_images:
                img_path = os.path.join(val_images_dir, img_name)
                print(f"\n图片: {img_name}")
                
                # 在不同置信度下测试
                for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    test_results = test_model(img_path, conf=conf, verbose=False)
                    
                    if test_results and len(test_results) > 0:
                        result = test_results[0]
                        if result.boxes is not None:
                            detections = len(result.boxes)
                            print(f"  置信度 {conf}: 检测到 {detections} 个苹果")
                        else:
                            print(f"  置信度 {conf}: 未检测到苹果")
                    else:
                        print(f"  置信度 {conf}: 检测失败")
        else:
            print("警告: 验证集目录不存在，跳过测试")
    else:
        print(f"警告: 最佳模型文件不存在 {best_model_path}")
    
    print("\n" + "=" * 80)
    print("训练流程完成!")
    print("下一步建议:")
    print("1. 测试模型性能: python test_model_accuracy.py compare")
    print("2. 在真实图片上测试: python test_apple_quick.py")
    print("3. 更新应用中的模型路径为 apple_high_precision.pt")
    print("4. 如果仍有误识别，考虑添加负样本图像进行训练")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()