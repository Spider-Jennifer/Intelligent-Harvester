#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门针对减少误识别的苹果检测模型训练
重点：区分苹果与人物、眼镜、头发等
"""

from ultralytics import YOLO
import os
import shutil

print("=" * 80)
print("苹果检测模型训练 - 专门减少误识别")
print("重点：提高模型区分苹果与非苹果物体的能力")
print("=" * 80)

# 检查数据集配置
data_yaml = "apple_dataset/data.yaml"
if not os.path.exists(data_yaml):
    print(f"❌ 错误: 找不到配置文件 {data_yaml}")
    exit(1)

print(f"使用配置文件: {data_yaml}")

# 训练参数 - 专门针对减少误识别优化
training_args = {
    'data': data_yaml,
    'epochs': 50,           # 足够轮次
    'imgsz': 416,           # 中等分辨率
    'batch': 4,
    'workers': 0,
    'device': 'cpu',
    'name': 'apple_no_fp',  # "no false positives"
    'patience': 20,         # 更多耐心
    'save': True,
    'save_period': 10,      # 每10个epoch保存一次
    'pretrained': True,
    'optimizer': 'AdamW',
    'lr0': 0.0003,          # 低学习率
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'box': 7.5,
    'cls': 1.2,             # 提高分类损失权重，增强分类能力
    'dfl': 1.5,
    
    # 数据增强 - 适度，避免生成混淆样本
    'hsv_h': 0.01,
    'hsv_s': 0.5,
    'hsv_v': 0.3,
    'fliplr': 0.3,          # 较低水平的翻转
    'degrees': 5.0,
    'translate': 0.05,
    'scale': 0.1,
    'shear': 0.5,
    'mosaic': 0.2,          # 降低马赛克概率，避免背景混乱
    'mixup': 0.0,           # 禁用mixup，防止类别混淆
    'copy_paste': 0.0,      # 禁用copy-paste
    
    # 其他参数
    'val': True,
    'plots': True,
    'verbose': True,
}

print("\n⚙️  训练配置:")
print(f"  Epochs: {training_args['epochs']}")
print(f"  图像尺寸: {training_args['imgsz']}")
print(f"  分类损失权重: {training_args['cls']} (提高以增强苹果/非苹果区分能力)")
print(f"  数据增强: 适度，避免产生混淆样本")
print(f"  早停耐心: {training_args['patience']}")
print(f"  模型名称: {training_args['name']}")

print("\n🚀 开始训练...")
print("⏱️  预计时间: 40-80分钟 (取决于CPU性能)")
print("   您可以在 runs/detect/apple_no_fp 目录中查看训练进度")
print("-" * 80)

try:
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(**training_args)
    
    print("\n✅ 训练完成!")
    
    # 复制最佳模型到根目录
    best_model_path = f"runs/detect/{training_args['name']}/weights/best.pt"
    if os.path.exists(best_model_path):
        output_model = "apple_no_fp.pt"
        shutil.copy2(best_model_path, output_model)
        print(f"💾 模型已保存: {output_model}")
        
        # 验证模型
        print("\n🔍 验证模型性能...")
        val_results = model.val(data=data_yaml, device='cpu')
        
        if hasattr(val_results, 'box'):
            print(f"📊 验证结果:")
            print(f"  mAP50: {val_results.box.map50:.4f}")
            print(f"  mAP50-95: {val_results.box.map:.4f}")
            print(f"  精确率（越高越好，减少误识别）: {val_results.box.precision.mean():.4f}")
            print(f"  召回率: {val_results.box.recall.mean():.4f}")
            
            if val_results.box.precision.mean() > 0.7:
                print(f"\n🎯 精确率良好，模型应能有效减少误识别!")
            else:
                print(f"\n⚠️  精确率较低，可能仍需改进")
        
        print(f"\n📄 训练曲线图保存于: runs/detect/{training_args['name']}")
        
    else:
        print(f"⚠️  未找到最佳模型文件 {best_model_path}")
        
except KeyboardInterrupt:
    print("\n⏹️  训练被用户中断")
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("训练过程结束")
print("=" * 80)