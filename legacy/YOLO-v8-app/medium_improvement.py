#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中期改进方案 - 数据增强重新训练
"""

import torch
import shutil
from pathlib import Path

# 修复PyTorch兼容性
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from ultralytics import YOLO

def prepare_enhanced_dataset():
    """准备增强数据集"""
    print("=== 准备增强数据集 ===")
    
    # 创建目录
    dirs = [
        'enhanced_dataset/images/train',
        'enhanced_dataset/images/val',
        'enhanced_dataset/labels/train',
        'enhanced_dataset/labels/val'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 复制现有数据
    source_images = Path('train_dataset/images')
    source_labels = Path('train_dataset/labels')
    
    if source_images.exists():
        # 复制训练图片
        for img_file in (source_images / 'train').glob('*'):
            shutil.copy2(img_file, 'enhanced_dataset/images/train/')
        
        # 复制验证图片
        for img_file in (source_images / 'val').glob('*'):
            shutil.copy2(img_file, 'enhanced_dataset/images/val/')
        
        # 复制标签
        if source_labels.exists():
            shutil.copytree(source_labels / 'train', 'enhanced_dataset/labels/train/', dirs_exist_ok=True)
            shutil.copytree(source_labels / 'val', 'enhanced_dataset/labels/val/', dirs_exist_ok=True)
    
    print("数据集准备完成")

def create_enhanced_config():
    """创建增强训练配置"""
    config = """# 增强苹果检测数据集
path: ./enhanced_dataset
train: images/train
val: images/val

nc: 1
names: ['apple']
"""
    
    with open('enhanced_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(config)
    
    print("创建增强配置文件：enhanced_dataset/data.yaml")

def train_enhanced_model():
    """使用数据增强训练模型"""
    print("\n=== 开始增强训练 ===")
    
    # 升级到YOLOv8s模型（更高精度）
    model = YOLO('yolov8s.pt')  # small版本，比nano更准确
    print("加载YOLOv8s模型（更高精度）...")
    
    # 数据增强配置
    augment_config = {
        'hsv_h': 0.015,        # 色调增强
        'hsv_s': 0.7,          # 饱和度增强
        'hsv_v': 0.4,          # 明度增强
        'degrees': 15,         # 旋转角度
        'translate': 0.1,      # 平移
        'scale': 0.2,          # 缩放
        'shear': 0,            # 剪切
        'perspective': 0,      # 透视
        'flipud': 0,           # 上下翻转
        'fliplr': 0.5,         # 左右翻转
        'mosaic': 1.0,         # 马赛克增强
        'mixup': 0.1,          # 混合增强
    }
    
    # 训练参数
    training_config = {
        'data': 'enhanced_dataset/data.yaml',
        'epochs': 100,              # 增加训练轮数
        'imgsz': 640,
        'batch': 16,                # 增大批次
        'name': 'apple_enhanced',
        'save': True,
        'plots': True,
        'device': 'cpu',
        'patience': 50,             # 早停耐心值
        'lr0': 0.0005,              # 更小学习率
        'optimizer': 'AdamW',       # 更好的优化器
        'warmup_epochs': 10,         # 预热轮数
        'augment': True,             # 开启数据增强
        **augment_config
    }
    
    print("训练配置：")
    print(f"  模型：YOLOv8s（更高精度）")
    print(f"  训练轮数：{training_config['epochs']}")
    print(f"  批次大小：{training_config['batch']}")
    print(f"  学习率：{training_config['lr0']}")
    print(f"  优化器：{training_config['optimizer']}")
    print(f"  数据增强：开启")
    
    # 开始训练
    results = model.train(**training_config)
    
    # 验证模型
    print("\n=== 验证增强模型 ===")
    results = model.val()
    
    # 保存增强模型
    best_model_path = 'runs/detect/apple_enhanced/weights/best.pt'
    if Path(best_model_path).exists():
        # 备份原模型
        if Path('YOLOv8-app-master/weights/best.pt').exists():
            shutil.copy2('YOLOv8-app-master/weights/best.pt', 
                        'YOLOv8-app-master/weights/best_original_backup.pt')
        
        # 复制增强模型
        shutil.copy2(best_model_path, 'YOLOv8-app-master/weights/best_enhanced.pt')
        
        print(f"\n✅ 增强训练完成！")
        print(f"增强模型：YOLOv8-app-master/weights/best_enhanced.pt")
        print(f"原模型备份：YOLOv8-app-master/weights/best_original_backup.pt")
        
        return True
    
    return False

def compare_models():
    """对比模型效果"""
    print("\n=== 模型对比 ===")
    
    if Path('YOLOv8-app-master/weights/best_original_backup.pt').exists():
        print("模型对比：")
        print("  原始模型：YOLOv8n + 基础训练")
        print("  增强模型：YOLOv8s + 数据增强 + 优化训练")
        print("  预期提升：准确率 +20-30%")

def apply_medium_improvements():
    """应用中期改进方案"""
    print("=" * 50)
    print("中期改进方案 - 数据增强重新训练")
    print("=" * 50)
    
    try:
        prepare_enhanced_dataset()
        create_enhanced_config()
        
        print("\n中期改进方案包含：")
        print("[OK] 1. 升级模型：YOLOv8n → YOLOv8s")
        print("[OK] 2. 数据增强：旋转、缩放、颜色变换")
        print("[OK] 3. 训练优化：更多轮数、更小学习率")
        print("[OK] 4. 优化器升级：SGD → AdamW")
        print("[OK] 5. 批次增大：8 → 16")
        
        # 询问是否开始训练
        response = input("\n是否开始增强训练？（需要30-60分钟）(y/n): ").strip().lower()
        if response == 'y':
            if train_enhanced_model():
                compare_models()
                print("\n🎉 中期改进完成！重启应用查看效果")
            else:
                print("训练失败，请检查数据集")
        else:
            print("数据集已准备，可随时运行训练")
            print("命令：python medium_improvement.py")
        
    except Exception as e:
        print(f"过程中出现错误：{e}")

if __name__ == "__main__":
    apply_medium_improvements()