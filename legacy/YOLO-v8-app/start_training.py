#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动微调训练 - 交互式版本
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

def prepare_dataset(images_folder):
    """准备数据集"""
    print("=== 准备训练数据集 ===")
    
    # 创建目录
    dirs = [
        'train_dataset/images/train',
        'train_dataset/images/val', 
        'train_dataset/labels/train',
        'train_dataset/labels/val'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 获取图片
    image_folder = Path(images_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 分割数据集
    train_count = int(len(image_files) * 0.8)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # 复制图片
    for i, img_file in enumerate(train_files):
        dest = f"train_dataset/images/train/apple_{i:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)
    
    for i, img_file in enumerate(val_files):
        dest = f"train_dataset/images/val/apple_{i:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)
    
    print(f"训练集：{len(train_files)} 张")
    print(f"验证集：{len(val_files)} 张")
    
    return train_files, val_files

def generate_labels(train_files, val_files):
    """生成标签"""
    print("\n=== 生成训练标签 ===")
    
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    
    def create_labels(images, label_dir):
        label_path = Path(label_dir)
        label_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in images:
            results = model.predict(str(img_file), conf=0.25)
            
            label_file = label_path / f"{img_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = box.conf[0].cpu().numpy()
                            if conf > 0.3:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                img_w, img_h = result.orig_shape
                                
                                x_center = (x1 + x2) / 2 / img_w
                                y_center = (y1 + y2) / 2 / img_h
                                width = (x2 - x1) / img_w
                                height = (y2 - y1) / img_h
                                
                                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    create_labels(train_files, 'train_dataset/labels/train')
    create_labels(val_files, 'train_dataset/labels/val')
    print("标签生成完成")

def create_config():
    """创建配置文件"""
    import os
    current_dir = os.getcwd()
    config = f"""# 苹果检测训练数据集
path: {current_dir}/train_dataset
train: images/train
val: images/val

nc: 1
names: ['apple']
"""
    
    with open('train_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(config)
    print("创建配置文件：train_dataset/data.yaml")

def start_training():
    """开始训练"""
    print("\n=== 开始微调训练 ===")
    
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    print("加载现有权重...")
    
    results = model.train(
        data='train_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='apple_custom_fine_tune',
        save=True,
        plots=True,
        device='cpu',
        patience=20,
        lr0=0.001,
        optimizer='Adam',
        augment=True,
        pretrained=True,
    )
    
    print("\n=== 验证模型 ===")
    results = model.val()
    
    # 保存新权重
    best_model_path = 'runs/detect/apple_custom_fine_tune/weights/best.pt'
    if Path(best_model_path).exists():
        shutil.copy2('YOLOv8-app-master/weights/best.pt', 
                    'YOLOv8-app-master/weights/best_backup_before_custom.pt')
        shutil.copy2(best_model_path, 'YOLOv8-app-master/weights/best.pt')
        
        print("\n✅ 训练完成！")
        print(f"新权重：YOLOv8-app-master/weights/best.pt")
        print(f"备份：YOLOv8-app-master/weights/best_backup_before_custom.pt")

def main():
    print("=" * 50)
    print("YOLOv8 苹果检测 - 微调训练")
    print("=" * 50)
    
    images_folder = input("\n请输入苹果图片文件夹路径：").strip()
    
    if not images_folder or not Path(images_folder).exists():
        print("错误：文件夹不存在")
        return
    
    try:
        train_files, val_files = prepare_dataset(images_folder)
        generate_labels(train_files, val_files)
        create_config()
        start_training()
        
        print("\n" + "=" * 50)
        print("微调训练完成！")
        print("重启应用查看效果：")
        print("cd YOLOv8-app-master && streamlit run app.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中出现错误：{e}")

if __name__ == "__main__":
    main()