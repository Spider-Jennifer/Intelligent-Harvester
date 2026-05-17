#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8苹果检测项目 - 微调训练脚本
功能：基于现有苹果检测权重，使用用户的新图片进行微调训练
作者：AI Assistant
版本：1.0
使用方法：python fine_tune_model.py
前置条件：需要现有的best.pt权重文件和新的苹果图片
"""

import os          # 操作系统接口库
import shutil      # 文件操作库
from pathlib import Path  # 现代化路径操作库
from ultralytics import YOLO  # YOLOv8模型库

def prepare_custom_dataset(apple_images_folder):
    """
    准备自定义数据集用于微调训练
    功能：将用户提供的苹果图片整理成YOLOv8训练格式
    流程：
    1. 创建标准目录结构
    2. 扫描并复制图片文件
    3. 分割训练集和验证集
    4. 生成对应的标签文件
    
    参数说明：
    :param apple_images_folder: 包含苹果图片的文件夹路径（字符串）
    
    返回值：
    bool: 成功返回True，失败返回False
    """
    
    print("=== 准备自定义数据集 ===")
    
    # 步骤1：创建YOLOv8标准数据集目录结构
    dataset_dirs = [
        'custom_dataset/images/train',  # 训练图片目录
        'custom_dataset/images/val',    # 验证图片目录
        'custom_dataset/labels/train',  # 训练标签目录
        'custom_dataset/labels/val'     # 验证标签目录
    ]
    
    # 逐个创建目录，parents=True自动创建父目录
    for dir_path in dataset_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 步骤2：验证并扫描图片文件夹
    apple_folder = Path(apple_images_folder)
    if not apple_folder.exists():
        print(f"错误：图片文件夹不存在 {apple_images_folder}")
        return False
    
    # 定义支持的图片格式（常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []  # 存储找到的图片文件路径
    
    # 扫描各种格式的图片文件（包括大小写）
    for ext in image_extensions:
        image_files.extend(apple_folder.glob(f'*{ext}'))      # 小写扩展名
        image_files.extend(apple_folder.glob(f'*{ext.upper()}'))  # 大写扩展名
    
    if not image_files:
        print("错误：未找到任何图片文件")
        print("支持的格式：.jpg, .jpeg, .png, .bmp, .tiff")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 分割训练集和验证集（80%训练，20%验证）
    train_count = int(len(image_files) * 0.8)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # 复制训练图片
    for i, img_file in enumerate(train_files):
        dest = f"custom_dataset/images/train/apple_{i:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)
    
    # 复制验证图片
    for i, img_file in enumerate(val_files):
        dest = f"custom_dataset/images/val/apple_{i:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)
    
    print(f"训练集：{len(train_files)} 张图片")
    print(f"验证集：{len(val_files)} 张图片")
    
    return True

def create_auto_labels():
    """
    使用现有模型自动生成标签（半监督学习）
    """
    print("\n=== 自动生成标签 ===")
    
    # 加载现有模型
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    
    # 为训练图片生成标签
    train_images = list(Path('custom_dataset/images/train').glob('*'))
    val_images = list(Path('custom_dataset/images/val').glob('*'))
    
    def generate_labels(images, label_dir):
        label_path = Path(label_dir)
        label_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in images:
            # 使用现有模型进行预测
            results = model.predict(str(img_file), conf=0.25)  # 低置信度获取更多候选
            
            # 生成对应的标签文件
            label_file = label_path / f"{img_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # YOLO格式：class_id x_center y_center width height
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # 转换为相对坐标
                            img_w, img_h = result.orig_shape
                            x_center = (x1 + x2) / 2 / img_w
                            y_center = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            # 只保留高置信度的检测
                            if conf > 0.3:
                                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"生成标签：{label_file}")
    
    generate_labels(train_images, 'custom_dataset/labels/train')
    generate_labels(val_images, 'custom_dataset/labels/val')
    
    print("自动标签生成完成")

def create_data_yaml():
    """创建数据配置文件"""
    
    data_config = """# 自定义苹果检测数据集
path: ./custom_dataset
train: images/train
val: images/val

# 类别信息
nc: 1
names: ['apple']
"""
    
    with open('custom_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_config)
    
    print("创建数据配置文件：custom_dataset/data.yaml")

def fine_tune_model():
    """执行微调训练"""
    
    print("\n=== 开始微调训练 ===")
    
    # 加载现有权重
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    print("加载现有权重：YOLOv8-app-master/weights/best.pt")
    
    # 微调训练
    results = model.train(
        data='custom_dataset/data.yaml',
        epochs=50,              # 微调用较少轮数
        imgsz=640,
        batch=8,                # 小批次避免过拟合
        name='apple_fine_tune',
        save=True,
        plots=True,
        device='cpu',
        patience=20,            # 早停
        lr0=0.001,              # 较小学习率
        optimizer='Adam',
        augment=True,
        pretrained=True,        # 使用预训练权重
        resume=False,           # 不恢复，重新开始微调
    )
    
    # 验证微调后的模型
    print("\n=== 验证微调模型 ===")
    results = model.val()
    
    # 保存微调后的权重
    best_model_path = 'runs/detect/apple_fine_tune/weights/best.pt'
    if Path(best_model_path).exists():
        # 备份原权重
        shutil.copy2('YOLOv8-app-master/weights/best.pt', 
                    'YOLOv8-app-master/weights/best_original_backup.pt')
        
        # 替换为新权重
        shutil.copy2(best_model_path, 'YOLOv8-app-master/weights/best.pt')
        
        print(f"微调完成！新权重已保存到：YOLOv8-app-master/weights/best.pt")
        print(f"原权重已备份到：YOLOv8-app-master/weights/best_original_backup.pt")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 苹果检测 - 微调训练")
    print("=" * 60)
    
    # 用户输入图片文件夹路径
    apple_images_folder = input("\n请输入苹果图片文件夹路径：").strip()
    
    if not apple_images_folder:
        print("错误：请提供有效的图片文件夹路径")
        return
    
    # 执行微调流程
    if prepare_custom_dataset(apple_images_folder):
        create_data_yaml()
        create_auto_labels()
        fine_tune_model()
        
        print("\n" + "=" * 60)
        print("微调训练完成！")
        print("现在可以重新启动应用测试新模型")
        print("=" * 60)
    else:
        print("数据准备失败，请检查图片文件夹")

if __name__ == "__main__":
    main()