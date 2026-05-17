#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能数据集过滤脚本
功能：处理包含多种水果的图片，只保留高质量苹果样本
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def analyze_image_quality(model, image_path):
    """
    分析图片中苹果检测的质量
    返回：(是否适合训练, 检测数量, 平均置信度, 详细信息)
    """
    
    # 使用模型预测
    results = model.predict(str(image_path), conf=0.1)  # 低阈值获取所有检测
    
    if not results or results[0].boxes is None:
        return False, 0, 0, "未检测到苹果"
    
    boxes = results[0].boxes
    apple_count = len(boxes)
    
    if apple_count == 0:
        return False, 0, 0, "未检测到苹果"
    
    # 计算平均置信度
    confidences = boxes.conf.cpu().numpy()
    avg_confidence = np.mean(confidences)
    
    # 分析检测质量
    details = []
    
    # 检查置信度分布
    high_conf_count = np.sum(confidences > 0.7)
    medium_conf_count = np.sum((confidences > 0.4) & (confidences <= 0.7))
    low_conf_count = np.sum(confidences <= 0.4)
    
    details.append(f"高置信度(>0.7): {high_conf_count}个")
    details.append(f"中置信度(0.4-0.7): {medium_conf_count}个") 
    details.append(f"低置信度(≤0.4): {low_conf_count}个")
    
    # 检查检测框大小（避免过小的检测）
    img_h, img_w = results[0].orig_shape
    valid_boxes = 0
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        width, height = x2 - x1, y2 - y1
        
        # 转换为相对尺寸
        rel_width = width / img_w
        rel_height = height / img_h
        rel_area = rel_width * rel_height
        
        # 苹果应该有一定大小（占图像0.5%以上）
        if rel_area > 0.005:
            valid_boxes += 1
    
    details.append(f"有效尺寸苹果: {valid_boxes}个")
    
    # 质量判断标准
    quality_score = 0
    
    # 高置信度检测加分
    quality_score += high_conf_count * 3
    quality_score += medium_conf_count * 1
    quality_score -= low_conf_count * 2
    
    # 有效检测加分
    quality_score += valid_boxes * 2
    
    # 平均置信度加分
    quality_score += avg_confidence * 10
    
    # 判断是否适合训练
    is_suitable = (
        valid_boxes >= 1 and                    # 至少1个有效苹果
        avg_confidence > 0.3 and                # 平均置信度不太低
        high_conf_count >= 1 and                # 至少1个高置信度
        quality_score > 5                       # 总体质量合格
    )
    
    return is_suitable, apple_count, avg_confidence, "; ".join(details)

def filter_mixed_fruit_dataset(images_folder, output_folder):
    """
    过滤包含多种水果的数据集
    """
    
    print("=== 智能数据集过滤 ===")
    
    # 加载模型
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    
    # 创建输出目录
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 扫描图片
    image_folder = Path(images_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 分析每张图片
    suitable_images = []
    rejected_images = []
    
    for i, img_file in enumerate(image_files):
        print(f"\n分析 [{i+1}/{len(image_files)}]: {img_file.name}")
        
        is_suitable, count, avg_conf, details = analyze_image_quality(model, img_file)
        
        print(f"  检测到 {count} 个苹果")
        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  详情: {details}")
        
        if is_suitable:
            print(f"  ✅ 适合训练")
            suitable_images.append(img_file)
        else:
            print(f"  ❌ 不适合训练")
            rejected_images.append(img_file)
    
    # 复制适合的图片
    print(f"\n=== 复制适合训练的图片 ===")
    
    for i, img_file in enumerate(suitable_images):
        dest = output_path / f"apple_{i:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)
        print(f"复制: {img_file.name} -> {dest.name}")
    
    # 生成报告
    print(f"\n=== 过滤报告 ===")
    print(f"原始图片数量: {len(image_files)}")
    print(f"适合训练: {len(suitable_images)} 张")
    print(f"被拒绝: {len(rejected_images)} 张")
    print(f"保留率: {len(suitable_images)/len(image_files)*100:.1f}%")
    
    # 保存被拒绝的图片列表
    if rejected_images:
        with open(output_path / 'rejected_images.txt', 'w', encoding='utf-8') as f:
            f.write("被拒绝的图片及原因:\n")
            f.write("=" * 50 + "\n")
            
            for img_file in rejected_images:
                is_suitable, count, avg_conf, details = analyze_image_quality(model, img_file)
                f.write(f"{img_file.name}\n")
                f.write(f"  原因: {details}\n")
                f.write(f"  置信度: {avg_conf:.3f}\n\n")
    
    return len(suitable_images) > 0

def create_filtered_dataset(filtered_images_folder):
    """
    基于过滤后的图片创建训练数据集
    """
    
    print("\n=== 创建训练数据集 ===")
    
    # 创建数据集目录
    dataset_dirs = [
        'filtered_dataset/images/train',
        'filtered_dataset/images/val',
        'filtered_dataset/labels/train',
        'filtered_dataset/labels/val'
    ]
    
    for dir_path in dataset_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 获取过滤后的图片
    filtered_folder = Path(filtered_images_folder)
    image_files = list(filtered_folder.glob('*'))
    
    if not image_files:
        print("错误：没有找到过滤后的图片")
        return False
    
    # 分割训练集和验证集
    train_count = int(len(image_files) * 0.8)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # 复制图片
    for i, img_file in enumerate(train_files):
        dest = f"filtered_dataset/images/train/{img_file.name}"
        shutil.copy2(img_file, dest)
    
    for i, img_file in enumerate(val_files):
        dest = f"filtered_dataset/images/val/{img_file.name}"
        shutil.copy2(img_file, dest)
    
    print(f"训练集：{len(train_files)} 张")
    print(f"验证集：{len(val_files)} 张")
    
    # 生成标签
    model = YOLO('YOLOv8-app-master/weights/best.pt')
    
    def generate_labels(images, label_dir):
        label_path = Path(label_dir)
        label_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in images:
            results = model.predict(str(img_file), conf=0.3)  # 使用较高置信度
            
            label_file = label_path / f"{img_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = box.conf[0].cpu().numpy()
                            
                            # 只保留高置信度的检测
                            if conf > 0.5:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                img_w, img_h = result.orig_shape
                                
                                x_center = (x1 + x2) / 2 / img_w
                                y_center = (y1 + y2) / 2 / img_h
                                width = (x2 - x1) / img_w
                                height = (y2 - y1) / img_h
                                
                                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    generate_labels(train_files, 'filtered_dataset/labels/train')
    generate_labels(val_files, 'filtered_dataset/labels/val')
    
    # 创建配置文件
    data_config = """# 过滤后的苹果检测数据集
path: ./filtered_dataset
train: images/train
val: images/val

nc: 1
names: ['apple']
"""
    
    with open('filtered_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_config)
    
    print("创建数据配置文件：filtered_dataset/data.yaml")
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("智能数据集过滤 - 处理混合水果图片")
    print("=" * 60)
    
    # 用户输入
    images_folder = input("\n请输入包含混合水果的图片文件夹路径：").strip()
    
    if not images_folder or not Path(images_folder).exists():
        print("错误：文件夹不存在")
        return
    
    # 第一步：过滤图片
    output_folder = "filtered_apple_images"
    if filter_mixed_fruit_dataset(images_folder, output_folder):
        # 第二步：创建训练数据集
        if create_filtered_dataset(output_folder):
            print("\n" + "=" * 60)
            print("数据集准备完成！")
            print("现在可以运行微调训练：")
            print("python train_filtered_model.py")
            print("=" * 60)
        else:
            print("创建训练数据集失败")
    else:
        print("没有找到适合训练的图片")

if __name__ == "__main__":
    main()