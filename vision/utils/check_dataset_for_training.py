#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据集是否适合训练
"""

import os
import glob
import yaml

def check_dataset():
    """检查数据集"""
    dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
    
    print("=" * 60)
    print("数据集检查报告")
    print("=" * 60)
    print()
    
    # 1. 检查data.yaml文件
    data_yaml = os.path.join(dataset_path, "data.yaml")
    print("1. 检查数据集配置文件:")
    if os.path.exists(data_yaml):
        print(f"   找到配置文件: {data_yaml}")
        
        # 读取配置文件
        try:
            with open(data_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"   配置文件内容:")
            print(f"     类别数量 (nc): {config.get('nc', '未指定')}")
            print(f"     类别名称 (names): {config.get('names', '未指定')}")
            print(f"     训练集路径 (train): {config.get('train', '未指定')}")
            print(f"     验证集路径 (val): {config.get('val', '未指定')}")
            
        except Exception as e:
            print(f"   ✗ 读取配置文件失败: {e}")
    else:
        print(f"   找不到配置文件: {data_yaml}")
    
    print()
    
    # 2. 检查图片文件
    print("2. 检查图片文件:")
    
    # 训练集图片
    train_img_dir = os.path.join(dataset_path, "images", "train")
    if os.path.exists(train_img_dir):
        train_images = glob.glob(os.path.join(train_img_dir, "*.jpg"))
        train_images += glob.glob(os.path.join(train_img_dir, "*.png"))
        train_images += glob.glob(os.path.join(train_img_dir, "*.jpeg"))
        print(f"   训练集图片: {len(train_images)} 张")
    else:
        print(f"   ✗ 训练集图片目录不存在: {train_img_dir}")
    
    # 验证集图片
    val_img_dir = os.path.join(dataset_path, "images", "val")
    if os.path.exists(val_img_dir):
        val_images = glob.glob(os.path.join(val_img_dir, "*.jpg"))
        val_images += glob.glob(os.path.join(val_img_dir, "*.png"))
        val_images += glob.glob(os.path.join(val_img_dir, "*.jpeg"))
        print(f"   验证集图片: {len(val_images)} 张")
    else:
        print(f"   ✗ 验证集图片目录不存在: {val_img_dir}")
    
    print()
    
    # 3. 检查标签文件
    print("3. 检查标签文件:")
    
    # 训练集标签
    train_label_dir = os.path.join(dataset_path, "labels", "train")
    if os.path.exists(train_label_dir):
        train_labels = glob.glob(os.path.join(train_label_dir, "*.txt"))
        print(f"   训练集标签: {len(train_labels)} 个")
        
        # 检查标签文件格式
        if train_labels:
            sample_label = train_labels[0]
            try:
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        first_line = lines[0].strip()
                        parts = first_line.split()
                        print(f"   示例标签格式: {first_line}")
                        print(f"   字段数量: {len(parts)} (应为5个: class_id x_center y_center width height)")
            except:
                pass
    else:
        print(f"   训练集标签目录不存在: {train_label_dir}")
    
    # 验证集标签
    val_label_dir = os.path.join(dataset_path, "labels", "val")
    if os.path.exists(val_label_dir):
        val_labels = glob.glob(os.path.join(val_label_dir, "*.txt"))
        print(f"   验证集标签: {len(val_labels)} 个")
    else:
        print(f"   验证集标签目录不存在: {val_label_dir}")
    
    print()
    
    # 4. 检查图片和标签的对应关系
    print("4. 检查图片和标签对应关系:")
    
    if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
        # 获取所有图片文件名（不含扩展名）
        train_image_files = glob.glob(os.path.join(train_img_dir, "*.*"))
        train_image_names = set()
        for img_file in train_image_files:
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                name = os.path.splitext(os.path.basename(img_file))[0]
                train_image_names.add(name)
        
        # 获取所有标签文件名（不含扩展名）
        train_label_files = glob.glob(os.path.join(train_label_dir, "*.txt"))
        train_label_names = set()
        for label_file in train_label_files:
            name = os.path.splitext(os.path.basename(label_file))[0]
            train_label_names.add(name)
        
        # 检查对应关系
        images_without_labels = train_image_names - train_label_names
        labels_without_images = train_label_names - train_image_names
        
        print(f"   训练集:")
        print(f"     - 图片数量: {len(train_image_names)}")
        print(f"     - 标签数量: {len(train_label_names)}")
        print(f"     - 缺少标签的图片: {len(images_without_labels)}")
        print(f"     - 缺少图片的标签: {len(labels_without_images)}")
        
        if images_without_labels:
            print(f"     示例缺少标签的图片: {list(images_without_labels)[:3]}")
    
    print()
    
    # 5. 训练建议
    print("5. 训练建议:")
    
    total_images = len(train_images) if 'train_images' in locals() else 0
    total_labels = len(train_labels) if 'train_labels' in locals() else 0
    
    if total_labels == 0:
        print("   严重问题: 没有找到标签文件")
        print("     建议: 使用标注工具为图片创建标签")
        print("     推荐工具: LabelImg, CVAT, Roboflow")
        print()
        print("     或者使用以下方法:")
        print("     1. 使用预训练模型进行零样本学习")
        print("     2. 使用少量标注数据进行微调")
        print("     3. 使用自动标注工具")
    
    elif total_labels < total_images * 0.5:
        print("   警告: 标签文件不足")
        print(f"     只有 {total_labels}/{total_images} 张图片有标签")
        print("     建议: 补充标注更多图片")
    
    elif total_labels >= total_images * 0.8:
        print("   良好: 数据集准备就绪")
        print(f"     {total_labels}/{total_images} 张图片有标签")
        print("     可以开始训练")
    
    else:
        print("   注意: 数据集基本可用")
        print(f"     {total_labels}/{total_images} 张图片有标签")
        print("     可以开始训练，但效果可能不是最佳")
    
    print()
    print("=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == "__main__":
    check_dataset()