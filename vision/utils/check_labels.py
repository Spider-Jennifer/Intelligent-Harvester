#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据集标签文件
"""

import os
import glob

def check_dataset():
    dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
    
    print("检查数据集结构:")
    print(f"数据集路径: {dataset_path}")
    print()
    
    # 检查图片文件
    train_images = glob.glob(os.path.join(dataset_path, "images", "train", "*.jpg"))
    train_images += glob.glob(os.path.join(dataset_path, "images", "train", "*.png"))
    val_images = glob.glob(os.path.join(dataset_path, "images", "val", "*.jpg"))
    val_images += glob.glob(os.path.join(dataset_path, "images", "val", "*.png"))
    
    print(f"训练集图片数量: {len(train_images)}")
    print(f"验证集图片数量: {len(val_images)}")
    print()
    
    # 检查标签文件
    train_labels = glob.glob(os.path.join(dataset_path, "labels", "train", "*.txt"))
    val_labels = glob.glob(os.path.join(dataset_path, "labels", "val", "*.txt"))
    
    print(f"训练集标签文件数量: {len(train_labels)}")
    print(f"验证集标签文件数量: {len(val_labels)}")
    print()
    
    # 检查是否有对应的标签文件
    print("检查标签文件对应关系:")
    
    # 检查训练集
    missing_train_labels = []
    for img_path in train_images[:10]:  # 只检查前10个
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(dataset_path, "labels", "train", f"{img_name}.txt")
        if not os.path.exists(label_path):
            missing_train_labels.append(img_name)
    
    if missing_train_labels:
        print(f"训练集缺少标签文件: {len(missing_train_labels)} 个")
        print(f"示例: {missing_train_labels[:5]}")
    else:
        print("训练集标签文件完整")
    
    # 检查验证集
    missing_val_labels = []
    for img_path in val_images[:10]:  # 只检查前10个
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(dataset_path, "labels", "val", f"{img_name}.txt")
        if not os.path.exists(label_path):
            missing_val_labels.append(img_name)
    
    if missing_val_labels:
        print(f"验证集缺少标签文件: {len(missing_val_labels)} 个")
        print(f"示例: {missing_val_labels[:5]}")
    else:
        print("验证集标签文件完整")
    
    print()
    print("建议:")
    if len(train_labels) < len(train_images) or len(val_labels) < len(val_images):
        print("1. 需要为图片创建对应的标签文件")
        print("2. 标签文件格式: 每行 'class_id x_center y_center width height'")
        print("3. 坐标需要归一化到 [0, 1]")
    else:
        print("数据集准备就绪，可以开始训练")

if __name__ == "__main__":
    check_dataset()