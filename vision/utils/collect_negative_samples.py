#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集负样本图片，用于减少误识别
特别针对眼镜、人脸等常见误识别物体
"""

import os
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

def detect_person_and_glasses(image_path):
    """
    检测图片中是否包含人或眼镜
    返回：是否包含人/眼镜，检测结果数量
    """
    try:
        # 加载预训练的YOLOv8模型（包含80个类别）
        model = YOLO('yolov8n.pt')
        
        # 运行检测
        results = model(image_path, conf=0.3, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                # 检查检测到的类别
                detected_classes = set()
                for cls_id in result.boxes.cls.cpu().numpy():
                    cls_name = model.names[int(cls_id)]
                    detected_classes.add(cls_name)
                
                # 检查是否包含人或眼镜相关类别
                person_related = ['person', 'face', 'head', 'eye', 'glasses', 'sunglasses']
                found = any(any(related in cls_name.lower() for related in person_related) 
                           for cls_name in detected_classes)
                
                return found, len(result.boxes)
    
    except Exception as e:
        print(f"检测出错: {e}")
    
    return False, 0

def collect_negative_samples():
    """从苹果照片检测素材中收集负样本"""
    print("=" * 80)
    print("收集负样本图片 - 用于减少误识别")
    print("=" * 80)
    
    # 源目录（苹果照片检测素材）
    source_dir = "C:\\Users\\李晨鑫\\Desktop\\苹果照片检测素材"
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 {source_dir}")
        return
    
    # 目标目录（负样本）
    negative_dir = "apple_dataset/negative_samples"
    os.makedirs(negative_dir, exist_ok=True)
    
    # 获取所有图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(source_dir).glob(ext)))
    
    if not image_files:
        print(f"错误: 源目录中没有图片 {source_dir}")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("\n开始分析图片，寻找包含人或眼镜的负样本...")
    
    negative_images = []
    person_with_apple_images = []
    
    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"处理进度: {i}/{len(image_files)}")
        
        # 检测是否包含人或眼镜
        contains_person_or_glasses, detections = detect_person_and_glasses(str(img_path))
        
        if contains_person_or_glasses:
            # 复制到负样本目录
            dst_path = os.path.join(negative_dir, f"negative_{i:04d}.jpg")
            shutil.copy2(img_path, dst_path)
            negative_images.append(str(img_path))
            
            print(f"  负样本: {img_path.name} (检测到 {detections} 个相关物体)")
    
    print(f"\n收集完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"负样本数: {len(negative_images)}")
    
    # 创建负样本文件列表
    if negative_images:
        list_file = os.path.join(negative_dir, "negative_samples.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for img_path in negative_images:
                f.write(f"{img_path}\n")
        
        print(f"负样本列表已保存: {list_file}")
        
        # 生成YOLO格式的负样本标签文件（空标签）
        print("\n创建YOLO格式的负样本标签...")
        for img_path in negative_images:
            # 对应的标签文件路径
            label_path_base = os.path.splitext(os.path.basename(img_path))[0]
            label_file = os.path.join(negative_dir, f"{label_path_base}.txt")
            
            # 创建空标签文件（表示没有苹果）
            with open(label_file, 'w', encoding='utf-8') as f:
                pass  # 空文件表示没有标签
            
            print(f"  创建空标签: {os.path.basename(label_file)}")
        
        print(f"\n负样本已准备好，可以用以下方式整合到训练集:")
        print("1. 将负样本图片复制到 apple_dataset/images/train/")
        print("2. 将对应的空标签文件复制到 apple_dataset/labels/train/")
        print("3. 重新运行训练脚本")
        
        # 提供自动整合选项
        print("\n是否自动整合负样本到训练集? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            integrate_with_training_dataset(negative_dir)
    
    else:
        print("未找到合适的负样本图片")
        print("可能需要手动收集包含人或眼镜的图片")

def integrate_with_training_dataset(negative_dir):
    """将负样本整合到训练数据集"""
    print("\n开始整合负样本到训练数据集...")
    
    # 训练集目录
    train_images_dir = "apple_dataset/images/train"
    train_labels_dir = "apple_dataset/labels/train"
    
    # 确保目录存在
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # 统计现有文件数量
    existing_images = len([f for f in os.listdir(train_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    existing_labels = len([f for f in os.listdir(train_labels_dir) 
                         if f.endswith('.txt')])
    
    print(f"现有训练图片: {existing_images}")
    print(f"现有训练标签: {existing_labels}")
    
    # 复制负样本图片
    negative_images = [f for f in os.listdir(negative_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    copied_count = 0
    for img_file in negative_images:
        src_img = os.path.join(negative_dir, img_file)
        dst_img = os.path.join(train_images_dir, f"negative_{img_file}")
        
        # 复制图片
        shutil.copy2(src_img, dst_img)
        
        # 创建对应的空标签文件
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(negative_dir, label_file)
        dst_label = os.path.join(train_labels_dir, f"negative_{label_file}")
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            # 创建空标签文件
            with open(dst_label, 'w', encoding='utf-8') as f:
                pass
        
        copied_count += 1
    
    print(f"成功整合 {copied_count} 个负样本到训练集")
    print(f"训练集更新后: {existing_images + copied_count} 张图片")
    print(f"标签集更新后: {existing_labels + copied_count} 个标签")
    
    # 统计负样本比例
    total_images = existing_images + copied_count
    negative_ratio = copied_count / total_images * 100
    print(f"负样本比例: {negative_ratio:.1f}%")
    
    return copied_count

def main():
    """主函数"""
    print("负样本收集工具")
    print("用途: 收集可能被误识别为苹果的图片（如眼镜、人脸等）")
    print("这些图片将作为负样本帮助模型学习区分苹果和非苹果物体")
    print()
    
    print("选项:")
    print("1. 自动收集负样本")
    print("2. 仅整合已有负样本")
    print("3. 检查当前负样本比例")
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        collect_negative_samples()
    elif choice == "2":
        negative_dir = "apple_dataset/negative_samples"
        if os.path.exists(negative_dir):
            integrate_with_training_dataset(negative_dir)
        else:
            print(f"错误: 负样本目录不存在 {negative_dir}")
    elif choice == "3":
        # 检查当前负样本比例
        train_labels_dir = "apple_dataset/labels/train"
        if os.path.exists(train_labels_dir):
            label_files = [f for f in os.listdir(train_labels_dir) 
                          if f.endswith('.txt')]
            empty_labels = 0
            for f in label_files:
                p = os.path.join(train_labels_dir, f)
                if os.path.getsize(p) == 0:
                    empty_labels += 1
            
            print(f"当前训练集标签统计:")
            print(f"  总标签文件: {len(label_files)}")
            print(f"  空标签(负样本): {empty_labels}")
            print(f"  负样本比例: {empty_labels/len(label_files)*100:.1f}%")
        else:
            print(f"错误: 训练标签目录不存在 {train_labels_dir}")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()