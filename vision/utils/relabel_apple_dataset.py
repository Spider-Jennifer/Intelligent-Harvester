#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新标注苹果数据集
使用预训练的YOLOv8n模型进行自动标注，只保留苹果检测结果
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def backup_labels():
    """备份现有标签文件"""
    backup_dir = "apple_dataset/labels_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    shutil.copytree("apple_dataset/labels", backup_dir)
    print(f"标签已备份到: {backup_dir}")
    return True

def get_coco_class_info():
    """获取COCO数据集的类别信息，返回苹果类别索引"""
    # COCO数据集类别列表 (80类)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 查找苹果类别索引
    apple_index = None
    for i, name in enumerate(coco_classes):
        if name.lower() == 'apple':
            apple_index = i
            break
    
    if apple_index is None:
        # 如果COCO类别列表中没有'apple'，使用其他水果类别
        fruit_classes = ['apple', 'orange', 'banana', 'fruit']
        for i, name in enumerate(coco_classes):
            if name.lower() in fruit_classes:
                apple_index = i
                print(f"警告: COCO类别中没有'apple'，使用 '{name}' 作为替代")
                break
    
    if apple_index is None:
        # 如果仍然没有找到，使用第一个类别（虽然不理想）
        apple_index = 0
        print("警告: 未找到水果类别，使用第一个类别")
    
    print(f"苹果类别索引: {apple_index}")
    return apple_index, coco_classes

def auto_label_with_filter(model, split="train", conf_threshold=0.3, iou_threshold=0.5):
    """
    使用预训练模型自动标注，只保留苹果检测结果
    """
    apple_index, coco_classes = get_coco_class_info()
    
    images_dir = f"apple_dataset/images/{split}"
    labels_dir = f"apple_dataset/labels/{split}"
    
    # 确保标签目录存在
    os.makedirs(labels_dir, exist_ok=True)
    
    # 获取图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(images_dir).glob(f"*{ext}")))
    
    print(f"\n处理 {split} 集: {len(image_files)} 张图片")
    
    images_with_apple = 0
    total_apple_detections = 0
    
    for img_path in image_files:
        # 对应的标签文件路径
        label_path = os.path.join(labels_dir, f"{img_path.stem}.txt")
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  警告: 无法读取图片 {img_path}, 跳过")
            continue
        
        img_h, img_w = img.shape[:2]
        
        # 使用预训练模型检测
        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 查找苹果检测结果
        apple_detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = coco_classes[cls_id] if cls_id < len(coco_classes) else f"class_{cls_id}"
                    conf = float(boxes.conf[i])
                    
                    # 只保留苹果检测结果
                    if cls_id == apple_index:
                        # 获取边界框坐标 (归一化)
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        
                        # 转换为YOLO格式 (center_x, center_y, width, height)
                        center_x = (x1 + x2) / 2 / img_w
                        center_y = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        # 确保坐标在[0,1]范围内
                        center_x = max(0.0, min(1.0, center_x))
                        center_y = max(0.0, min(1.0, center_y))
                        width = max(0.001, min(1.0, width))
                        height = max(0.001, min(1.0, height))
                        
                        apple_detections.append((0, center_x, center_y, width, height, conf))
                    
                    # 调试信息：显示检测到的其他类别
                    # else:
                    #     print(f"    检测到非苹果类别: {cls_name} (置信度: {conf:.2f})")
        
        # 保存标签文件
        with open(label_path, 'w') as f:
            for det in apple_detections:
                # YOLO格式: class_id center_x center_y width height
                f.write(f"{det[0]} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f}\n")
        
        if len(apple_detections) > 0:
            images_with_apple += 1
            total_apple_detections += len(apple_detections)
            print(f"  {img_path.name}: 检测到 {len(apple_detections)} 个苹果")
        else:
            print(f"  {img_path.name}: 未检测到苹果 (创建空标签)")
            # 如果没有检测到苹果，创建空标签文件（表示没有对象）
            # 注意: YOLO训练时需要空标签文件
    
    print(f"\n{split} 集统计:")
    print(f"  包含苹果的图片: {images_with_apple}/{len(image_files)} ({images_with_apple/len(image_files)*100:.1f}%)")
    print(f"  苹果总数: {total_apple_detections}")
    if images_with_apple > 0:
        print(f"  平均每张有苹果的图片检测数: {total_apple_detections/images_with_apple:.2f}")
    
    return images_with_apple

def main():
    print("=" * 80)
    print("苹果数据集重新标注工具")
    print("使用预训练的YOLOv8n模型进行自动标注，只保留苹果检测结果")
    print("=" * 80)
    
    # 检查数据集目录
    if not os.path.exists("apple_dataset"):
        print("错误: apple_dataset目录不存在")
        return
    
    # 备份现有标签
    print("\n1. 备份现有标签文件...")
    backup_labels()
    
    # 加载预训练模型
    print("\n2. 加载预训练模型 yolov8n.pt...")
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"错误: 预训练模型不存在 {model_path}")
        print("请下载: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        return
    
    model = YOLO(model_path)
    print("模型加载完成")
    
    # 设置检测参数
    conf_threshold = 0.25  # 置信度阈值
    iou_threshold = 0.45   # IoU阈值
    
    print(f"检测参数: 置信度={conf_threshold}, IoU={iou_threshold}")
    
    # 处理训练集和验证集
    print("\n3. 开始自动标注...")
    
    train_apple_count = auto_label_with_filter(model, "train", conf_threshold, iou_threshold)
    val_apple_count = auto_label_with_filter(model, "val", conf_threshold, iou_threshold)
    
    print("\n" + "=" * 80)
    print("自动标注完成!")
    print("=" * 80)
    print("重要说明:")
    print("1. 自动标注结果可能需要人工检查和修正")
    print("2. 建议使用标注工具(如LabelImg)检查标签准确性")
    print("3. 空标签文件表示图片中没有检测到苹果")
    print("4. 如果标签不准确，可以手动编辑txt文件或使用标注工具修正")
    print("\n下一步:")
    print("1. 运行训练脚本: python train_improved_apple_model.py")
    print("2. 测试模型: python test_model_accuracy.py")
    print("=" * 80)
    
    # 返回统计信息
    return {
        "train_images_with_apple": train_apple_count,
        "val_images_with_apple": val_apple_count
    }

if __name__ == "__main__":
    main()