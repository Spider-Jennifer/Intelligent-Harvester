#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将新苹果照片添加到数据集并重新训练模型
解决误识别问题
"""

import os
import shutil
import random
from pathlib import Path
import sys

def add_new_photos():
    """将新苹果照片添加到数据集中"""
    print("=== 添加新苹果照片到数据集 ===")
    
    # 新照片目录
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    
    if not os.path.exists(new_photos_dir):
        print(f"错误: 新照片目录不存在 {new_photos_dir}")
        return False
    
    # 获取所有图片文件
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']
    new_photos = []
    for ext in image_exts:
        new_photos.extend(list(Path(new_photos_dir).glob(f"*{ext}")))
    
    # 排除可能不是图片的文件（如脚本文件）
    new_photos = [p for p in new_photos if p.suffix.lower() in image_exts]
    
    print(f"找到 {len(new_photos)} 张新苹果照片")
    
    if not new_photos:
        print("错误: 新照片目录中没有图片文件")
        return False
    
    # 数据集目录
    dataset_images_dir = "apple_dataset/images"
    dataset_labels_dir = "apple_dataset/labels"
    
    # 确保目录存在
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(dataset_labels_dir, split), exist_ok=True)
    
    # 获取现有图片的最大索引
    def get_max_index(split):
        split_dir = os.path.join(dataset_images_dir, split)
        max_index = -1
        for f in os.listdir(split_dir):
            if f.startswith("apple_"):
                # 提取数字部分
                try:
                    num_part = f.split('_')[1].split('.')[0]
                    idx = int(num_part)
                    if idx > max_index:
                        max_index = idx
                except:
                    pass
        return max_index
    
    train_max_idx = get_max_index("train")
    val_max_idx = get_max_index("val")
    print(f"训练集现有最大索引: {train_max_idx}")
    print(f"验证集现有最大索引: {val_max_idx}")
    
    # 使用较大的索引作为起始
    start_idx = max(train_max_idx, val_max_idx) + 1
    
    # 随机打乱并分割 (80%训练, 20%验证)
    random.shuffle(new_photos)
    split_idx = int(len(new_photos) * 0.8)
    train_photos = new_photos[:split_idx]
    val_photos = new_photos[split_idx:]
    
    print(f"训练集新增: {len(train_photos)} 张")
    print(f"验证集新增: {len(val_photos)} 张")
    
    # 复制训练集图片
    for i, photo_path in enumerate(train_photos):
        new_idx = start_idx + i
        dst_filename = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "train", dst_filename)
        shutil.copy2(photo_path, dst_path)
        print(f"  复制到训练集: {photo_path.name} -> {dst_filename}")
    
    # 复制验证集图片
    for i, photo_path in enumerate(val_photos):
        new_idx = start_idx + len(train_photos) + i
        dst_filename = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "val", dst_filename)
        shutil.copy2(photo_path, dst_path)
        print(f"  复制到验证集: {photo_path.name} -> {dst_filename}")
    
    print("新照片已添加到数据集")
    return True

def auto_relabel_dataset():
    """自动重新标注整个数据集"""
    print("\n=== 自动重新标注数据集 ===")
    
    # 检查relabel脚本是否存在
    if not os.path.exists("relabel_apple_dataset.py"):
        print("错误: relabel_apple_dataset.py 不存在")
        return False
    
    # 导入relabel函数
    import subprocess
    import sys
    
    print("运行自动标注脚本...")
    try:
        result = subprocess.run([sys.executable, "relabel_apple_dataset.py"], 
                              capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        if result.returncode != 0:
            print(f"自动标注失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"运行自动标注脚本时出错: {e}")
        return False
    
    return True

def train_improved_model():
    """训练改进模型"""
    print("\n=== 训练改进模型 ===")
    
    # 检查训练脚本是否存在
    if not os.path.exists("improve_model_accuracy.py"):
        print("错误: improve_model_accuracy.py 不存在")
        return False
    
    import subprocess
    import sys
    
    print("运行训练脚本...")
    try:
        result = subprocess.run([sys.executable, "improve_model_accuracy.py"], 
                              capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        if result.returncode != 0:
            print(f"训练失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"运行训练脚本时出错: {e}")
        return False
    
    return True

def test_new_model():
    """测试新模型在新增照片上的表现"""
    print("\n=== 测试新模型 ===")
    
    # 使用新增照片目录
    test_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    
    # 查找最新模型
    model_candidates = ["apple_improved.pt", "apple_best.pt", "apple_sensitive.pt"]
    model_to_test = None
    
    for model_name in model_candidates:
        if os.path.exists(model_name):
            model_to_test = model_name
            print(f"使用模型进行测试: {model_name}")
            break
    
    if not model_to_test:
        print("警告: 未找到训练好的模型")
        return False
    
    # 简单的测试脚本
    test_script = f"""
import os
import glob
from ultralytics import YOLO
import cv2

model = YOLO('{model_to_test}')
photo_dir = r'{test_dir}'

image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []
for ext in image_exts:
    image_paths.extend(glob.glob(os.path.join(photo_dir, ext)))

print(f"测试模型: {{model_to_test}}")
print(f"测试图片: {{len(image_paths)}} 张")

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    results = model(img, conf=0.3, verbose=False)
    detections = 0
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            detections = len(result.boxes)
    
    img_name = os.path.basename(img_path)
    print(f"  {{img_name}}: 检测到 {{detections}} 个苹果")
    
    # 误识别检查
    if detections > 1:
        print(f"    ⚠️  警告: 可能误识别 (检测到多个苹果)")
"""
    
    # 执行测试
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
    except Exception as e:
        print(f"测试时出错: {e}")
    
    return True

def main():
    print("=" * 80)
    print("苹果检测模型改进流程")
    print("解决误识别问题 - 添加新照片并重新训练")
    print("=" * 80)
    
    # 步骤1: 添加新照片
    if not add_new_photos():
        print("添加新照片失败，退出")
        return
    
    # 步骤2: 自动重新标注
    if not auto_relabel_dataset():
        print("自动重新标注失败，退出")
        return
    
    # 步骤3: 训练改进模型
    if not train_improved_model():
        print("训练改进模型失败，退出")
        return
    
    # 步骤4: 测试新模型
    test_new_model()
    
    print("\n" + "=" * 80)
    print("改进流程完成！")
    print("下一步:")
    print("1. 检查测试结果，如果仍有误识别，需要添加负样本")
    print("2. 可以考虑添加人物图片作为负样本以减少误识别")
    print("3. 运行 test_misdetection.py 进行更详细的误识别测试")
    print("=" * 80)

if __name__ == "__main__":
    main()