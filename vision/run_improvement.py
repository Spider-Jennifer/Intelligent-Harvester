#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进苹果检测模型的主脚本
解决误识别问题，添加新照片并重新训练
"""

import os
import sys
import subprocess
import shutil
import random
from pathlib import Path

def run_script(script_name, args=None):
    """运行Python脚本并捕获输出"""
    if not os.path.exists(script_name):
        print(f"错误: 脚本不存在 {script_name}")
        return False
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"\n运行脚本: {script_name}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        print(result.stdout)
        if result.stderr:
            print("标准错误:", result.stderr)
        
        if result.returncode != 0:
            print(f"脚本返回错误码: {result.returncode}")
            return False
    except Exception as e:
        print(f"运行脚本时出错: {e}")
        return False
    
    print("-" * 60)
    return True

def add_new_photos():
    """添加新照片到数据集"""
    print("\n=== 步骤1: 添加新苹果照片到数据集 ===")
    
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    if not os.path.exists(new_photos_dir):
        print(f"错误: 目录不存在 {new_photos_dir}")
        return False
    
    # 获取图片文件
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']
    new_photos = []
    for ext in image_exts:
        new_photos.extend(list(Path(new_photos_dir).glob(f"*{ext}")))
    
    # 过滤出确实是图片的文件
    new_photos = [p for p in new_photos if p.suffix.lower() in [ext.lower() for ext in image_exts]]
    
    print(f"找到 {len(new_photos)} 张新苹果照片")
    
    if not new_photos:
        print("错误: 没有找到图片文件")
        return False
    
    dataset_images_dir = "apple_dataset/images"
    
    # 确保目录存在
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_images_dir, split), exist_ok=True)
    
    # 获取现有最大索引
    def get_max_index(split):
        split_dir = os.path.join(dataset_images_dir, split)
        max_idx = -1
        for f in os.listdir(split_dir):
            if f.startswith("apple_"):
                try:
                    num = f.split('_')[1].split('.')[0]
                    idx = int(num)
                    if idx > max_idx:
                        max_idx = idx
                except:
                    pass
        return max_idx
    
    train_max = get_max_index("train")
    val_max = get_max_index("val")
    start_idx = max(train_max, val_max) + 1
    
    # 分割
    random.shuffle(new_photos)
    split_idx = int(len(new_photos) * 0.8)
    train_photos = new_photos[:split_idx]
    val_photos = new_photos[split_idx:]
    
    print(f"训练集新增: {len(train_photos)} 张")
    print(f"验证集新增: {len(val_photos)} 张")
    
    # 复制训练集
    for i, photo_path in enumerate(train_photos):
        new_idx = start_idx + i
        dst_name = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "train", dst_name)
        shutil.copy2(photo_path, dst_path)
        print(f"  训练集: {photo_path.name} -> {dst_name}")
    
    # 复制验证集
    for i, photo_path in enumerate(val_photos):
        new_idx = start_idx + len(train_photos) + i
        dst_name = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "val", dst_name)
        shutil.copy2(photo_path, dst_path)
        print(f"  验证集: {photo_path.name} -> {dst_name}")
    
    print("新照片添加完成")
    return True

def main():
    print("=" * 80)
    print("苹果检测模型改进流程")
    print("解决误识别问题 - 自动执行")
    print("=" * 80)
    
    # 步骤1: 添加新照片
    if not add_new_photos():
        print("添加新照片失败，退出")
        return
    
    # 步骤2: 自动重新标注
    if not run_script("relabel_apple_dataset.py"):
        print("自动重新标注失败，退出")
        return
    
    # 步骤3: 训练改进模型
    if not run_script("improve_model_accuracy.py"):
        print("训练改进模型失败，退出")
        return
    
    # 步骤4: 测试新模型
    print("\n=== 步骤4: 测试新模型 ===")
    
    # 查找最新模型
    model_candidates = ["apple_improved.pt", "apple_best.pt", "apple_sensitive.pt"]
    model_path = None
    for m in model_candidates:
        if os.path.exists(m):
            model_path = m
            break
    
    if not model_path:
        print("警告: 未找到训练好的模型")
        return
    
    # 简单测试
    test_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return
    
    # 导入ultralytics进行测试
    try:
        from ultralytics import YOLO
        import glob
        import cv2
        
        model = YOLO(model_path)
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        
        print(f"测试模型: {model_path}")
        print(f"测试图片: {len(image_paths)} 张")
        print("置信度阈值: 0.3")
        
        misdetected = 0
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
            status = "✓"
            if detections > 1:
                status = "⚠️"
                misdetected += 1
            
            print(f"  {status} {img_name}: 检测到 {detections} 个苹果")
        
        print(f"\n误识别统计: {misdetected}/{len(image_paths)} 张图片检测到多个苹果")
        if misdetected == 0:
            print("✓ 未发现误识别")
        else:
            print("⚠️ 仍存在误识别，建议添加负样本或调整训练参数")
    
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保已安装 ultralytics 和 opencv-python")
    except Exception as e:
        print(f"测试过程中出错: {e}")
    
    print("\n" + "=" * 80)
    print("改进流程完成！")
    print("=" * 80)

if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()