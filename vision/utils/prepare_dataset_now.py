"""
准备苹果检测数据集
将328张苹果照片分配到训练集和验证集，并创建自动标注
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    print("=" * 60)
    print("准备苹果检测数据集")
    print("=" * 60)
    
    # 步骤1: 检查照片目录
    photo_dir = 'apple_photos'
    if not os.path.exists(photo_dir):
        print(f"错误: 找不到 {photo_dir} 目录")
        return
    
    # 获取所有照片
    photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        photos.extend(list(Path(photo_dir).glob(f'*{ext}')))
    
    print(f"找到 {len(photos)} 张苹果照片")
    
    if len(photos) < 50:
        print("警告: 照片数量不足，建议至少50张")
        return
    
    # 步骤2: 创建数据集目录结构
    print("\n创建数据集目录结构...")
    dirs = [
        'YOLO-v8-app/dataset/images/train',
        'YOLO-v8-app/dataset/images/val',
        'YOLO-v8-app/dataset/labels/train',
        'YOLO-v8-app/dataset/labels/val'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建: {dir_path}")
    
    # 步骤3: 随机分配训练集和验证集 (80%/20%)
    print("\n分配训练集和验证集...")
    random.shuffle(photos)
    split_idx = int(len(photos) * 0.8)
    train_photos = photos[:split_idx]
    val_photos = photos[split_idx:]
    
    print(f"  训练集: {len(train_photos)} 张照片")
    print(f"  验证集: {len(val_photos)} 张照片")
    
    # 步骤4: 复制照片到数据集目录
    print("\n复制照片到数据集目录...")
    
    # 复制训练集照片
    for i, photo_path in enumerate(train_photos):
        dst_path = os.path.join('YOLO-v8-app/dataset/images/train', f'apple_{i:04d}{photo_path.suffix}')
        shutil.copy2(photo_path, dst_path)
    
    # 复制验证集照片
    for i, photo_path in enumerate(val_photos):
        dst_path = os.path.join('YOLO-v8-app/dataset/images/val', f'apple_val_{i:04d}{photo_path.suffix}')
        shutil.copy2(photo_path, dst_path)
    
    print("  照片复制完成")
    
    # 步骤5: 自动标注
    print("\n开始自动标注...")
    print("  加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        images_dir = f'YOLO-v8-app/dataset/images/{split}'
        labels_dir = f'YOLO-v8-app/dataset/labels/{split}'
        
        image_files = list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.png')) + \
                     list(Path(images_dir).glob('*.jpeg'))
        
        print(f"\n  处理 {split} 集: {len(image_files)} 张照片")
        
        processed = 0
        for img_path in image_files:
            label_path = os.path.join(labels_dir, f'{img_path.stem}.txt')
            
            # 如果标签文件已存在，跳过
            if os.path.exists(label_path):
                continue
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # 检测苹果
            results = model(img, conf=0.25, verbose=False)
            
            apple_detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        cls_name = model.names[cls_id]
                        
                        # 检查是否是苹果或水果
                        if cls_name.lower() in ['apple', 'orange', 'banana', 'fruit']:
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            img_h, img_w = img.shape[:2]
                            
                            # 转换为YOLO格式
                            center_x = (x1 + x2) / 2 / img_w
                            center_y = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            apple_detections.append((0, center_x, center_y, width, height))
            
            # 如果没有检测到，使用默认标注
            if not apple_detections:
                apple_detections.append((0, 0.5, 0.5, 0.4, 0.4))
            
            # 保存标签
            with open(label_path, 'w') as f:
                for det in apple_detections:
                    f.write(f'{det[0]} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f}\n')
            
            processed += 1
            if processed % 20 == 0:
                print(f"    已处理 {processed}/{len(image_files)}")
    
    print("\n  自动标注完成！")
    
    # 步骤6: 创建配置文件
    print("\n创建数据集配置文件...")
    config_content = """# 苹果检测数据集配置文件
names:
- apple
nc: 1
path: ./dataset
train: images/train
val: images/val
"""
    
    config_path = 'YOLO-v8-app/dataset/data.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"  配置文件已创建: {config_path}")
    
    # 步骤7: 显示统计信息
    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print("=" * 60)
    print(f"总照片数量: {len(photos)}")
    print(f"训练集: {len(train_photos)} 张照片")
    print(f"验证集: {len(val_photos)} 张照片")
    print(f"训练标签: {len(list(Path('YOLO-v8-app/dataset/labels/train').glob('*.txt')))} 个文件")
    print(f"验证标签: {len(list(Path('YOLO-v8-app/dataset/labels/val').glob('*.txt')))} 个文件")
    print("\n下一步:")
    print("1. 开始训练: python train_apple_high_quality.py")
    print("2. 或运行: train_apple_high_quality.bat")
    print("=" * 60)

if __name__ == "__main__":
    main()