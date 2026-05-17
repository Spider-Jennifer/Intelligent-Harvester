"""
快速苹果检测模型训练
直接在项目根目录下创建数据集
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import cv2

print("快速苹果检测模型训练")
print("=" * 60)

# 步骤1: 在项目根目录创建数据集
print("1. 创建数据集目录...")
dataset_dirs = [
    'apple_dataset/images/train',
    'apple_dataset/images/val',
    'apple_dataset/labels/train',
    'apple_dataset/labels/val'
]

for dir_path in dataset_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"   创建: {dir_path}")

# 步骤2: 复制苹果照片
print("\n2. 复制苹果照片...")
photo_dir = 'apple_photos'
if os.path.exists(photo_dir):
    photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        photos.extend(list(Path(photo_dir).glob(f'*{ext}')))
    
    print(f"   找到 {len(photos)} 张苹果照片")
    
    if len(photos) < 20:
        print("   错误: 照片数量不足")
        exit(1)
    
    # 随机分配训练集和验证集
    random.shuffle(photos)
    split_idx = int(len(photos) * 0.8)
    train_photos = photos[:split_idx]
    val_photos = photos[split_idx:]
    
    print(f"   训练集: {len(train_photos)} 张")
    print(f"   验证集: {len(val_photos)} 张")
    
    # 复制训练集照片
    for i, photo_path in enumerate(train_photos):
        dst_path = os.path.join('apple_dataset/images/train', f'apple_{i:04d}{photo_path.suffix}')
        shutil.copy2(photo_path, dst_path)
    
    # 复制验证集照片
    for i, photo_path in enumerate(val_photos):
        dst_path = os.path.join('apple_dataset/images/val', f'apple_val_{i:04d}{photo_path.suffix}')
        shutil.copy2(photo_path, dst_path)
    
    print("   照片复制完成")
else:
    print(f"   错误: 找不到 {photo_dir} 目录")
    exit(1)

# 步骤3: 创建简单的data.yaml文件
print("\n3. 创建配置文件...")
config_content = """# 苹果检测数据集
names:
- apple
nc: 1
path: ./apple_dataset
train: images/train
val: images/val
"""

config_path = 'apple_dataset/data.yaml'
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"   配置文件: {config_path}")

# 步骤4: 创建简单标签（假设苹果在图片中心）
print("\n4. 创建简单标签...")
for split in ['train', 'val']:
    images_dir = f'apple_dataset/images/{split}'
    labels_dir = f'apple_dataset/labels/{split}'
    
    image_files = list(Path(images_dir).glob('*.jpg')) + \
                 list(Path(images_dir).glob('*.png')) + \
                 list(Path(images_dir).glob('*.jpeg'))
    
    print(f"   处理 {split} 集: {len(image_files)} 张照片")
    
    for img_path in image_files:
        label_path = os.path.join(labels_dir, f'{img_path.stem}.txt')
        
        # 创建简单标签：苹果在图片中心，占40%大小
        with open(label_path, 'w') as f:
            f.write("0 0.5 0.5 0.4 0.4\n")

print("   标签创建完成")

# 步骤5: 开始训练
print("\n5. 开始训练...")
print("   注意: 训练可能需要10-20分钟")

model = YOLO('yolov8n.pt')

training_args = {
    'data': config_path,
    'epochs': 20,
    'imgsz': 320,
    'batch': 4,
    'workers': 0,
    'device': 'cpu',
    'name': 'apple_quick',
    'patience': 3,
    'save': True,
    'pretrained': True,
}

print("\n训练配置:")
for key, value in training_args.items():
    if key != 'data':
        print(f"   {key}: {value}")

print("\n开始训练...")
results = model.train(**training_args)

print("\n训练完成！")

# 步骤6: 复制模型到根目录
best_model_path = "runs/detect/apple_quick/weights/best.pt"
if os.path.exists(best_model_path):
    shutil.copy2(best_model_path, "apple_quick.pt")
    print(f"最佳模型已保存: apple_quick.pt")
    
    # 快速测试
    print("\n快速测试模型...")
    test_model = YOLO("apple_quick.pt")
    
    # 测试一张图片
    test_img = os.path.join('apple_dataset/images/train', os.listdir('apple_dataset/images/train')[0])
    test_results = test_model(test_img, conf=0.1, verbose=False)  # 降低置信度阈值以提高检测灵敏度
    
    if test_results and len(test_results) > 0:
        result = test_results[0]
        if result.boxes is not None:
            print(f"测试结果: 检测到 {len(result.boxes)} 个苹果")
        else:
            print("测试结果: 未检测到苹果")

print("\n" + "=" * 60)
print("训练完成！")
print("下一步:")
print("1. 测试模型: python test_trained_model.py apple_quick.pt")
print("2. 更新应用中的模型路径")
print("=" * 60)