@echo off
chcp 65001 >nul
echo ========================================
echo   高精度苹果检测模型训练准备
echo ========================================
echo.

echo 步骤1: 请将160张苹果照片放在以下目录:
echo   c:\Users\李晨鑫\Desktop\yolov8\apple_photos\
echo.
echo 如果没有该目录，请创建它:
echo   mkdir apple_photos
echo.
echo 然后将所有苹果照片复制到该目录
echo.
pause

echo.
echo 步骤2: 检查照片数量...
python -c "
import os
from pathlib import Path

photo_dir = 'apple_photos'
if os.path.exists(photo_dir):
    photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        photos.extend(list(Path(photo_dir).glob(f'*{ext}')))
    
    print(f'找到 {len(photos)} 张苹果照片')
    
    if len(photos) >= 50:
        print('✓ 照片数量足够，可以开始训练')
    else:
        print('⚠ 警告: 照片数量不足，建议至少50张')
else:
    print('✗ 错误: 找不到 apple_photos 目录')
"
echo.
pause

echo.
echo 步骤3: 准备数据集结构...
python -c "
import os

# 创建数据集目录结构
dirs = [
    'YOLO-v8-app/dataset/images/train',
    'YOLO-v8-app/dataset/images/val',
    'YOLO-v8-app/dataset/labels/train',
    'YOLO-v8-app/dataset/labels/val'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f'创建目录: {dir_path}')

print('✓ 数据集目录结构已创建')
"
echo.
pause

echo.
echo 步骤4: 复制照片到数据集...
python -c "
import os
import shutil
import random
from pathlib import Path

# 源目录和目标目录
src_dir = 'apple_photos'
dst_root = 'YOLO-v8-app/dataset/images'

if not os.path.exists(src_dir):
    print('✗ 错误: 找不到苹果照片目录')
    exit(1)

# 获取所有照片
photos = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
    photos.extend(list(Path(src_dir).glob(f'*{ext}')))

print(f'找到 {len(photos)} 张照片')

if len(photos) == 0:
    print('✗ 错误: 没有找到照片')
    exit(1)

# 随机打乱
random.shuffle(photos)

# 分配训练集和验证集 (80%/20%)
split_idx = int(len(photos) * 0.8)
train_photos = photos[:split_idx]
val_photos = photos[split_idx:]

print(f'训练集: {len(train_photos)} 张')
print(f'验证集: {len(val_photos)} 张')

# 复制训练集照片
for i, photo_path in enumerate(train_photos):
    dst_path = os.path.join(dst_root, 'train', f'apple_{i:04d}{photo_path.suffix}')
    shutil.copy2(photo_path, dst_path)

# 复制验证集照片
for i, photo_path in enumerate(val_photos):
    dst_path = os.path.join(dst_root, 'val', f'apple_val_{i:04d}{photo_path.suffix}')
    shutil.copy2(photo_path, dst_path)

print('✓ 照片已复制到数据集')
"
echo.
pause

echo.
echo 步骤5: 创建自动标注...
echo 将使用预训练模型自动标注苹果位置
echo 注意: 自动标注可能不完美，训练后精度会提高
echo.
pause

python -c "
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

print('加载预训练模型...')
model = YOLO('yolov8n.pt')

# 处理训练集和验证集
for split in ['train', 'val']:
    images_dir = f'YOLO-v8-app/dataset/images/{split}'
    labels_dir = f'YOLO-v8-app/dataset/labels/{split}'
    
    image_files = list(Path(images_dir).glob('*.jpg')) + \
                 list(Path(images_dir).glob('*.png')) + \
                 list(Path(images_dir).glob('*.jpeg'))
    
    print(f'\\n处理 {split} 集: {len(image_files)} 张照片')
    
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
                f.write(f'{det[0]} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f}\\n')
        
        processed += 1
        if processed % 10 == 0:
            print(f'  已处理 {processed}/{len(image_files)}')

print('\\n✓ 自动标注完成！')
"
echo.
pause

echo.
echo 步骤6: 创建配置文件...
python -c "
config_content = '''# 苹果检测数据集配置文件
names:
- apple
nc: 1
path: ./dataset
train: images/train
val: images/val
'''

config_path = 'YOLO-v8-app/dataset/data.yaml'
with open(config_path, 'w') as f:
    f.write(config_content)

print(f'配置文件已创建: {config_path}')
print('配置内容:')
print(config_content)
"
echo.
pause

echo.
echo ========================================
echo   准备完成！现在可以开始训练
echo ========================================
echo.
echo 运行以下命令开始训练:
echo   python train_apple_high_quality.py
echo.
echo 或者直接运行训练批处理文件:
echo   train_apple_high_quality.bat
echo.
pause