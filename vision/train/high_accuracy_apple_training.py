"""
高精度苹果检测模型训练脚本
使用160张各种场景的苹果照片进行训练
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import yaml

def prepare_dataset():
    """准备数据集，将160张苹果照片分配到训练集和验证集"""
    print("=== 准备苹果检测数据集 ===")
    
    # 数据集目录
    dataset_root = "YOLO-v8-app/dataset"
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")
    
    # 创建目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # 假设你已经将160张苹果照片放在以下目录
    apple_photos_dir = "apple_photos"  # 请将160张照片放在这个目录
    
    if not os.path.exists(apple_photos_dir):
        print(f"错误: 苹果照片目录不存在: {apple_photos_dir}")
        print("请将160张苹果照片放在 '{apple_photos_dir}' 目录下")
        return False
    
    # 获取所有苹果照片
    apple_photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        apple_photos.extend(list(Path(apple_photos_dir).glob(f"*{ext}")))
    
    print(f"找到 {len(apple_photos)} 张苹果照片")
    
    if len(apple_photos) < 50:
        print(f"警告: 照片数量不足，建议至少50张，当前只有 {len(apple_photos)} 张")
    
    # 随机打乱并分配训练集/验证集 (80%训练, 20%验证)
    random.shuffle(apple_photos)
    split_idx = int(len(apple_photos) * 0.8)
    train_photos = apple_photos[:split_idx]
    val_photos = apple_photos[split_idx:]
    
    print(f"训练集: {len(train_photos)} 张照片")
    print(f"验证集: {len(val_photos)} 张照片")
    
    # 复制照片到相应目录
    for i, photo_path in enumerate(train_photos):
        dst_path = os.path.join(images_dir, "train", f"apple_{i:04d}.jpg")
        shutil.copy2(photo_path, dst_path)
    
    for i, photo_path in enumerate(val_photos):
        dst_path = os.path.join(images_dir, "val", f"apple_val_{i:04d}.jpg")
        shutil.copy2(photo_path, dst_path)
    
    print("照片已复制到数据集目录")
    return True

def create_labels():
    """创建标签文件 - 需要手动标注或使用自动标注工具"""
    print("\n=== 创建标签文件 ===")
    print("重要: 需要为每张苹果照片创建标注文件")
    print("标签文件格式: class_id center_x center_y width height")
    print("对于苹果检测: class_id = 0")
    
    # 这里提供两种选择：
    print("\n请选择标注方式:")
    print("1. 手动标注 (推荐，精度最高)")
    print("2. 使用预训练模型自动标注 (快速，但需要人工修正)")
    
    choice = input("请输入选择 (1或2): ").strip()
    
    if choice == "1":
        print("\n手动标注步骤:")
        print("1. 使用标注工具如 LabelImg、CVAT 或 Roboflow")
        print("2. 标注每张照片中的苹果边界框")
        print("3. 导出为YOLO格式的txt文件")
        print("4. 将标签文件放在 labels/train/ 和 labels/val/ 目录")
        return "manual"
    elif choice == "2":
        print("\n将使用预训练模型进行自动标注...")
        return "auto"
    else:
        print("无效选择，使用自动标注")
        return "auto"

def auto_label_photos():
    """使用预训练模型自动标注苹果照片"""
    print("\n=== 自动标注苹果照片 ===")
    
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    # 加载预训练模型
    print("加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # 数据集目录
    dataset_root = "YOLO-v8-app/dataset"
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")
    
    # 处理训练集和验证集
    for split in ["train", "val"]:
        split_images_dir = os.path.join(images_dir, split)
        split_labels_dir = os.path.join(labels_dir, split)
        
        image_files = list(Path(split_images_dir).glob("*.jpg")) + \
                     list(Path(split_images_dir).glob("*.png")) + \
                     list(Path(split_images_dir).glob("*.jpeg"))
        
        print(f"\n处理 {split} 集: {len(image_files)} 张照片")
        
        for img_path in image_files:
            # 对应的标签文件路径
            label_path = os.path.join(split_labels_dir, f"{img_path.stem}.txt")
            
            # 如果标签文件已存在，跳过
            if os.path.exists(label_path):
                continue
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  警告: 无法读取图片 {img_path}")
                continue
            
            # 使用预训练模型检测
            results = model(img, conf=0.3, verbose=False)
            
            # 查找苹果检测结果
            apple_detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        cls_name = model.names[cls_id]
                        
                        # 检查是否是苹果或类似水果
                        if cls_name.lower() in ['apple', 'orange', 'banana', 'fruit']:
                            # 获取边界框坐标 (归一化)
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            img_h, img_w = img.shape[:2]
                            
                            # 转换为YOLO格式 (center_x, center_y, width, height)
                            center_x = (x1 + x2) / 2 / img_w
                            center_y = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            apple_detections.append((0, center_x, center_y, width, height))
            
            # 如果没有检测到苹果，创建一个中心位置的默认标注
            if not apple_detections:
                print(f"  警告: {img_path.name} 未检测到苹果，使用默认标注")
                apple_detections.append((0, 0.5, 0.5, 0.4, 0.4))
            
            # 保存标签文件
            with open(label_path, 'w') as f:
                for det in apple_detections:
                    f.write(f"{det[0]} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f}\n")
            
            print(f"  已创建: {label_path.name} ({len(apple_detections)} 个苹果)")
    
    print("\n自动标注完成！")
    print("注意: 自动标注可能不准确，建议人工检查修正")

def update_data_yaml():
    """更新data.yaml配置文件"""
    print("\n=== 更新数据集配置文件 ===")
    
    config_content = """# 苹果检测数据集配置文件
names:
- apple
nc: 1
path: ./dataset
train: images/train
val: images/val
"""
    
    config_path = "YOLO-v8-app/dataset/data.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"配置文件已更新: {config_path}")
    print("配置内容:")
    print(config_content)

def train_high_accuracy_model():
    """训练高精度苹果检测模型"""
    print("\n=== 开始训练高精度苹果检测模型 ===")
    
    # 训练参数
    training_config = {
        'data': 'YOLO-v8-app/dataset/data.yaml',
        'epochs': 100,  # 更多epoch以获得更好精度
        'imgsz': 640,   # 更高分辨率
        'batch': 8,     # 适当批次大小
        'workers': 2,   # 数据加载工作线程
        'device': 'cpu',  # 使用CPU训练
        'name': 'high_accuracy_apple',
        'patience': 20,  # 早停耐心
        'save': True,
        'save_period': 10,
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'AdamW',  # 更好的优化器
        'lr0': 0.001,  # 初始学习率
        'lrf': 0.01,   # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,    # 框损失权重
        'cls': 0.5,    # 分类损失权重
        'dfl': 1.5,    # DFL损失权重
        'hsv_h': 0.015,  # 图像增强
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("训练配置:")
    for key, value in training_config.items():
        if key != 'data':
            print(f"  {key}: {value}")
    
    print("\n开始训练...")
    print("注意: 训练可能需要30-60分钟，请耐心等待")
    
    # 加载模型
    model = YOLO('yolov8n.pt')  # 使用预训练模型
    
    # 开始训练
    results = model.train(**training_config)
    
    print("\n训练完成！")
    print(f"最佳模型保存在: runs/detect/high_accuracy_apple/weights/best.pt")
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("高精度苹果检测模型训练流程")
    print("=" * 60)
    
    # 步骤1: 准备数据集
    if not prepare_dataset():
        return
    
    # 步骤2: 创建标签
    label_method = create_labels()
    
    if label_method == "auto":
        auto_label_photos()
    else:
        print("\n请按照手动标注步骤完成标注后，按回车键继续...")
        input()
    
    # 步骤3: 更新配置文件
    update_data_yaml()
    
    # 步骤4: 开始训练
    train_high_accuracy_model()
    
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("下一步:")
    print("1. 测试新模型: python test_trained_model.py")
    print("2. 更新应用: 修改 fast_camera_app.py 中的模型路径")
    print("3. 运行应用: python -m streamlit run fast_camera_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()