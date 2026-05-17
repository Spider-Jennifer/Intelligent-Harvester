#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用少量标签进行苹果检测模型训练
即使标签文件很少，也可以使用预训练模型进行微调
"""

import os
import sys
import glob
import random
from pathlib import Path
from ultralytics import YOLO

def create_minimal_labels():
    """为部分图片创建最小标签文件"""
    dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
    
    # 训练集图片目录
    train_img_dir = os.path.join(dataset_path, "images", "train")
    train_label_dir = os.path.join(dataset_path, "labels", "train")
    
    # 验证集图片目录
    val_img_dir = os.path.join(dataset_path, "images", "val")
    val_label_dir = os.path.join(dataset_path, "labels", "val")
    
    # 确保标签目录存在
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    created_count = 0
    
    # 为训练集创建一些简单标签（假设苹果在图片中央）
    train_images = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    train_images += glob.glob(os.path.join(train_img_dir, "*.png"))
    
    # 随机选择10%的图片创建标签
    sample_size = max(1, len(train_images) // 10)
    sampled_images = random.sample(train_images, min(sample_size, 20))
    
    for img_path in sampled_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(train_label_dir, f"{img_name}.txt")
        
        # 如果标签文件不存在，创建一个简单的标签
        if not os.path.exists(label_path):
            # 假设苹果在图片中央，占图片的30-50%
            with open(label_path, 'w') as f:
                # class_id x_center y_center width height
                # 假设苹果在中央，占40%的宽度和高度
                f.write("0 0.5 0.5 0.4 0.4\n")
            created_count += 1
    
    # 为验证集创建一些简单标签
    val_images = glob.glob(os.path.join(val_img_dir, "*.jpg"))
    val_images += glob.glob(os.path.join(val_img_dir, "*.png"))
    
    if val_images:
        sample_size = max(1, len(val_images) // 5)
        sampled_val = random.sample(val_images, min(sample_size, 10))
        
        for img_path in sampled_val:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(val_label_dir, f"{img_name}.txt")
            
            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.4 0.4\n")
                created_count += 1
    
    return created_count

def train_with_transfer_learning():
    """使用迁移学习训练模型"""
    print("=" * 60)
    print("苹果检测模型训练 - 迁移学习版")
    print("=" * 60)
    
    # 检查并创建最小标签
    print("检查数据集标签...")
    created = create_minimal_labels()
    if created > 0:
        print(f"为 {created} 张图片创建了简单标签文件")
        print("注意: 这些是假设的标签，实际训练效果取决于真实标注")
    else:
        print("使用现有的标签文件")
    
    print()
    
    # 设置路径
    dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # 检查预训练模型
    pretrained_model = "yolov8n.pt"
    if not os.path.exists(pretrained_model):
        print(f"警告: 找不到预训练模型 {pretrained_model}")
        print("将尝试从网络下载...")
    
    print(f"数据集配置文件: {data_yaml}")
    print(f"预训练模型: {pretrained_model}")
    print()
    
    try:
        # 加载预训练模型
        print("正在加载YOLOv8n预训练模型...")
        model = YOLO(pretrained_model)
        
        # 训练参数配置 - 针对少量标签优化
        print("开始迁移学习训练...")
        print("训练参数:")
        print("  - 训练轮数: 30 (少量标签，避免过拟合)")
        print("  - 批次大小: 4 (小批次，适合少量数据)")
        print("  - 图像尺寸: 416 (中等分辨率)")
        print("  - 设备: CPU")
        print("  - 学习率: 较低 (0.001，微调而不是重新训练)")
        print()
        
        # 开始训练
        results = model.train(
            data=data_yaml,           # 数据集配置文件
            epochs=30,                # 训练轮数 - 少量数据，避免过拟合
            imgsz=416,                # 图像尺寸
            batch=4,                  # 批次大小 - 小批次
            name='apple_transfer_learning',  # 实验名称
            save=True,                # 保存检查点
            plots=True,               # 生成图表
            device='cpu',             # 使用CPU训练
            patience=15,              # 早停耐心值
            lr0=0.001,                # 较低的学习率 - 微调
            lrf=0.01,                 # 最终学习率因子
            momentum=0.9,             # 动量
            weight_decay=0.0001,      # 较小的权重衰减
            warmup_epochs=2,          # 热身轮数
            warmup_momentum=0.8,      # 热身动量
            box=5.0,                  # 边界框损失权重
            cls=0.3,                  # 分类损失权重
            dfl=1.0,                  # DFL损失权重
            verbose=True,             # 显示训练进度
            cos_lr=True,              # 使用余弦学习率调度
            dropout=0.1,              # 添加dropout防止过拟合
            freeze=10,                # 冻结前10层，只训练后面层
        )
        
        print()
        print("=" * 60)
        print("迁移学习训练完成!")
        print("=" * 60)
        
        # 显示训练结果保存位置
        runs_dir = Path("runs/detect/apple_transfer_learning")
        if runs_dir.exists():
            print(f"训练结果保存在: {runs_dir}")
            
            # 检查生成的模型文件
            weights_dir = runs_dir / "weights"
            if weights_dir.exists():
                best_model = weights_dir / "best.pt"
                last_model = weights_dir / "last.pt"
                
                if best_model.exists():
                    print(f"最佳模型: {best_model}")
                    print(f"文件大小: {best_model.stat().st_size / 1024 / 1024:.2f} MB")
                    
                    # 复制到项目目录
                    import shutil
                    target_path = "apple_transfer_best.pt"
                    shutil.copy2(best_model, target_path)
                    print(f"已复制到: {target_path}")
                
                if last_model.exists():
                    print(f"最终模型: {last_model}")
        
        print()
        print("训练说明:")
        print("1. 由于标签文件较少，模型主要依赖预训练知识")
        print("2. 实际检测精度可能不如使用完整标注的数据集")
        print("3. 建议后续补充更多准确标注")
        print()
        print("下一步:")
        print("1. 测试模型: python test_trained_model.py")
        print("2. 使用新模型: 修改fast_camera_app.py中的模型路径")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        print()
        print("备选方案:")
        print("1. 直接使用预训练模型 yolov8n.pt")
        print("2. 使用更小的模型 yolov8n，减少内存需求")
        print("3. 降低训练参数: batch=2, imgsz=320")

def main():
    """主函数"""
    print("苹果检测模型训练 - 少量标签解决方案")
    print()
    print("选项:")
    print("1. 使用迁移学习训练 (推荐)")
    print("2. 直接使用预训练模型")
    print()
    
    choice = input("请选择 (1 或 2, 默认 1): ").strip()
    
    if choice == "2":
        print()
        print("直接使用预训练模型 yolov8n.pt")
        print("该模型已经具备通用物体检测能力")
        print("可以在苹果检测任务上直接使用")
        print()
        print("使用方法:")
        print("1. 在 fast_camera_app.py 中确保 model = YOLO('yolov8n.pt')")
        print("2. 运行 python fast_camera_app.py")
    else:
        train_with_transfer_learning()

if __name__ == "__main__":
    main()