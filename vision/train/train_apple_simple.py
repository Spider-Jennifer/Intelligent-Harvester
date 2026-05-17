#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
苹果检测模型训练 - 简化版
使用YOLOv8n预训练模型进行微调训练
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def check_dataset():
    """检查数据集是否完整"""
    dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
    
    # 检查data.yaml文件
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到数据集配置文件 {data_yaml}")
        return False
    
    # 检查图片目录
    train_img_dir = os.path.join(dataset_path, "images", "train")
    val_img_dir = os.path.join(dataset_path, "images", "val")
    
    if not os.path.exists(train_img_dir):
        print(f"错误: 找不到训练集图片目录 {train_img_dir}")
        return False
    
    if not os.path.exists(val_img_dir):
        print(f"错误: 找不到验证集图片目录 {val_img_dir}")
        return False
    
    # 检查标签目录
    train_label_dir = os.path.join(dataset_path, "labels", "train")
    val_label_dir = os.path.join(dataset_path, "labels", "val")
    
    if not os.path.exists(train_label_dir):
        print(f"警告: 找不到训练集标签目录 {train_label_dir}")
        print("将使用预训练模型进行训练，但效果可能不佳")
    
    if not os.path.exists(val_label_dir):
        print(f"警告: 找不到验证集标签目录 {val_label_dir}")
        print("将使用预训练模型进行训练，但效果可能不佳")
    
    return True

def train_model():
    """训练模型"""
    print("=" * 60)
    print("苹果检测模型训练 - 简化版")
    print("=" * 60)
    
    # 检查数据集
    if not check_dataset():
        print("数据集检查失败，请确保数据集完整")
        return
    
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
        print("正在加载预训练模型...")
        model = YOLO(pretrained_model)
        
        # 训练参数配置
        print("开始训练模型...")
        print("训练参数:")
        print("  - 训练轮数: 50 (快速训练)")
        print("  - 批次大小: 8 (适合CPU训练)")
        print("  - 图像尺寸: 320 (降低分辨率，加快训练)")
        print("  - 设备: CPU")
        print()
        
        # 开始训练
        results = model.train(
            data=data_yaml,           # 数据集配置文件
            epochs=50,                # 训练轮数
            imgsz=320,                # 图像尺寸
            batch=8,                  # 批次大小
            name='apple_detection_simple',  # 实验名称
            save=True,                # 保存检查点
            plots=True,               # 生成图表
            device='cpu',             # 使用CPU训练
            patience=20,              # 早停耐心值
            lr0=0.01,                 # 初始学习率
            lrf=0.01,                 # 最终学习率因子
            momentum=0.937,           # 动量
            weight_decay=0.0005,      # 权重衰减
            warmup_epochs=3,          # 热身轮数
            warmup_momentum=0.8,      # 热身动量
            box=7.5,                  # 边界框损失权重
            cls=0.5,                  # 分类损失权重
            dfl=1.5,                  # DFL损失权重
            verbose=True              # 显示训练进度
        )
        
        print()
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        # 显示训练结果保存位置
        runs_dir = Path("runs/detect/apple_detection_simple")
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
                
                if last_model.exists():
                    print(f"最终模型: {last_model}")
                    print(f"文件大小: {last_model.stat().st_size / 1024 / 1024:.2f} MB")
        
        print()
        print("下一步:")
        print("1. 使用训练好的模型进行测试: python test_trained_model.py")
        print("2. 将模型复制到项目目录: copy runs\\detect\\apple_detection_simple\\weights\\best.pt apple_best.pt")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print()
        print("可能的原因:")
        print("1. 数据集标签文件不完整")
        print("2. 内存不足")
        print("3. 依赖库未正确安装")
        print()
        print("建议:")
        print("1. 确保所有图片都有对应的标签文件")
        print("2. 减少批次大小: batch=4")
        print("3. 降低图像尺寸: imgsz=256")

def test_trained_model():
    """测试训练好的模型"""
    print("=" * 60)
    print("测试训练好的模型")
    print("=" * 60)
    
    model_path = "runs/detect/apple_detection_simple/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的模型 {model_path}")
        print("请先运行训练脚本")
        return
    
    try:
        # 加载训练好的模型
        model = YOLO(model_path)
        
        # 使用验证集进行测试
        dataset_path = r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset"
        val_images = os.path.join(dataset_path, "images", "val")
        
        if os.path.exists(val_images):
            # 获取验证集图片
            import glob
            val_files = glob.glob(os.path.join(val_images, "*.jpg"))[:3]  # 只测试前3张
            
            if val_files:
                print(f"测试 {len(val_files)} 张验证集图片...")
                
                for img_path in val_files:
                    print(f"处理: {os.path.basename(img_path)}")
                    
                    # 进行预测
                    results = model(img_path, conf=0.5, verbose=False)
                    
                    # 显示结果
                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None:
                            detections = len(result.boxes)
                            print(f"  检测到 {detections} 个苹果")
                        else:
                            print("  未检测到苹果")
                    else:
                        print("  未检测到苹果")
            else:
                print("验证集没有图片")
        else:
            print("验证集目录不存在")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_trained_model()
    else:
        train_model()