#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8苹果检测项目 - 自定义模型训练脚本
功能：使用用户准备的苹果数据集训练YOLOv8目标检测模型
作者：AI Assistant
版本：1.0
使用方法：python train_custom_model.py
前置条件：需要先运行create_dataset.py并准备好标注数据
"""

from ultralytics import YOLO  # 导入YOLOv8模型库
import shutil                # 文件操作库，用于复制模型文件
from pathlib import Path    # 路径操作库

def train_custom_model():
    """
    训练自定义苹果检测模型
    功能：加载预训练YOLOv8模型，使用苹果数据集进行微调训练
    训练流程：
    1. 加载YOLOv8n预训练模型（迁移学习）
    2. 使用苹果数据集进行训练
    3. 验证模型性能
    4. 保存最佳权重到项目目录
    
    返回值：None - 训练过程会自动保存结果
    """
    
    print("=" * 50)
    print("开始训练YOLOv8苹果检测模型")
    print("=" * 50)
    
    # 步骤1：加载预训练模型（迁移学习）
    # 使用YOLOv8n（nano版本）作为基础，速度快，适合快速原型开发
    print("正在加载YOLOv8n预训练模型...")
    model = YOLO('yolov8n.pt')
    
    # 步骤2：配置训练参数并开始训练
    print("开始训练模型...")
    results = model.train(
        data='dataset/data.yaml',    # 数据集配置文件路径
        epochs=100,                  # 训练轮数：100轮适合充分学习
        imgsz=640,                   # 输入图像尺寸：640x640是YOLOv8标准尺寸
        batch=16,                    # 批次大小：16适合中等配置CPU/GPU
        name='apple_detection',      # 实验名称：用于保存训练结果
        save=True,                   # 保存训练过程中的检查点
        plots=True,                   # 生成训练过程的可视化图表
        device='cpu',                # 训练设备：'cpu'或'cuda'（如果有GPU）
        patience=50,                 # 早停耐心值：50轮无改善则停止
        lr0=0.01,                    # 初始学习率：0.01适合从头训练
        optimizer='SGD',            # 优化器：SGD稳定可靠
        augment=True,                # 数据增强：开启提升泛化能力
        pretrained=True,             # 使用预训练权重
    )
    
    # 3. 验证模型
    results = model.val()
    
    # 4. 导出模型
    model.export(format='onnx')  # 可选：导出为ONNX格式
    
    print("训练完成！最佳模型保存在 runs/detect/apple_detection/weights/best.pt")

if __name__ == "__main__":
    train_custom_model()