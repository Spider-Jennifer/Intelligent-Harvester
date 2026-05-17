#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8苹果检测项目 - 数据集创建脚本
功能：自动生成YOLOv8训练所需的标准数据集目录结构和配置文件
作者：AI Assistant
版本：1.0
使用方法：python create_dataset.py
"""

import os          # 操作系统接口库，用于文件和目录操作
import yaml        # YAML文件处理库，用于创建配置文件
from pathlib import Path  # 现代化的路径操作库

def create_dataset_structure():
    """
    创建YOLOv8标准数据集目录结构
    功能：按照YOLOv8要求创建完整的目录层次结构
    目录结构：
    dataset/
    ├── images/
    │   ├── train/    # 训集图片
    │   └── val/      # 验证集图片
    └── labels/
        ├── train/    # 训练集标签
        └── val/      # 验证集标签
    
    返回值：None - 直接创建目录结构
    """
    
    # 定义YOLOv8标准目录结构列表
    dirs = [
        'dataset/images/train',  # 训练图片目录
        'dataset/images/val',    # 验证图片目录
        'dataset/labels/train',  # 训练标签目录
        'dataset/labels/val'     # 验证标签目录
    ]
    
    # 遍历目录列表，逐个创建目录
    for dir_path in dirs:
        # 使用Path.mkdir创建目录，parents=True自动创建父目录
        # exist_ok=True如果目录已存在不报错
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")  # 打印创建进度

def create_data_yaml():
    """
    创建YOLOv8数据集配置文件
    功能：生成data.yaml配置文件，告诉YOLOv8数据集的位置和类别信息
    配置文件包含：
    - 数据集根目录路径
    - 训练和验证图片的相对路径
    - 类别数量和名称映射
    
    返回值：None - 直接创建配置文件
    """
    
    # 构建YOLOv8数据集配置字典
    data_config = {
        'path': './dataset',        # 数据集根目录的相对路径
        'train': 'images/train',    # 训练图片相对于根目录的路径
        'val': 'images/val',        # 验证图片相对于根目录的路径
        'nc': 1,                    # 类别数量：1个类别（苹果）
        'names': ['apple']          # 类别名称列表：索引0对应'apple'
    }
    
    # 将配置字典写入YAML文件
    with open('dataset/data.yaml', 'w', encoding='utf-8') as f:
        # 使用yaml.dump将字典转换为YAML格式
        # default_flow_style=False：使用块格式，更易读
        # allow_unicode=True：支持中文字符
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print("创建数据配置文件: dataset/data.yaml")  # 打印创建完成信息

def create_sample_labels():
    """
    创建YOLOv8格式的示例标签文件
    功能：生成标准的YOLO标签文件示例，帮助用户理解标签格式
    YOLO标签格式：class_id x_center y_center width height
    所有坐标都是相对于图像宽高的归一化值（0-1之间）
    
    示例说明：
    - class_id=0：表示苹果类别
    - x_center=0.5, y_center=0.5：苹果中心在图像中心
    - width=0.3, height=0.4：苹果占图像30%宽度和40%高度
    
    返回值：None - 直接创建示例文件
    """
    
    # YOLO格式标签示例：class_id x_center y_center width height
    # 0: 苹果类别ID
    # 0.5 0.5: 苹果中心在图像中心（归一化坐标）
    # 0.3 0.4: 苹果的宽度和高度（归一化尺寸）
    sample_label = "0 0.5 0.5 0.3 0.4\n"  # 苹果位于图像中心
    
    # 创建训练集标签示例文件
    with open('dataset/labels/train/sample.txt', 'w') as f:
        f.write(sample_label)  # 写入示例标签
    
    # 创建验证集标签示例文件
    with open('dataset/labels/val/sample.txt', 'w') as f:
        f.write(sample_label)  # 写入相同的示例标签
    
    print("创建示例标签文件")  # 打印创建完成信息

def print_instructions():
    """
    打印详细的数据集准备和使用说明
    功能：为用户提供完整的从数据准备到模型训练的指导
    包含：数据收集、标注、训练、部署的完整流程
    
    返回值：None - 直接打印说明信息
    """
    
    # 多行字符串包含详细的使用说明
    instructions = """
    === YOLOv8苹果检测数据集准备说明 ===
    
    1. 收集苹果图片数据：
       - 训练集：建议100-1000张不同角度、光照的苹果图片
       - 验证集：建议20-200张苹果图片，用于模型验证
       - 图片格式：支持jpg, jpeg, png, bmp等常见格式
       - 图片尺寸：建议640x640像素，YOLOv8会自动调整
    
    2. 数据标注（关键步骤）：
       - 安装标注工具：pip install labelImg
       - 启动标注工具：labelImg dataset/images/train dataset/labels/train
       - 标注方法：用矩形框精确框出每个苹果
       - 保存格式：选择YOLO格式（自动生成归一化坐标）
       - 标注要求：确保每个苹果都被准确标注
    
    3. 数据增强（可选，推荐）：
       - 几何变换：旋转、翻转、缩放、平移
       - 颜色变换：亮度、对比度、饱和度调整
       - 推荐库：Albumentations或YOLOv8内置增强
    
    4. 开始模型训练：
       - 运行命令：python train_custom_model.py
       - 训练时间：根据数据量，通常30分钟-2小时
       - 监控指标：关注mAP、loss曲线
    
    5. 训练完成后部署：
       - 最佳权重位置：runs/detect/apple_detection/weights/best.pt
       - 部署到项目：cp runs/detect/apple_detection/weights/best.pt weights/
       - 重启应用：streamlit run app.py
    
    6. 性能优化建议：
       - 如果准确率低：增加训练数据或调整超参数
       - 如果速度慢：使用YOLOv8n或减小输入尺寸
       - 如果误检多：调整置信度阈值或增加负样本
    """
    
    print(instructions)  # 打印完整的使用说明

if __name__ == "__main__":
    """
    主程序入口点
    功能：当直接运行此脚本时，按顺序执行所有数据集创建步骤
    执行流程：
    1. 创建目录结构
    2. 生成配置文件
    3. 创建示例标签
    4. 显示使用说明
    """
    print("=" * 50)
    print("YOLOv8苹果检测数据集创建工具")
    print("=" * 50)
    
    # 按顺序执行数据集创建步骤
    create_dataset_structure()  # 步骤1：创建目录结构
    create_data_yaml()          # 步骤2：创建配置文件
    create_sample_labels()      # 步骤3：创建示例标签
    print_instructions()        # 步骤4：显示使用说明
    
    print("\n" + "=" * 50)
    print("数据集创建完成！")
    print("请按照上述说明准备训练数据。")
    print("=" * 50)