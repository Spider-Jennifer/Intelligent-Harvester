# YOLOv8 苹果检测项目

## 项目概述
本项目使用YOLOv8模型进行苹果的高精度检测，包含完整的数据集、训练脚本和测试代码。

## 目录结构
```
├── Dataset/             # 数据集目录
│   ├── apple_dataset/   # 苹果数据集
│   ├── images/          # 划分后的图像数据(train/val/test)
│   ├── labels/          # 对应的标签文件
│   └── *.yaml           # 数据集配置文件
├── apple_detection_model/  # 训练模型目录
├── apple_detection_model2/ # 训练模型目录(包含best.pt)
├── train_apple_model.py    # 训练脚本
└── test_apple_model.py     # 测试脚本
```

## 环境配置

### 克隆项目
```bash
git clone https://github.com/Homekj/yolov8-apple-detection.git
cd yolov8-apple-detection
```

### 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/MacOS
# source .venv/bin/activate
```

### 安装依赖
```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
```

## 模型训练

### 数据准备
数据已包含在仓库中，无需额外准备。数据集划分如下：
- 训练集: Dataset/images/train
- 验证集: Dataset/images/val
- 测试集: Dataset/images/test

### 开始训练
```bash
python train_apple_model.py
```

训练参数可以在脚本中修改，如：
- model: 模型类型
- data: 数据集配置文件
- epochs: 训练轮数
- imgsz: 图像大小
- batch: 批次大小

### 继续训练
如果需要继续之前的训练：
```bash
# 编辑train_apple_model.py，修改resume参数为True
# 确保model参数指向last.pt文件
python train_apple_model.py
```

## 模型测试
```bash
python test_apple_model.py
```

## 高精度训练建议
1. 增加训练轮数(epochs)
2. 使用更大的图像大小(imgsz)
3. 尝试不同的模型规模(n, s, m, l, x)
4. 使用数据增强技术
5. 调整学习率和优化器参数

## 注意事项
- 训练可能需要较长时间，建议在GPU环境下进行
- 模型权重文件较大，请确保有足够的磁盘空间
- 如果数据集需要更新，请保持目录结构一致