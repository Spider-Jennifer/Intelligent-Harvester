# YOLOv8 苹果检测模型训练恢复指南

本文档提供了在更换带有英伟达GPU的新设备后，如何恢复苹果检测模型训练的详细步骤。

## 项目进度概述

当前项目已完成以下配置：

1. ✅ 创建了苹果类别配置文件 `data_Apple.yaml`
2. ✅ 准备了苹果专用训练数据集，路径为 `D:/python/yolov8-fruit/Dataset/apple_dataset`
3. ✅ 开始训练苹果检测模型，使用本地预训练权重（已保存部分训练进度）
4. ✅ 更新了YOLO-v8-app应用配置，以支持使用新训练的模型
5. ✅ 创建了测试脚本 `test_apple_model.py` 用于评估模型性能

## 新设备环境配置

### 1. 克隆项目

首先将整个项目目录复制到新设备上。项目结构如下：

```
yolov8-fruit/
├── Dataset/                 # 数据集目录
│   ├── apple_dataset/       # 苹果数据集
│   └── data_Apple.yaml      # 苹果数据集配置
├── YOLO-v8-app/             # YOLOv8应用目录
├── apple_detection_model2/  # 当前训练进度目录（包含已训练的权重）
├── train_apple_model.py     # 训练脚本
└── test_apple_model.py      # 测试脚本
```

### 2. 安装依赖

确保安装了所有必要的依赖，特别是支持CUDA的PyTorch和Ultralytics YOLOv8：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# 安装PyTorch（选择与您的CUDA版本匹配的版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装Ultralytics YOLOv8
pip install ultralytics

# 安装其他依赖
pip install opencv-python matplotlib pandas
```

### 3. 验证GPU可用性

确保PyTorch能正确识别您的NVIDIA GPU：

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU设备数量: {torch.cuda.device_count()}")
print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
```

## 从上次进度恢复训练

当前训练进度保存在 `apple_detection_model2/weights/last.pt` 中，这是模型训练中断时的最后一个检查点。

### 修改训练脚本以支持恢复

确保 `train_apple_model.py` 脚本可以从检查点恢复：

1. 打开 `train_apple_model.py` 文件
2. 确保训练代码中使用了 `resume=True` 参数
3. 设置 `model=YOLO('apple_detection_model2/weights/last.pt')` 来加载最后一个检查点

### 运行恢复训练

在新设备上运行训练脚本，它将自动从上次中断的地方继续：

```bash
python train_apple_model.py
```

## 训练参数优化（GPU版本）

由于现在使用GPU训练，您可以调整以下参数以加快训练速度：

1. **批量大小**：增加 `batch_size` 参数（例如从默认值增加到16或32，根据GPU内存而定）
2. **工作线程数**：增加 `workers` 参数以加速数据加载
3. **混合精度训练**：启用 `amp=True` 进行自动混合精度训练

您可以修改 `train_apple_model.py` 中的训练参数：

```python
# 示例：GPU优化参数
model.train(
    data='Dataset/data_Apple.yaml',
    epochs=100,
    batch=16,  # 增加批量大小
    workers=8,  # 增加工作线程数
    amp=True,   # 启用自动混合精度训练
    device=0,   # 指定使用第一个GPU设备
    resume=True # 从检查点恢复训练
)
```

## 验证训练效果

训练完成后，您可以使用以下命令测试模型性能：

```bash
python test_apple_model.py
```

测试结果将保存在 `test_results/` 目录中，显示带边界框的检测结果。

## 注意事项

1. 确保新设备上的数据集路径与配置文件中的路径一致
2. 如果需要更改路径，请更新 `data_Apple.yaml` 中的 `path` 字段
3. GPU训练可能需要调整学习率以获得更好的性能
4. 定期检查 `apple_detection_model2/weights/` 目录中的 `best.pt` 文件，这是性能最好的模型版本

祝您在新设备上训练顺利！