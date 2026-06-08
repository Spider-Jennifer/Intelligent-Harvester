# 智能采收车

基于 YOLOv8 的苹果智能检测与采收机器人系统。项目结合机器视觉识别与机械结构设计，实现对苹果的实时检测、定位与自动采收。

## 项目结构

```
智能采收车/
├── mechanical/               # 机械模型（SolidWorks）
│   ├── 机械臂.SLDRW          #   机械臂工程图
│   ├── 机械臂.STEP            #   机械臂3D模型
│   ├── 气动剪刀结构.SLDRW     #   气动剪刀工程图
│   ├── 气动剪刀结构.STEP      #   气动剪刀3D模型
│   ├── 装配抓手.STEP          #   抓手装配体
│   └── 轮履结合.STEP          #   轮履结合机构
│
├── vision/                   # 视觉识别系统
│   ├── train/                #   训练脚本（22个）
│   │   ├── train_apple_final.py          #  最终训练脚本
│   │   ├── train_apple_ultra_precision.py # 超高精度训练
│   │   ├── high_accuracy_apple_training.py # 高精度训练
│   │   └── ...
│   ├── detect/               #   检测/推理脚本（9个）
│   │   ├── detect_apple_improved.py       #  改进版苹果检测
│   │   ├── detect_apple_ultra_precision.py # 超高精度检测
│   │   └── ...
│   ├── camera/               #   摄像头集成（13个）
│   │   ├── fast_camera_app.py            #  快速摄像头应用
│   │   ├── raspberry_pi_camera.py        #  树莓派摄像头
│   │   └── ...
│   ├── utils/                #   工具脚本（43个）
│   │   ├── check_env.py                 #   环境检查
│   │   ├── check_gpu.py                 #   GPU检查
│   │   ├── plot_training_metrics.py      #   训练指标可视化
│   │   └── ...
│   ├── models/               #   训练模型文件（Git LFS）
│   │   ├── apple_best.pt                 #  最佳模型
│   │   ├── apple_improved.pt             #  改进模型
│   │   └── yolov8n.pt                   #  YOLOv8n预训练模型
│   ├── config/               #   配置文件
│   │   └── dataset_fixed.yaml
│   └── test_results/         #   测试结果
│
├── data/                      # 本地训练数据（不纳入Git）
│   ├── apple_dataset/         #   完整苹果检测数据集
│   ├── apple_photos/          #   原始苹果照片
│   └── models/                #   训练过程模型检查点
│
├── datasets/                 # 数据集（仅少量示例）
│   ├── apple_dataset/        #   苹果检测数据集样本
│   │   ├── samples/images/   #     样本图片（2张）
│   │   ├── samples/labels/   #     样本标签
│   │   └── data.yaml         #     数据集配置
│   └── apple_photos/         #   苹果照片样本（2张）
│
├── docs/                     # 文档
│   ├── 苹果模型训练指南.md
│   ├── 苹果识别延迟分析报告.md
│   ├── 服务器连接修复指南.md
│   └── ...
│
├── media/                    # 媒体文件
│   ├── video/                #   视频
│   │   └── 总装配.mp4
│   └── images/               #   图片
│       ├── 屏幕截图 2026-04-21 104731.png
│       └── ...
│
├── scripts/                  # 启动/部署脚本
│   ├── start_project.bat     #   一键启动
│   ├── train_apple.bat       #   训练启动
│   └── ...
│
├── legacy/                   # 历史版本（参考）
│   ├── YOLO-v8-app/
│   └── yolov8-apple-detection-master/
│
├── .gitignore               # Git 忽略规则
├── .gitattributes           # Git LFS 配置
└── README.md
```

> **注意**：`data/` 目录存放完整训练数据集，已被 `.gitignore` 排除，不会纳入版本控制。仓库中的 `datasets/` 仅保留少量示例图片供参考。完整数据集请按下方说明获取。

## 使用方法

### 环境要求

- Python 3.8+
- CUDA（可选，用于GPU加速）
- 树莓派（可选，用于边缘部署）

### 安装依赖

```bash
pip install ultralytics opencv-python numpy pillow matplotlib streamlit
```

### 准备数据集

完整训练数据集需单独获取并放置到 `data/` 目录（该目录不纳入Git）：

```bash
# 1. 创建本地数据目录（仓库已包含 data/ 目录结构）
# 2. 将训练图片放入对应目录：
#    data/apple_dataset/images/train/   ← 训练集图片
#    data/apple_dataset/images/val/     ← 验证集图片
#    data/apple_dataset/labels/train/   ← 训练集标注（YOLO格式）
#    data/apple_dataset/labels/val/     ← 验证集标注
#    data/apple_photos/                 ← 原始苹果照片
```

**数据集格式要求：**
- 图片格式：`.jpg` / `.png`
- 标注格式：YOLO 格式 `.txt`，每行 `class_id x_center y_center width height`（归一化坐标）
- 数据集配置：参考 `datasets/apple_dataset/data.yaml`

**获取方式：**
- 自行采集苹果图片并使用 LabelImg / CVAT / Roboflow 标注
- 公共数据集：[Roboflow Apple Detection](https://universe.roboflow.com)
- 联系项目维护者获取原始数据集副本

### 训练模型

```bash
# 基础训练
python vision/train/train_apple_final.py

# 高精度训练
python vision/train/high_accuracy_apple_training.py

# 超高精度训练
python vision/train/train_apple_ultra_precision.py
```

### 运行检测

```bash
# 图片检测
python vision/detect/detect_apple_improved.py

# 实时摄像头检测
python vision/camera/fast_camera_app.py

# 树莓派摄像头
python vision/camera/raspberry_pi_camera.py
```

### Web 应用（推荐）

启动浏览器端苹果检测页面，支持图片上传和摄像头拍照：

```bash
streamlit run app.py
```

启动后浏览器访问：**http://localhost:8501**

> 首次启动如卡在邮箱输入提示，Ctrl+C 终止后重新运行即可（项目已配置跳过该提示）。

### 环境检查

```bash
python vision/utils/check_env.py      # 检查Python环境
python vision/utils/check_gpu.py      # 检查GPU状态
python vision/utils/check_model.py    # 检查模型文件
```

## 技术栈

| 技术 | 用途 |
|------|------|
| **YOLOv8** | 苹果目标检测与定位 |
| **Python** | 核心算法与系统集成 |
| **OpenCV** | 图像处理与摄像头采集 |
| **SolidWorks** | 机械结构设计与建模 |
| **Raspberry Pi** | 边缘端实时推理部署 |
| **Ultralytics** | YOLOv8训练框架 |
| **PyTorch** | 深度学习框架 |

## Git LFS

本项目使用 Git LFS 管理大文件：

- 模型文件（`.pt`）
- 视频文件（`.mp4`）
- CAD模型（`.STEP`, `.SLDDRW`）

克隆后请确保已安装 Git LFS：

```bash
git lfs install
git lfs pull
```
