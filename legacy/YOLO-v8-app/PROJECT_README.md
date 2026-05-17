# YOLOv8 苹果检测应用

## 项目概述
这是一个基于 YOLOv8 的苹果检测 Web 应用，使用 Streamlit 构建交互式界面。

## 功能特性
- 🖼️ 图片上传检测
- 🎥 视频文件检测
- 📹 实时摄像头检测
- 🍎 专门训练的苹果检测模型

## 项目结构
```
YOLOv8-app-master/
├── app.py              # 主应用文件
├── config.py           # 配置文件
├── utils.py            # 工具函数
├── requirements.txt    # 依赖包列表
├── weights/            # 模型权重目录
│   └── best.pt        # 苹果检测模型
└── pic_bed/           # 图片存储目录
```

## 环境要求
- Python 3.8+
- 虚拟环境推荐

## 安装步骤

### 1. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 模型配置
确保 `weights/best.pt` 文件存在（苹果检测模型）

## 运行应用

### 方式一：使用 Streamlit（推荐）
```bash
streamlit run app.py
```

### 方式二：直接运行 Python
```bash
python app.py
```

应用将在浏览器中打开：http://localhost:8501

## 使用说明
1. 启动应用后，在侧边栏选择检测源（图片/视频/摄像头）
2. 调整置信度阈值
3. 上传文件或开启摄像头
4. 查看检测结果

## 技术栈
- **后端**: Python, Ultralytics YOLOv8
- **前端**: Streamlit
- **计算机视觉**: OpenCV, PIL
- **深度学习**: PyTorch

## 模型信息
- 模型类型: YOLOv8
- 训练数据: 苹果检测数据集
- 输入尺寸: 640x640
- 模型大小: 49.66 MB

## 故障排除

### 常见问题
1. **ModuleNotFoundError**: 确保已安装所有依赖
2. **模型加载失败**: 检查 `weights/best.pt` 是否存在
3. **PyTorch 权重加载错误**: 已在 utils.py 中处理

### 依赖版本兼容性
- numpy: 2.x
- torch: 2.x
- opencv-python: 4.x
- streamlit: 1.x

## 开发者信息
- 基于 Ultralytics YOLOv8 框架
- 使用 Streamlit 构建 Web 界面
- 支持 GPU 加速（如果可用）

## 许可证
请参考项目根目录的 LICENSE 文件