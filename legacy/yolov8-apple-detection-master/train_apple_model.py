import os
import argparse
import torch
from ultralytics import YOLO

# 设置数据集配置文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, 'Dataset', 'data_Apple.yaml')

# 创建保存模型的目录
MODEL_DIR = os.path.join(BASE_DIR, 'apple_detection_model')
os.makedirs(MODEL_DIR, exist_ok=True)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8苹果检测模型训练')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='模型类型: yolov8n.pt, yolov8s.pt, yolov8m.pt等')
    parser.add_argument('--resume', action='store_true', help='是否继续训练')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=800, help='图像大小')
    parser.add_argument('--name', type=str, default='apple_detection_model_optimized', help='模型名称')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    return parser.parse_args()

# 开始训练
def train_apple_model():
    args = parse_args()
    
    print("开始训练苹果检测模型...")
    print(f"数据集配置: {DATASET_YAML}")
    print(f"模型保存目录: {MODEL_DIR}")
    
    # 根据是否继续训练选择加载方式
    if args.resume:
        # 加载上次训练的最后一个模型
        last_model_path = os.path.join(BASE_DIR, args.name, 'weights', 'last.pt')
        if os.path.exists(last_model_path):
            print(f"继续训练，加载模型: {last_model_path}")
            model = YOLO(last_model_path)
        else:
            print("未找到上次训练的模型，使用预训练模型开始新训练")
            model = YOLO(args.model)  # 使用官方预训练模型
    else:
        # 加载预训练模型
        print(f"使用预训练模型: {args.model}")
        model = YOLO(args.model)  # 使用官方预训练模型
    
    # 自动检测设备：优先使用GPU，如果没有则使用CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU (CUDA:0)' if torch.cuda.is_available() else 'CPU'}")
    
    # 开始训练
    results = model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        save_dir=BASE_DIR,
        project=BASE_DIR,
        workers=0,  # Windows系统可能需要设置为0
        patience=args.patience,
        resume=args.resume,
        device=device,  # 自动选择设备
        # 高精度训练参数
        val=True,
        save=True,
        save_period=5,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,  # 增加权重衰减防止过拟合
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # 增强数据增强参数
        augment=True,
        degrees=10.0,  # 添加旋转增强
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0)
    
    print("训练完成！")
    print(f"最佳模型保存路径: {os.path.join(BASE_DIR, args.name, 'weights', 'best.pt')}")
    
    # 评估模型
    print("\n评估模型性能...")
    metrics = model.val()
    print(f"模型mAP50: {metrics.box.map50}")

if __name__ == '__main__':
    train_apple_model()