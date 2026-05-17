"""
高质量苹果检测模型训练脚本
简化版本，专注于训练过程
"""

from ultralytics import YOLO
import os

def train_apple_model():
    """训练苹果检测模型"""
    print("开始训练高质量苹果检测模型...")
    
    # 检查数据集配置
    data_yaml = "YOLO-v8-app/dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        print("请确保已经准备好数据集和标签文件")
        return
    
    print(f"使用配置文件: {data_yaml}")
    
    # 训练参数 - 针对高质量苹果检测优化
    training_args = {
        'data': data_yaml,
        'epochs': 80,           # 足够多的epoch
        'imgsz': 640,           # 高分辨率
        'batch': 4,             # 小批次，适合CPU
        'workers': 0,           # Windows上设为0避免问题
        'device': 'cpu',        # 使用CPU
        'name': 'apple_high_quality',
        'patience': 15,         # 早停
        'save': True,
        'pretrained': True,     # 使用预训练权重
        'optimizer': 'Adam',    # Adam优化器
        'lr0': 0.0005,          # 较低学习率，更稳定
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,         # 颜色增强
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,          # 水平翻转
        'mosaic': 0.5,          # 马赛克增强
        'degrees': 10.0,        # 旋转增强
        'translate': 0.1,       # 平移增强
        'scale': 0.2,           # 缩放增强
        'shear': 2.0,           # 剪切增强
        'perspective': 0.0005,  # 透视变换
        'copy_paste': 0.0,      # 不使用复制粘贴
        'mixup': 0.0,           # 不使用mixup
        'label_smoothing': 0.1, # 标签平滑
        'dropout': 0.1,         # Dropout防止过拟合
    }
    
    print("\n训练配置:")
    print(f"  数据集: {training_args['data']}")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  批次大小: {training_args['batch']}")
    print(f"  优化器: {training_args['optimizer']}")
    print(f"  学习率: {training_args['lr0']}")
    
    print("\n开始训练...")
    print("注意: 训练可能需要45-90分钟，请耐心等待")
    print("进度条会显示训练状态")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(**training_args)
    
    print("\n训练完成！")
    print("=" * 50)
    
    # 显示训练结果
    if hasattr(results, 'results_dict'):
        print("训练结果:")
        for key, value in results.results_dict.items():
            print(f"  {key}: {value}")
    
    # 找到最佳模型路径
    best_model_path = "runs/detect/apple_high_quality/weights/best.pt"
    if os.path.exists(best_model_path):
        print(f"\n最佳模型已保存到: {best_model_path}")
        
        # 复制到项目根目录方便使用
        import shutil
        shutil.copy2(best_model_path, "apple_high_quality.pt")
        print(f"已复制到: apple_high_quality.pt")
    else:
        print("\n警告: 找不到最佳模型文件")
        print("请检查 runs/detect/apple_high_quality/ 目录")
    
    print("\n下一步:")
    print("1. 测试模型: python test_trained_model.py apple_high_quality.pt")
    print("2. 更新应用中的模型路径")
    print("3. 享受高精度苹果检测！")

if __name__ == "__main__":
    train_apple_model()