"""
快速改进模型 - 使用现有模型进行微调
"""

from ultralytics import YOLO
import os
import shutil

print("快速改进苹果检测模型")
print("=" * 50)

# 使用现有模型进行微调
print("1. 使用 apple_quick.pt 进行微调...")

# 检查模型文件
if not os.path.exists("apple_quick.pt"):
    print("错误: 找不到 apple_quick.pt 模型")
    exit(1)

# 训练参数 - 针对提高检测灵敏度优化
training_args = {
    'data': 'apple_dataset/data.yaml',
    'epochs': 15,           # 较少的epoch，快速微调
    'imgsz': 416,           # 中等分辨率
    'batch': 4,
    'workers': 0,
    'device': 'cpu',
    'name': 'apple_sensitive',
    'patience': 5,
    'save': True,
    'pretrained': 'apple_quick.pt',  # 使用现有模型
    'optimizer': 'Adam',
    'lr0': 0.0005,          # 很低的学习率，微调
    'lrf': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'warmup_epochs': 2,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.02,
    'hsv_s': 0.8,
    'hsv_v': 0.5,
    'fliplr': 0.5,
    'degrees': 10.0,
}

print("\n训练配置:")
print(f"  基础模型: apple_quick.pt")
print(f"  Epochs: {training_args['epochs']}")
print(f"  图像尺寸: {training_args['imgsz']}")
print(f"  学习率: {training_args['lr0']} (微调)")

print("\n开始微调...")
print("注意: 微调可能需要10-20分钟")

# 加载现有模型
model = YOLO('apple_quick.pt')

# 开始微调
results = model.train(**training_args)

print("\n微调完成！")

# 复制模型到根目录
best_model_path = "runs/detect/apple_sensitive/weights/best.pt"
if os.path.exists(best_model_path):
    shutil.copy2(best_model_path, "apple_sensitive.pt")
    print(f"高灵敏度模型已保存: apple_sensitive.pt")
    
    # 快速测试
    print("\n快速测试新模型...")
    test_model = YOLO("apple_sensitive.pt")
    
    # 测试一张图片
    test_img = os.path.join('apple_dataset/images/train', os.listdir('apple_dataset/images/train')[0])
    
    print(f"测试图片: {os.path.basename(test_img)}")
    print("不同置信度下的检测结果:")
    
    for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  # 从0.1开始测试，提高检测灵敏度
        test_results = test_model(test_img, conf=conf, verbose=False)
        
        if test_results and len(test_results) > 0:
            result = test_results[0]
            if result.boxes is not None:
                print(f"  置信度 {conf}: 检测到 {len(result.boxes)} 个苹果")
            else:
                print(f"  置信度 {conf}: 未检测到苹果")

print("\n" + "=" * 50)
print("改进完成！")
print("下一步:")
print("1. 更新应用中的模型路径为 apple_sensitive.pt")
print("2. 测试更高置信度下的检测效果")
print("=" * 50)