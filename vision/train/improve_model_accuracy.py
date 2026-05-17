"""
提高苹果检测模型精度
重新训练模型，降低检测阈值，提高检测灵敏度
"""

from ultralytics import YOLO
import os
import shutil

print("提高苹果检测模型精度")
print("=" * 60)

# 步骤1: 重新训练模型，使用更低的检测阈值
print("1. 重新训练模型，提高检测灵敏度...")

# 检查数据集配置
data_yaml = "apple_dataset/data.yaml"
if not os.path.exists(data_yaml):
    print(f"错误: 找不到配置文件 {data_yaml}")
    exit(1)

print(f"使用配置文件: {data_yaml}")

# 训练参数 - 针对提高检测灵敏度优化
training_args = {
    'data': data_yaml,
    'epochs': 30,           # 更多epoch
    'imgsz': 416,           # 中等分辨率
    'batch': 4,
    'workers': 0,
    'device': 'cpu',
    'name': 'apple_improved',
    'patience': 8,
    'save': True,
    'pretrained': True,
    'optimizer': 'Adam',
    'lr0': 0.0008,          # 较低学习率，更稳定
    'lrf': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0003,
    'warmup_epochs': 3,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.02,          # 更强的颜色增强
    'hsv_s': 0.8,
    'hsv_v': 0.5,
    'fliplr': 0.5,
    'degrees': 15.0,        # 更强的旋转增强
    'translate': 0.15,
    'scale': 0.25,
    'shear': 3.0,
    'perspective': 0.001,
    'copy_paste': 0.0,
    'mixup': 0.0,
}

print("\n训练配置:")
print(f"  Epochs: {training_args['epochs']}")
print(f"  图像尺寸: {training_args['imgsz']}")
print(f"  学习率: {training_args['lr0']}")
print(f"  增强: 更强")

print("\n开始训练...")
print("注意: 训练可能需要20-40分钟")

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(**training_args)

print("\n训练完成！")

# 步骤2: 复制模型到根目录
best_model_path = "runs/detect/apple_improved/weights/best.pt"
if os.path.exists(best_model_path):
    shutil.copy2(best_model_path, "apple_improved.pt")
    print(f"改进模型已保存: apple_improved.pt")
    
    # 测试新模型
    print("\n测试改进模型...")
    test_model = YOLO("apple_improved.pt")
    
    # 使用不同置信度测试
    test_img = os.path.join('apple_dataset/images/train', os.listdir('apple_dataset/images/train')[0])
    
    print(f"测试图片: {os.path.basename(test_img)}")
    print("不同置信度下的检测结果:")
    
    for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        test_results = test_model(test_img, conf=conf, verbose=False)
        
        if test_results and len(test_results) > 0:
            result = test_results[0]
            if result.boxes is not None:
                print(f"  置信度 {conf}: 检测到 {len(result.boxes)} 个苹果")
            else:
                print(f"  置信度 {conf}: 未检测到苹果")
        else:
            print(f"  置信度 {conf}: 检测失败")

print("\n" + "=" * 60)
print("改进完成！")
print("下一步:")
print("1. 更新应用中的模型路径为 apple_improved.pt")
print("2. 测试更高置信度下的检测效果")
print("=" * 60)