"""
最简单的苹果检测模型训练脚本
"""

from ultralytics import YOLO
import os

print("开始训练苹果检测模型...")
print("=" * 50)

# 使用预训练模型
model = YOLO('yolov8n.pt')

# 训练参数
training_args = {
    'data': 'YOLO-v8-app/dataset/data.yaml',
    'epochs': 30,
    'imgsz': 320,
    'batch': 4,
    'workers': 0,
    'device': 'cpu',
    'name': 'apple_simple',
    'patience': 5,
    'save': True,
    'pretrained': True,
}

print("训练配置:")
for key, value in training_args.items():
    print(f"  {key}: {value}")

print("\n开始训练...")
print("注意: 训练可能需要15-30分钟")

# 开始训练
results = model.train(**training_args)

print("\n训练完成！")

# 检查生成的模型
best_model_path = "runs/detect/apple_simple/weights/best.pt"
if os.path.exists(best_model_path):
    print(f"最佳模型: {best_model_path}")
    
    # 复制到当前目录
    import shutil
    shutil.copy2(best_model_path, "apple_simple.pt")
    print(f"已复制到: apple_simple.pt")
    
    # 快速测试
    print("\n快速测试...")
    test_model = YOLO("apple_simple.pt")
    
    # 使用一张训练图片测试
    train_dir = "YOLO-v8-app/dataset/images/train"
    if os.path.exists(train_dir):
        train_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if train_images:
            test_img = os.path.join(train_dir, train_images[0])
            test_results = test_model(test_img, conf=0.1, verbose=False)  # 降低置信度阈值以提高检测灵敏度
            
            if test_results and len(test_results) > 0:
                result = test_results[0]
                if result.boxes is not None:
                    print(f"测试结果: 检测到 {len(result.boxes)} 个苹果")
                else:
                    print("测试结果: 未检测到苹果")

print("\n下一步:")
print("1. 测试模型: python test_trained_model.py apple_simple.pt")
print("2. 更新应用中的模型路径")
print("=" * 50)