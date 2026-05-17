"""
测试新训练的苹果检测模型
"""

from ultralytics import YOLO
import cv2
import os

print("测试 apple_quick.pt 模型...")
print("=" * 50)

# 加载模型
model = YOLO('apple_quick.pt')
print(f"模型类别: {model.names}")
print(f"模型类别数量: {len(model.names)}")

# 测试训练集中的图片
train_dir = 'apple_dataset/images/train'
if os.path.exists(train_dir):
    train_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n训练集图片数量: {len(train_images)}")
    
    # 测试前5张图片
    test_count = min(5, len(train_images))
    print(f"测试前 {test_count} 张图片:")
    
    total_detections = 0
    for i in range(test_count):
        img_path = os.path.join(train_dir, train_images[i])
        print(f"\n{i+1}. {train_images[i]}")
        
        # 使用不同置信度测试
        for conf in [0.1, 0.2, 0.3]:
            results = model(img_path, conf=conf, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    detections = len(result.boxes)
                    total_detections += detections
                    print(f"   置信度 {conf}: 检测到 {detections} 个苹果")
                else:
                    print(f"   置信度 {conf}: 未检测到苹果")
            else:
                print(f"   置信度 {conf}: 检测失败")
    
    print(f"\n平均每张图片检测到: {total_detections/test_count:.1f} 个苹果")

# 测试验证集中的图片
val_dir = 'apple_dataset/images/val'
if os.path.exists(val_dir):
    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n验证集图片数量: {len(val_images)}")

print("\n" + "=" * 50)
print("测试完成！")
print("现在可以运行应用测试实时检测效果")
print("=" * 50)