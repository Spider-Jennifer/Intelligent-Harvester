from ultralytics import YOLO

# 检查模型
model = YOLO('apple_best.pt')
print("模型类别:", model.names)
print("模型类别数量:", len(model.names))

# 测试模型在训练图片上的表现
import os
import cv2

# 检查训练图片
train_dir = "YOLO-v8-app/dataset/images/train"
if os.path.exists(train_dir):
    train_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"训练图片数量: {len(train_images)}")
    
    # 测试第一张图片
    if train_images:
        test_img_path = os.path.join(train_dir, train_images[0])
        print(f"测试图片: {test_img_path}")
        
        # 进行检测
        results = model(test_img_path, conf=0.1)
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                print(f"检测到 {len(result.boxes)} 个苹果")
                # 显示检测结果
                result.show()
            else:
                print("未检测到苹果")
        else:
            print("检测失败")
else:
    print(f"训练目录不存在: {train_dir}")