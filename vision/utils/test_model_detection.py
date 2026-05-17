from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO('apple_best.pt')

# 创建一个测试图像（红色苹果在绿色背景上）
test_image = np.zeros((320, 320, 3), dtype=np.uint8)
test_image[:, :] = (0, 100, 0)  # 绿色背景

# 在中心画一个红色圆形（模拟苹果）
center = (160, 160)
radius = 40
cv2.circle(test_image, center, radius, (0, 0, 255), -1)  # 红色苹果

print("测试图像：红色圆形在绿色背景上（模拟苹果）")

# 使用不同置信度进行检测
for conf in [0.1, 0.05, 0.01]:
    results = model(test_image, conf=conf, verbose=False)
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            print(f"置信度 {conf}: 检测到 {len(result.boxes)} 个物体")
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                print(f"  物体 {i+1}: 类别={cls}, 置信度={conf_score:.3f}")
        else:
            print(f"置信度 {conf}: 未检测到物体")
    else:
        print(f"置信度 {conf}: 检测失败")

# 测试预训练模型
print("\n测试预训练模型 yolov8n.pt:")
pretrained_model = YOLO('yolov8n.pt')
results = pretrained_model(test_image, conf=0.1, verbose=False)

if results and len(results) > 0:
    result = results[0]
    if result.boxes is not None:
        print(f"预训练模型检测到 {len(result.boxes)} 个物体")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            cls_name = pretrained_model.names[cls]
            print(f"  物体 {i+1}: {cls_name}, 置信度={conf_score:.3f}")
    else:
        print("预训练模型未检测到物体")
else:
    print("预训练模型检测失败")