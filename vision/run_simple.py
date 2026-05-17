#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 极简运行脚本 - 非交互模式
"""

import os
import sys
import subprocess

def install_dependencies():
    """安装必要的依赖"""
    print("正在安装依赖...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                          'torch', 'torchvision', 'ultralytics', 'opencv-python'])
    print("依赖安装完成！")

def main():
    print("=" * 60)
    print("YOLOv8 苹果检测 - 非交互模式")
    print("=" * 60)
    
    # 检查并安装依赖
    try:
        import torch
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"缺少依赖: {e}")
        install_dependencies()
        import torch
        from ultralytics import YOLO
        import cv2
    
    # 修复 PyTorch 兼容性
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load
    
    # 查找模型文件
    model_paths = [
        'yolov8-apple-detection-master/apple_detection_model_gpu/weights/best.pt',
        'YOLO-v8-app/YOLOv8-app-master/weights/best.pt',
        'yolov8-apple-detection-master/apple_detection_model2/weights/best.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("错误：未找到模型文件")
        return
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    print("模型加载成功！")
    
    # 查找测试图像
    test_dirs = [
        'YOLO-v8-app/quick_train/images',
        'yolov8-apple-detection-master/Dataset/apple_dataset/images/train',
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg', '.png'))]
            if images:
                test_image = os.path.join(test_dir, images[0])
                break
    
    if test_image and os.path.exists(test_image):
        print(f"测试图像: {test_image}")
        
        # 执行预测
        results = model(test_image, conf=0.1)  # 降低置信度阈值以提高检测灵敏度
        
        print("\n检测结果:")
        for r in results:
            boxes = r.boxes
            print(f"检测到 {len(boxes)} 个物体")
            for box in boxes:
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                print(f"  - {class_name}: 置信度 = {confidence:.2f}")
        
        # 保存结果
        output_dir = 'output_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, r in enumerate(results):
            im_array = r.plot()
            output_path = os.path.join(output_dir, 'result.jpg')
            cv2.imwrite(output_path, im_array)
            print(f"\n结果已保存到: {output_path}")
    else:
        print("警告：未找到测试图像")
    
    print("\n运行完成！")

if __name__ == "__main__":
    main()
