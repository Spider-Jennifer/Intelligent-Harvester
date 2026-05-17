#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试低置信度下的苹果检测
"""

import os
import cv2
from ultralytics import YOLO
import glob

def main():
    print("快速测试苹果检测 (置信度=0.25)")
    
    # 检查模型
    model_path = "apple_ultra_precision.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载模型
    model = YOLO(model_path)
    
    # 图片目录
    image_dir = r"C:\Users\李晨鑫\Desktop\苹果照片检测素材"
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在 {image_dir}")
        return
    
    # 获取第一张图片
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_files:
        print("错误: 没有找到jpg图片")
        return
    
    test_image = image_files[0]
    print(f"测试图片: {test_image}")
    
    # 读取图片
    img = cv2.imread(test_image)
    if img is None:
        print("错误: 无法读取图片")
        return
    
    # 在不同置信度下测试
    confidences = [0.15, 0.2, 0.25, 0.3, 0.35]
    
    for conf in confidences:
        results = model(img, conf=conf, verbose=False)
        apple_count = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                apple_count = len(result.boxes)
                print(f"置信度 {conf}: 检测到 {apple_count} 个苹果")
                for i, box in enumerate(result.boxes):
                    conf_val = box.conf.cpu().numpy()[0]
                    print(f"  第{i+1}个: 置信度={conf_val:.3f}")
            else:
                print(f"置信度 {conf}: 检测到 0 个苹果")
        else:
            print(f"置信度 {conf}: 检测到 0 个苹果")
    
    # 可视化检测结果 (置信度0.25)
    conf = 0.25
    results = model(img, conf=conf, verbose=False)
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                conf_val = boxes.conf[j].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"apple {conf_val:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存测试结果
    output_path = "quick_test_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n测试结果已保存: {output_path}")
    
    # 检查文件大小
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"图片大小: {size_kb:.1f} KB")
    
if __name__ == "__main__":
    main()