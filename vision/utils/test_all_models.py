#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有可用苹果检测模型
检查模型是否能正确检测苹果
"""

import os
import cv2
import glob
from ultralytics import YOLO

def find_models():
    """查找所有苹果检测模型"""
    pattern = "apple_*.pt"
    models = []
    
    for file in glob.glob(pattern):
        if os.path.isfile(file):
            models.append(file)
    
    # 排序：优先使用较新的模型
    models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return models

def find_test_images():
    """查找测试图片"""
    # 多个可能的目录
    possible_dirs = [
        r"C:\Users\李晨鑫\Desktop\苹果照片检测素材",
        r"C:\Users\李晨鑫\Desktop\yolov8\apple_photos",
        r"C:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset\images\val",
        r"C:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset\images\train"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            images = glob.glob(os.path.join(dir_path, "*.jpg")) + \
                     glob.glob(os.path.join(dir_path, "*.png")) + \
                     glob.glob(os.path.join(dir_path, "*.jpeg"))
            if images:
                print(f"找到图片目录: {dir_path} ({len(images)} 张图片)")
                return dir_path, images[:10]  # 返回前10张图片
    return None, []

def test_model(model_path, test_images, conf_threshold=0.25):
    """测试单个模型"""
    print(f"\n测试模型: {os.path.basename(model_path)}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  加载模型失败: {e}")
        return {}
    
    results = {}
    
    for img_idx, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 检测苹果
        detections = model(img, conf=conf_threshold, verbose=False)
        
        apple_count = 0
        confidences = []
        
        if detections and len(detections) > 0:
            result = detections[0]
            if result.boxes is not None:
                apple_count = len(result.boxes)
                for box in result.boxes:
                    conf = box.conf.cpu().numpy()[0]
                    confidences.append(conf)
        
        results[img_name] = {
            'count': apple_count,
            'confidences': confidences
        }
        
        print(f"  图片 {img_idx+1}: {img_name} - 检测到 {apple_count} 个苹果")
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"    平均置信度: {avg_conf:.3f}")
    
    return results

def main():
    print("=" * 80)
    print("苹果检测模型全面测试")
    print("=" * 80)
    
    # 查找所有模型
    models = find_models()
    if not models:
        print("错误: 没有找到苹果检测模型 (*.pt)")
        return
    
    print(f"找到 {len(models)} 个模型:")
    for model in models:
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"  - {os.path.basename(model)} ({size_mb:.1f} MB)")
    
    # 查找测试图片
    test_dir, test_images = find_test_images()
    if not test_images:
        print("错误: 没有找到测试图片")
        return
    
    print(f"\n使用测试图片: {test_dir}")
    print(f"测试图片数量: {len(test_images)}")
    
    # 在所有模型上测试
    all_results = {}
    conf_thresholds = [0.2, 0.25, 0.3]
    
    for conf in conf_thresholds:
        print(f"\n{'='*50}")
        print(f"置信度阈值: {conf}")
        print(f"{'='*50}")
        
        for model_path in models:
            results = test_model(model_path, test_images, conf)
            model_name = os.path.basename(model_path)
            all_results[(model_name, conf)] = results
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("检测结果汇总")
    print(f"{'='*80}")
    
    for model_path in models:
        model_name = os.path.basename(model_path)
        print(f"\n模型: {model_name}")
        
        for conf in conf_thresholds:
            key = (model_name, conf)
            if key in all_results:
                results = all_results[key]
                total_detections = sum(r['count'] for r in results.values())
                avg_detections = total_detections / len(results) if results else 0
                
                print(f"  置信度 {conf}: 总检测数={total_detections}, 平均每张图={avg_detections:.2f}")
                
                # 显示每张图片的结果
                for img_name, result in results.items():
                    if result['count'] > 0:
                        print(f"    {img_name}: {result['count']} 个苹果")
    
    # 建议最佳模型
    print(f"\n{'='*80}")
    print("建议")
    print(f"{'='*80}")
    
    best_model = None
    best_conf = None
    best_score = -1
    
    for model_path in models:
        for conf in [0.25, 0.3]:
            key = (os.path.basename(model_path), conf)
            if key in all_results:
                results = all_results[key]
                total_detections = sum(r['count'] for r in results.values())
                avg_detections = total_detections / len(results) if results else 0
                
                # 理想情况：平均每张图有1-3个苹果
                if 0.5 <= avg_detections <= 3.0:
                    score = avg_detections
                    if score > best_score:
                        best_score = score
                        best_model = os.path.basename(model_path)
                        best_conf = conf
    
    if best_model:
        print(f"推荐模型: {best_model} (置信度 {best_conf})")
        print(f"理由: 平均检测数适中 ({best_score:.2f} 个苹果/图片)")
    else:
        print("警告: 所有模型检测数都偏低")
        print("可能原因:")
        print("1. 测试图片中没有苹果")
        print("2. 模型训练不足")
        print("3. 置信度阈值过高")
    
    print(f"\n下一步操作:")
    print("1. 检查测试图片中是否包含苹果")
    print("2. 尝试降低置信度阈值 (如0.15)")
    print("3. 使用之前的敏感模型 (apple_sensitive.pt)")
    print("4. 重新训练模型，调整分类损失权重")

if __name__ == "__main__":
    main()