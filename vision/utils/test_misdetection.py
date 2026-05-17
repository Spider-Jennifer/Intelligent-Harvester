#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试苹果检测模型的误识别问题
比较新旧模型在包含人和苹果的图片上的表现
"""

import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np

def load_models():
    """加载新旧模型"""
    models = {}
    
    # 新模型（重新标注后训练）
    if os.path.exists("apple_5epoch_test.pt"):
        models["new_5epoch"] = YOLO("apple_5epoch_test.pt")
        print("加载新模型: apple_5epoch_test.pt")
    else:
        print("警告: 新模型不存在，跳过")
    
    # 旧模型（高灵敏度）
    if os.path.exists("apple_sensitive.pt"):
        models["old_sensitive"] = YOLO("apple_sensitive.pt")
        print("加载旧模型: apple_sensitive.pt")
    
    # 快速改进模型（如果存在）
    if os.path.exists("apple_quick_improved.pt"):
        models["quick_improved"] = YOLO("apple_quick_improved.pt")
        print("加载快速改进模型: apple_quick_improved.pt")
    
    if not models:
        print("错误: 没有可用的模型")
        return None
    
    return models

def test_images_with_people(models, test_dir="苹果照片检测素材"):
    """在包含人和苹果的图片上测试模型"""
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在 {test_dir}")
        test_dir = os.path.join(os.path.expanduser("~"), "Desktop", "苹果照片检测素材")
        if not os.path.exists(test_dir):
            print(f"错误: 测试目录不存在 {test_dir}")
            return
    
    # 获取测试图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))
    
    if not test_images:
        print(f"错误: 测试目录中没有图片 {test_dir}")
        return
    
    print(f"\n找到 {len(test_images)} 张测试图片")
    
    results_summary = {}
    
    # 对每张图片测试所有模型
    for img_path in test_images[:10]:  # 最多测试10张
        print(f"\n测试图片: {os.path.basename(img_path)}")
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图片 {img_path}")
            continue
        
        img_display = img.copy()
        
        for model_name, model in models.items():
            # 在不同置信度下测试
            conf_thresholds = [0.1, 0.2, 0.3]
            detections_by_conf = {}
            
            for conf in conf_thresholds:
                # 检测苹果
                results = model(img, conf=conf, verbose=False)
                
                apple_count = 0
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        apple_count = len(result.boxes)
                
                detections_by_conf[conf] = apple_count
            
            print(f"  {model_name}: 置信度0.1={detections_by_conf[0.1]}, 0.2={detections_by_conf[0.2]}, 0.3={detections_by_conf[0.3]}")
            
            # 记录结果
            if model_name not in results_summary:
                results_summary[model_name] = {}
            
            results_summary[model_name][os.path.basename(img_path)] = detections_by_conf
        
        # 使用预训练模型检测人（用于参考）
        try:
            person_model = YOLO('yolov8n.pt')
            person_results = person_model(img, classes=[0], conf=0.3, verbose=False)  # 只检测人（类别0）
            
            if person_results and len(person_results) > 0:
                result = person_results[0]
                if result.boxes is not None:
                    person_count = len(result.boxes)
                    print(f"  检测到人: {person_count} 个")
        except:
            pass
    
    return results_summary

def analyze_misdetection(results_summary):
    """分析误识别情况"""
    print("\n" + "=" * 80)
    print("误识别分析")
    print("=" * 80)
    
    if not results_summary:
        print("没有结果可分析")
        return
    
    for model_name, image_results in results_summary.items():
        print(f"\n模型: {model_name}")
        
        total_high_conf_detections = 0
        total_images = len(image_results)
        
        for img_name, detections_by_conf in image_results.items():
            # 高置信度（0.3）下的检测数
            high_conf_detections = detections_by_conf.get(0.3, 0)
            total_high_conf_detections += high_conf_detections
        
        avg_high_conf_detections = total_high_conf_detections / total_images if total_images > 0 else 0
        
        print(f"  测试图片数量: {total_images}")
        print(f"  平均高置信度(0.3)检测数: {avg_high_conf_detections:.2f}")
        
        # 误识别评估
        if avg_high_conf_detections > 5:
            print(f"  警告: 平均检测数过高，可能存在过度检测")
        elif avg_high_conf_detections < 1:
            print(f"  注意: 平均检测数较低，可能漏检")
        else:
            print(f"  良好: 平均检测数适中")

def main():
    print("=" * 80)
    print("苹果检测模型误识别测试")
    print("测试新旧模型在包含人和苹果的图片上的表现")
    print("=" * 80)
    
    # 加载模型
    models = load_models()
    if not models:
        return
    
    # 测试图片
    test_dir = "C:\\Users\\李晨鑫\\Desktop\\苹果照片检测素材"
    results_summary = test_images_with_people(models, test_dir)
    
    # 分析结果
    if results_summary:
        analyze_misdetection(results_summary)
    
    # 额外测试：在纯人物图片上测试（如果有）
    print("\n" + "=" * 80)
    print("额外测试：检查模型是否将纯人物误识别为苹果")
    print("=" * 80)
    
    # 尝试查找人物图片
    person_test_images = []
    search_dirs = [
        "C:\\Users\\李晨鑫\\Desktop\\苹果照片检测素材",
        "C:\\Users\\李晨鑫\\Desktop\\yolov8\\apple_photos"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                person_test_images.extend(glob.glob(os.path.join(search_dir, ext)))
    
    if person_test_images:
        # 随机选择1-2张图片
        import random
        test_sample = random.sample(person_test_images, min(2, len(person_test_images)))
        
        for img_path in test_sample:
            print(f"\n测试纯人物图片: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            for model_name, model in models.items():
                # 使用较高置信度测试
                results = model(img, conf=0.3, verbose=False)
                apple_count = 0
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        apple_count = len(result.boxes)
                
                if apple_count > 0:
                    print(f"  {model_name}: 误识别为苹果！检测到 {apple_count} 个苹果")
                else:
                    print(f"  {model_name}: 未误识别为苹果")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("建议:")
    print("1. 如果新模型在置信度0.3下检测数适中（1-3个），则效果良好")
    print("2. 如果仍有误识别，考虑添加负样本（不含苹果的图片）训练")
    print("3. 可以训练更多epoch以提高精度")
    print("=" * 80)

if __name__ == "__main__":
    main()