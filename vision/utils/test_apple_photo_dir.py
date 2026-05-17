#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型在更新后的苹果照片上的误识别情况
针对 C:\Users\李晨鑫\Desktop\apple（photo） 目录中的图片
"""

import os
import glob
from ultralytics import YOLO
import cv2
import sys

def find_models():
    """查找所有苹果检测模型"""
    model_patterns = ['apple_*.pt', 'yolov8n.pt']
    models = {}
    for pattern in model_patterns:
        for model_path in glob.glob(pattern):
            model_name = os.path.basename(model_path)
            # 排除可能不是模型的文件
            if model_name.endswith('.pt'):
                try:
                    # 尝试加载以验证
                    model = YOLO(model_path)
                    models[model_name] = model_path
                    print(f"找到模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {model_name} 失败: {e}")
    return models

def test_model_on_photos(model_path, photo_dir, conf_threshold=0.3):
    """在指定目录的图片上测试模型"""
    print(f"\n测试模型: {os.path.basename(model_path)}")
    print(f"图片目录: {photo_dir}")
    print(f"置信度阈值: {conf_threshold}")
    
    if not os.path.exists(photo_dir):
        print(f"错误: 图片目录不存在 {photo_dir}")
        return None
    
    # 获取图片文件
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(photo_dir, ext)))
    
    if not image_paths:
        print(f"错误: 没有找到图片")
        return None
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 加载模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    results = {}
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图片 {img_name}")
            results[img_name] = {'detections': 0, 'error': 'read failed'}
            continue
        
        # 推理
        try:
            inference_results = model(img, conf=conf_threshold, verbose=False)
        except Exception as e:
            print(f"  推理失败 {img_name}: {e}")
            results[img_name] = {'detections': 0, 'error': str(e)}
            continue
        
        detections = 0
        if inference_results and len(inference_results) > 0:
            result = inference_results[0]
            if result.boxes is not None:
                detections = len(result.boxes)
        
        results[img_name] = {'detections': detections, 'error': None}
        print(f"  {img_name}: 检测到 {detections} 个苹果")
    
    return results

def analyze_results(model_name, results):
    """分析检测结果"""
    total_images = len(results)
    total_detections = sum(r['detections'] for r in results.values())
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    # 误识别分析：假设每张图片只有一个苹果
    misdetected = 0
    for img_name, r in results.items():
        if r['detections'] > 1:
            misdetected += 1
    
    print(f"\n分析结果 - {model_name}:")
    print(f"  总图片数: {total_images}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每图检测数: {avg_detections:.2f}")
    print(f"  误识别图片数 (检测>1): {misdetected}")
    
    if misdetected > 0:
        print(f"  误识别率: {misdetected/total_images*100:.1f}%")
    
    return {
        'model': model_name,
        'total_images': total_images,
        'total_detections': total_detections,
        'avg_detections': avg_detections,
        'misdetected': misdetected,
        'misdetection_rate': misdetected/total_images if total_images > 0 else 0
    }

def main():
    photo_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    conf_threshold = 0.3
    
    print("=" * 80)
    print("苹果检测模型误识别测试")
    print(f"测试目录: {photo_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print("=" * 80)
    
    # 查找模型
    models = find_models()
    if not models:
        print("未找到任何模型")
        return
    
    all_analysis = []
    
    # 测试每个模型
    for model_name, model_path in models.items():
        results = test_model_on_photos(model_path, photo_dir, conf_threshold)
        if results is not None:
            analysis = analyze_results(model_name, results)
            all_analysis.append(analysis)
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for analysis in all_analysis:
        print(f"\n模型: {analysis['model']}")
        print(f"  误识别图片数: {analysis['misdetected']}/{analysis['total_images']}")
        print(f"  误识别率: {analysis['misdetection_rate']*100:.1f}%")
        if analysis['misdetected'] == 0:
            print("  ✓ 未发现误识别")
        else:
            print("  ✗ 存在误识别")
    
    # 找出误识别最少的模型
    best_model = min(all_analysis, key=lambda x: x['misdetection_rate']) if all_analysis else None
    if best_model:
        print(f"\n最佳模型: {best_model['model']} (误识别率: {best_model['misdetection_rate']*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()