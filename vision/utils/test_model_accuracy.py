#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型准确率 - 评估当前模型的性能
"""

import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np

def test_model_on_sample_images(model_path, test_images_dir="apple_dataset/images/val"):
    """在测试图片上评估模型性能"""
    print(f"\n测试模型: {os.path.basename(model_path)}")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return None
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取测试图片
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_images_dir, "*.png"))
    
    if not test_images:
        print(f"错误: 测试目录中没有图片 {test_images_dir}")
        return None
    
    print(f"测试图片数量: {len(test_images)}")
    
    # 测试参数
    conf_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    results_summary = {}
    
    for conf in conf_thresholds:
        print(f"\n测试置信度阈值: {conf}")
        
        total_detections = 0
        total_images = 0
        
        for img_path in test_images[:10]:  # 只测试前10张图片
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 进行预测
                results = model(img, conf=conf, verbose=False)
                
                # 统计检测结果
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        detections = len(result.boxes)
                        total_detections += detections
                
                total_images += 1
                
            except Exception as e:
                print(f"处理图片 {os.path.basename(img_path)} 时出错: {e}")
        
        if total_images > 0:
            avg_detections = total_detections / total_images
            results_summary[conf] = avg_detections
            print(f"  平均每张图片检测数: {avg_detections:.2f}")
    
    return results_summary

def compare_models():
    """比较所有可用模型"""
    print("=" * 60)
    print("苹果检测模型性能比较")
    print("=" * 60)
    
    # 查找所有模型文件
    model_files = []
    for file in os.listdir('.'):
        if file.endswith('.pt') and 'apple' in file.lower():
            model_files.append(file)
    
    # 添加预训练模型
    if os.path.exists('yolov8n.pt'):
        model_files.append('yolov8n.pt')
    
    if not model_files:
        print("错误: 没有找到模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型:")
    for model in model_files:
        size = os.path.getsize(model) / (1024 * 1024)
        print(f"  - {model} ({size:.1f} MB)")
    
    print("\n开始测试模型性能...")
    print("测试集: apple_dataset/images/val")
    print("-" * 60)
    
    all_results = {}
    
    for model_file in model_files:
        results = test_model_on_sample_images(model_file)
        if results:
            all_results[model_file] = results
    
    # 显示比较结果
    print("\n" + "=" * 60)
    print("模型性能比较结果")
    print("=" * 60)
    
    if all_results:
        print("\n模型名称\t\t置信度0.1\t0.15\t0.2\t0.25\t0.3")
        print("-" * 80)
        
        for model_name, results in all_results.items():
            short_name = os.path.basename(model_name)
            row = f"{short_name:15}"
            for conf in [0.1, 0.15, 0.2, 0.25, 0.3]:
                if conf in results:
                    row += f"\t{results[conf]:.2f}"
                else:
                    row += "\t-"
            print(row)
        
        # 给出建议
        print("\n" + "=" * 60)
        print("使用建议:")
        print("=" * 60)
        print("1. 检测数适中 (1-3个/图片) 的模型效果最好")
        print("2. 检测数过多可能产生误检")
        print("3. 检测数过少可能漏检")
        print("4. 建议选择在置信度0.2时检测数适中的模型")
        print("\n推荐模型:")
        
        # 找出最佳模型
        best_model = None
        best_score = 0
        
        for model_name, results in all_results.items():
            if 0.2 in results:
                score = results[0.2]
                # 理想检测数在1-3之间
                if 1.0 <= score <= 3.0:
                    distance = abs(score - 2.0)  # 距离理想值2.0的距离
                    if distance < best_score or best_score == 0:
                        best_score = distance
                        best_model = model_name
        
        if best_model:
            print(f"  - {os.path.basename(best_model)}")
            print(f"    在置信度0.2时，平均检测数: {all_results[best_model][0.2]:.2f}")
        else:
            print("  - 没有找到理想模型，建议重新训练")
    
    print("\n测试完成!")

def quick_test_current_model():
    """快速测试当前使用的模型"""
    print("=" * 60)
    print("快速测试当前模型性能")
    print("=" * 60)
    
    # 检查当前使用的模型
    current_models = ["apple_sensitive.pt", "apple_best.pt", "apple_improved.pt"]
    
    for model_file in current_models:
        if os.path.exists(model_file):
            print(f"\n测试模型: {model_file}")
            
            # 简单测试
            try:
                model = YOLO(model_file)
                
                # 找一张测试图片
                test_images = glob.glob("apple_dataset/images/val/*.jpg")[:1]
                if test_images:
                    img_path = test_images[0]
                    print(f"测试图片: {os.path.basename(img_path)}")
                    
                    # 在不同置信度下测试
                    for conf in [0.1, 0.15, 0.2, 0.25, 0.3]:
                        results = model(img_path, conf=conf, verbose=False)
                        if results and len(results) > 0:
                            result = results[0]
                            if result.boxes is not None:
                                detections = len(result.boxes)
                                print(f"  置信度 {conf}: 检测到 {detections} 个苹果")
                            else:
                                print(f"  置信度 {conf}: 未检测到苹果")
                else:
                    print("错误: 没有找到测试图片")
                    
            except Exception as e:
                print(f"测试失败: {e}")

if __name__ == "__main__":
    # 用户选择测试模式
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_models()
        elif sys.argv[1] == "quick":
            quick_test_current_model()
        else:
            test_model_on_sample_images(sys.argv[1])
    else:
        # 默认运行完整比较
        compare_models()