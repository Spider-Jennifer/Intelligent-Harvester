#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型推理延迟
测试 apple_photos 目录中的苹果图片识别延迟
"""

import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("模型推理延迟测试")
print("=" * 60)
print()

# 1. 选择要测试的模型
MODEL_PATHS = [
    "apple_sensitive.pt",           # 敏感模型
    "apple_improved.pt",            # 改进模型
    "apple_best.pt",                # 最佳模型
    "runs/cpu_long_training/apple_cpu_long/weights/best.pt",  # 最新训练模型
    "yolov8n.pt"                    # 基础模型
]

# 检查哪些模型可用
available_models = []
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        available_models.append(model_path)
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"[OK] 模型可用: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"[SKIP] 模型不存在: {model_path}")

print()

# 2. 准备测试图片
apple_photos_dir = "apple_photos"
if not os.path.exists(apple_photos_dir):
    print(f"[ERROR] 苹果图片目录不存在: {apple_photos_dir}")
    exit(1)

# 获取所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
all_images = []
for ext in image_extensions:
    all_images.extend(glob.glob(os.path.join(apple_photos_dir, ext)))

print(f"找到 {len(all_images)} 张苹果图片用于测试")
print()

# 限制测试图片数量，避免测试时间过长
MAX_TEST_IMAGES = 50
if len(all_images) > MAX_TEST_IMAGES:
    print(f"图片数量过多，随机选择 {MAX_TEST_IMAGES} 张进行测试")
    import random
    test_images = random.sample(all_images, MAX_TEST_IMAGES)
else:
    test_images = all_images

print(f"将使用 {len(test_images)} 张图片进行延迟测试")
print()

# 3. 测试每个模型的延迟
results = {}

for model_path in available_models:
    print(f"正在测试模型: {os.path.basename(model_path)}")
    print("-" * 40)
    
    try:
        # 加载模型
        start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_time
        print(f"  模型加载时间: {load_time:.3f} 秒")
        
        # 预热模型（运行几次推理）
        print("  正在预热模型...")
        warmup_times = []
        for i in range(3):
            if i < len(test_images):
                img_path = test_images[i]
                start = time.time()
                results_warmup = model(img_path, verbose=False)
                warmup_times.append(time.time() - start)
        
        # 正式测试
        print(f"  正在测试 {len(test_images)} 张图片...")
        inference_times = []
        detection_counts = []
        
        for i, img_path in enumerate(test_images):
            # 记录推理开始时间
            start_time = time.time()
            
            # 执行推理
            results_infer = model(img_path, verbose=False)
            
            # 记录推理结束时间
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 统计检测到的苹果数量
            if results_infer and len(results_infer) > 0:
                detections = len(results_infer[0].boxes)
                detection_counts.append(detections)
            else:
                detection_counts.append(0)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"    已处理 {i+1}/{len(test_images)} 张图片")
        
        # 计算统计信息
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        avg_detections = np.mean(detection_counts)
        
        # 计算FPS（每秒帧数）
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        print(f"  平均推理时间: {avg_inference_time:.4f} ± {std_inference_time:.4f} 秒")
        print(f"  最短推理时间: {min_inference_time:.4f} 秒")
        print(f"  最长推理时间: {max_inference_time:.4f} 秒")
        print(f"  平均FPS: {fps:.2f} 帧/秒")
        print(f"  平均检测数量: {avg_detections:.2f} 个苹果/图片")
        print()
        
        # 保存结果
        model_name = os.path.basename(model_path).replace('.pt', '')
        results[model_name] = {
            'model_path': model_path,
            'load_time': load_time,
            'inference_times': inference_times,
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'fps': fps,
            'avg_detections': avg_detections,
            'detection_counts': detection_counts,
            'test_images': test_images
        }
        
    except Exception as e:
        print(f"  测试失败: {e}")
        print()

print("=" * 60)
print("延迟测试完成")
print("=" * 60)

# 4. 生成延迟对比图表
if results:
    print("\n生成延迟对比图表...")
    
    # 创建图表目录
    output_dir = "latency_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    model_names = list(results.keys())
    avg_times = [results[m]['avg_inference_time'] for m in model_names]
    fps_values = [results[m]['fps'] for m in model_names]
    
    # 图表1：平均推理时间对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, avg_times, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
    plt.title('平均推理时间对比', fontsize=14, fontweight='bold')
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('时间 (秒)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{time_val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 图表2：FPS对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, fps_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
    plt.title('推理速度 (FPS) 对比', fontsize=14, fontweight='bold')
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('FPS (帧/秒)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, fps_val in zip(bars, fps_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{fps_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "model_latency_comparison.png")
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    print(f"  已保存对比图表: {chart1_path}")
    
    # 图表3：每个模型的延迟分布（箱线图）
    plt.figure(figsize=(10, 6))
    inference_data = [results[m]['inference_times'] for m in model_names]
    box = plt.boxplot(inference_data, labels=model_names, patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('推理时间分布 (箱线图)', fontsize=14, fontweight='bold')
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('推理时间 (秒)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加平均值点
    for i, (model, avg_time) in enumerate(zip(model_names, avg_times)):
        plt.plot(i+1, avg_time, 'ro', markersize=8, label='平均值' if i == 0 else "")
    
    plt.legend()
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "latency_distribution_boxplot.png")
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"  已保存分布图表: {chart2_path}")
    
    # 图表4：延迟随图片数量的变化（折线图）
    plt.figure(figsize=(12, 8))
    
    for i, model_name in enumerate(model_names):
        inference_times = results[model_name]['inference_times']
        # 使用移动平均平滑曲线
        window_size = 5
        if len(inference_times) >= window_size:
            smoothed_times = np.convolve(inference_times, np.ones(window_size)/window_size, mode='valid')
            x_values = range(window_size-1, len(inference_times))
            plt.plot(x_values, smoothed_times, label=model_name, linewidth=2, alpha=0.8)
        else:
            plt.plot(inference_times, label=model_name, linewidth=2, alpha=0.8)
    
    plt.title('推理延迟随图片数量的变化', fontsize=14, fontweight='bold')
    plt.xlabel('图片序号', fontsize=12)
    plt.ylabel('推理时间 (秒)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 添加平均线
    for model_name in model_names:
        avg_time = results[model_name]['avg_inference_time']
        plt.axhline(y=avg_time, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "latency_trend_line_chart.png")
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
    print(f"  已保存趋势图表: {chart3_path}")
    
    # 图表5：检测数量与延迟的关系
    plt.figure(figsize=(12, 6))
    
    for i, model_name in enumerate(model_names):
        inference_times = results[model_name]['inference_times']
        detection_counts = results[model_name]['detection_counts']
        
        plt.subplot(2, 3, i+1 if i < 6 else 6)
        plt.scatter(detection_counts, inference_times, alpha=0.6, s=30)
        plt.title(f'{model_name}', fontsize=10)
        plt.xlabel('检测数量', fontsize=9)
        plt.ylabel('时间 (秒)', fontsize=9)
        plt.grid(alpha=0.3)
        
        # 添加趋势线
        if len(detection_counts) > 1:
            z = np.polyfit(detection_counts, inference_times, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(detection_counts), max(detection_counts), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=1)
    
    plt.suptitle('检测数量与推理时间的关系', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "detection_vs_latency_scatter.png")
    plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
    print(f"  已保存散点图表: {chart4_path}")
    
    # 5. 生成详细报告
    print("\n生成详细报告...")
    report_path = os.path.join(output_dir, "latency_test_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型推理延迟测试报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试图片目录: {apple_photos_dir}\n")
        f.write(f"测试图片数量: {len(test_images)}\n")
        f.write(f"测试模型数量: {len(available_models)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("各模型性能对比\n")
        f.write("=" * 60 + "\n\n")
        
        # 创建性能对比表格
        f.write(f"{'模型名称':<20} {'平均时间(秒)':<15} {'FPS':<10} {'最短时间(秒)':<15} {'最长时间(秒)':<15} {'平均检测数':<12}\n")
        f.write("-" * 90 + "\n")
        
        for model_name in model_names:
            data = results[model_name]
            f.write(f"{model_name:<20} {data['avg_inference_time']:<15.4f} {data['fps']:<10.2f} "
                   f"{data['min_inference_time']:<15.4f} {data['max_inference_time']:<15.4f} "
                   f"{data['avg_detections']:<12.2f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("性能排名\n")
        f.write("=" * 60 + "\n\n")
        
        # 按FPS排序
        sorted_by_fps = sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True)
        f.write("按FPS排序 (从高到低):\n")
        for i, (model_name, data) in enumerate(sorted_by_fps, 1):
            f.write(f"  {i}. {model_name}: {data['fps']:.2f} FPS (平均时间: {data['avg_inference_time']:.4f}秒)\n")
        
        f.write("\n按推理时间排序 (从快到慢):\n")
        sorted_by_time = sorted(results.items(), key=lambda x: x[1]['avg_inference_time'])
        for i, (model_name, data) in enumerate(sorted_by_time, 1):
            f.write(f"  {i}. {model_name}: {data['avg_inference_time']:.4f}秒 (FPS: {data['fps']:.2f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("测试图片列表\n")
        f.write("=" * 60 + "\n\n")
        
        for i, img_path in enumerate(test_images):
            f.write(f"{i+1:3d}. {os.path.basename(img_path)}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("生成的图表文件\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"1. 模型对比图: {chart1_path}\n")
        f.write(f"2. 延迟分布图: {chart2_path}\n")
        f.write(f"3. 延迟趋势图: {chart3_path}\n")
        f.write(f"4. 检测数量关系图: {chart4_path}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("结论与建议\n")
        f.write("=" * 60 + "\n\n")
        
        # 找出最佳模型
        best_by_fps = sorted_by_fps[0]
        best_by_time = sorted_by_time[0]
        
        f.write(f"1. 最快的模型: {best_by_fps[0]} ({best_by_fps[1]['fps']:.2f} FPS)\n")
        f.write(f"2. 最低延迟的模型: {best_by_time[0]} ({best_by_time[1]['avg_inference_time']:.4f}秒)\n")
        
        # 计算性能提升百分比
        if len(sorted_by_fps) > 1:
            fastest_fps = best_by_fps[1]['fps']
            slowest_fps = sorted_by_fps[-1][1]['fps']
            improvement = ((fastest_fps - slowest_fps) / slowest_fps) * 100
            f.write(f"3. 最快与最慢模型FPS差异: {improvement:.1f}%\n")
        
        f.write("\n4. 建议:\n")
        f.write("   - 对于实时应用: 选择FPS最高的模型\n")
        f.write("   - 对于准确性要求高的应用: 选择检测数量稳定的模型\n")
        f.write("   - 对于资源受限的环境: 选择模型文件较小的模型\n")
    
    print(f"  已保存详细报告: {report_path}")
    
    # 显示主要结果
    print("\n" + "=" * 60)
    print("主要测试结果")
    print("=" * 60)
    
    # 找出最佳模型
    best_fps_model = max(results.items(), key=lambda x: x[1]['fps'])
    best_time_model = min(results.items(), key=lambda x: x[1]['avg_inference_time'])
    
    print(f"最快的模型: {best_fps_model[0]}")
    print(f"  - FPS: {best_fps_model[1]['fps']:.2f} 帧/秒")
    print(f"  - 平均推理时间: {best_fps_model[1]['avg_inference_time']:.4f} 秒")
    print()
    
    print(f"最低延迟的模型: {best_time_model[0]}")
    print(f"  - 平均推理时间: {best_time_model[1]['avg_inference_time']:.4f} 秒")
    print(f"  - FPS: {best_time_model[1]['fps']:.2f} 帧/秒")
    print()
    
    print(f"所有图表和报告已保存到: {output_dir}/")
    print("1. model_latency_comparison.png - 模型对比图")
    print("2. latency_distribution_boxplot.png - 延迟分布图")
    print("3. latency_trend_line_chart.png - 延迟趋势图")
    print("4. detection_vs_latency_scatter.png - 检测数量关系图")
    print("5. latency_test_report.txt - 详细报告")
    
else:
    print("没有可用的测试结果")

print()
print("测试完成!")
input("按 Enter 键退出...")