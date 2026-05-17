#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版延迟测试 - 只测试最新模型
"""

import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("=" * 60)
print("简化版模型推理延迟测试")
print("=" * 60)
print()

# 1. 选择要测试的模型（只测试最新训练模型）
MODEL_PATH = "runs/cpu_long_training/apple_cpu_long/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
    # 尝试其他模型
    alt_models = ["apple_sensitive.pt", "apple_improved.pt", "apple_best.pt"]
    for alt_model in alt_models:
        if os.path.exists(alt_model):
            MODEL_PATH = alt_model
            print(f"[INFO] 使用备用模型: {MODEL_PATH}")
            break
    else:
        print("[ERROR] 没有找到可用的模型文件")
        exit(1)

size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
print(f"[OK] 测试模型: {MODEL_PATH} ({size_mb:.1f} MB)")
print()

# 2. 准备测试图片
apple_photos_dir = "apple_photos"
if not os.path.exists(apple_photos_dir):
    print(f"[ERROR] 苹果图片目录不存在: {apple_photos_dir}")
    exit(1)

# 获取所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png']
all_images = []
for ext in image_extensions:
    all_images.extend(glob.glob(os.path.join(apple_photos_dir, ext)))

print(f"找到 {len(all_images)} 张苹果图片")
print()

# 限制测试图片数量
MAX_TEST_IMAGES = 30
if len(all_images) > MAX_TEST_IMAGES:
    print(f"选择前 {MAX_TEST_IMAGES} 张图片进行测试")
    test_images = all_images[:MAX_TEST_IMAGES]
else:
    test_images = all_images

print(f"将测试 {len(test_images)} 张图片")
print()

# 3. 测试模型延迟
print(f"正在测试模型: {os.path.basename(MODEL_PATH)}")
print("-" * 40)

try:
    # 加载模型
    start_time = time.time()
    model = YOLO(MODEL_PATH)
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.3f} 秒")
    
    # 预热
    print("预热模型...")
    for i in range(min(3, len(test_images))):
        model(test_images[i], verbose=False)
    
    # 正式测试
    print(f"测试 {len(test_images)} 张图片...")
    inference_times = []
    detection_counts = []
    
    for i, img_path in enumerate(test_images):
        # 记录推理时间
        start_time = time.time()
        results = model(img_path, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # 统计检测数量
        if results and len(results) > 0:
            detections = len(results[0].boxes)
            detection_counts.append(detections)
        else:
            detection_counts.append(0)
        
        # 显示进度
        if (i + 1) % 5 == 0:
            print(f"  已处理 {i+1}/{len(test_images)} 张图片")
    
    # 计算统计信息
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    avg_detections = np.mean(detection_counts)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print()
    print(f"平均推理时间: {avg_inference_time:.4f} ± {std_inference_time:.4f} 秒")
    print(f"最短推理时间: {min_inference_time:.4f} 秒")
    print(f"最长推理时间: {max_inference_time:.4f} 秒")
    print(f"平均FPS: {fps:.2f} 帧/秒")
    print(f"平均检测数量: {avg_detections:.2f} 个苹果/图片")
    
    # 4. 生成延迟折线图
    print("\n生成延迟折线图...")
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 延迟随图片数量的变化
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(inference_times) + 1), inference_times, 'b-', linewidth=2, alpha=0.7, label='单次推理时间')
    
    # 添加移动平均线
    window_size = 5
    if len(inference_times) >= window_size:
        moving_avg = np.convolve(inference_times, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size, len(inference_times) + 1), moving_avg, 'r-', linewidth=2, label=f'{window_size}点移动平均')
    
    plt.axhline(y=avg_inference_time, color='g', linestyle='--', linewidth=2, label=f'平均值: {avg_inference_time:.4f}s')
    plt.title('推理延迟随图片数量的变化', fontsize=12, fontweight='bold')
    plt.xlabel('图片序号', fontsize=10)
    plt.ylabel('推理时间 (秒)', fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2: 延迟分布直方图
    plt.subplot(2, 2, 2)
    plt.hist(inference_times, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_inference_time, color='red', linestyle='--', linewidth=2, label=f'平均值: {avg_inference_time:.4f}s')
    plt.title('推理时间分布直方图', fontsize=12, fontweight='bold')
    plt.xlabel('推理时间 (秒)', fontsize=10)
    plt.ylabel('频次', fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图3: 检测数量与延迟的关系
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(detection_counts, inference_times, c=range(len(inference_times)), 
                         cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='图片序号')
    plt.title('检测数量与推理时间的关系', fontsize=12, fontweight='bold')
    plt.xlabel('检测到的苹果数量', fontsize=10)
    plt.ylabel('推理时间 (秒)', fontsize=10)
    
    # 添加趋势线
    if len(detection_counts) > 1:
        z = np.polyfit(detection_counts, inference_times, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(detection_counts), max(detection_counts), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label='趋势线')
        plt.legend()
    
    plt.grid(alpha=0.3)
    
    # 子图4: 累计时间
    plt.subplot(2, 2, 4)
    cumulative_time = np.cumsum(inference_times)
    plt.plot(range(1, len(cumulative_time) + 1), cumulative_time, 'g-', linewidth=2)
    plt.title('累计推理时间', fontsize=12, fontweight='bold')
    plt.xlabel('已处理图片数量', fontsize=10)
    plt.ylabel('累计时间 (秒)', fontsize=10)
    plt.grid(alpha=0.3)
    
    # 添加总时间标注
    total_time = cumulative_time[-1]
    plt.annotate(f'总时间: {total_time:.2f}s', 
                xy=(len(cumulative_time), total_time),
                xytext=(len(cumulative_time)*0.7, total_time*0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.suptitle(f'模型推理延迟分析 - {os.path.basename(MODEL_PATH)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_dir = "latency_results"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "inference_latency_analysis.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"已保存延迟分析图表: {chart_path}")
    
    # 5. 生成简单报告
    report_path = os.path.join(output_dir, "simple_latency_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("模型推理延迟测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试模型: {MODEL_PATH}\n")
        f.write(f"测试图片: {apple_photos_dir}\n")
        f.write(f"测试数量: {len(test_images)} 张\n\n")
        
        f.write("性能指标:\n")
        f.write(f"  平均推理时间: {avg_inference_time:.4f} 秒\n")
        f.write(f"  时间标准差: {std_inference_time:.4f} 秒\n")
        f.write(f"  最短时间: {min_inference_time:.4f} 秒\n")
        f.write(f"  最长时间: {max_inference_time:.4f} 秒\n")
        f.write(f"  平均FPS: {fps:.2f} 帧/秒\n")
        f.write(f"  平均检测数量: {avg_detections:.2f} 个/图片\n\n")
        
        f.write("延迟分布:\n")
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            percentile_value = np.percentile(inference_times, p)
            f.write(f"  {p}% 的图片推理时间 ≤ {percentile_value:.4f} 秒\n")
        
        f.write(f"\n处理 {len(test_images)} 张图片总耗时: {total_time:.2f} 秒\n")
        f.write(f"平均每张图片: {total_time/len(test_images):.3f} 秒\n")
        
        if fps > 10:
            f.write("\n评价: 性能优秀，适合实时应用\n")
        elif fps > 5:
            f.write("\n评价: 性能良好，适合准实时应用\n")
        elif fps > 2:
            f.write("\n评价: 性能一般，适合非实时应用\n")
        else:
            f.write("\n评价: 性能较低，建议优化\n")
    
    print(f"已保存简单报告: {report_path}")
    
    # 6. 显示关键结果
    print("\n" + "=" * 60)
    print("关键性能指标")
    print("=" * 60)
    print(f"模型: {os.path.basename(MODEL_PATH)}")
    print(f"平均推理延迟: {avg_inference_time:.4f} 秒")
    print(f"推理速度: {fps:.2f} FPS")
    print(f"处理30张图片预计时间: {avg_inference_time*30:.2f} 秒")
    print(f"处理100张图片预计时间: {avg_inference_time*100:.2f} 秒")
    
    # 计算实时性指标
    if fps >= 30:
        print("实时性: 优秀 (≥30 FPS)")
    elif fps >= 15:
        print("实时性: 良好 (15-30 FPS)")
    elif fps >= 5:
        print("实时性: 一般 (5-15 FPS)")
    else:
        print("实时性: 较差 (<5 FPS)")
    
    print()
    print(f"图表已保存: {chart_path}")
    print(f"报告已保存: {report_path}")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("测试完成!")
input("按 Enter 键退出...")