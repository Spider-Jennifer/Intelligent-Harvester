#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
显示当前识别率
"""

import pandas as pd
import os

print("=" * 70)
print("CURRENT DETECTION ACCURACY REPORT")
print("=" * 70)
print()

results_file = "runs/cpu_long_training/apple_cpu_long/results.csv"

if not os.path.exists(results_file):
    print("Error: Results file not found")
    print("Training may still be in progress")
    exit(1)

try:
    # 读取数据
    df = pd.read_csv(results_file)
    
    print(f"Training Progress: {len(df)} / 150 epochs ({len(df)/150*100:.1f}%)")
    print()
    
    # 获取最新数据
    last_row = df.iloc[-1]
    
    print("CURRENT RECOGNITION ACCURACY")
    print("-" * 40)
    print(f"Epoch: {int(last_row['epoch'])}")
    print(f"mAP50 (IoU=0.5):      {last_row['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95 (IoU=0.5:0.95): {last_row['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision:           {last_row['metrics/precision(B)']:.4f}")
    print(f"Recall:              {last_row['metrics/recall(B)']:.4f}")
    print()
    
    # 计算改进
    first_row = df.iloc[0]
    improvement_map50 = last_row['metrics/mAP50(B)'] - first_row['metrics/mAP50(B)']
    improvement_map = last_row['metrics/mAP50-95(B)'] - first_row['metrics/mAP50-95(B)']
    
    print("IMPROVEMENT SINCE START")
    print("-" * 40)
    print(f"mAP50 improvement:    +{improvement_map50:.4f}")
    print(f"mAP50-95 improvement: +{improvement_map:.4f}")
    print()
    
    # 最佳表现
    best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
    best_map50_value = df['metrics/mAP50(B)'].max()
    
    best_map_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
    best_map_value = df['metrics/mAP50-95(B)'].max()
    
    print("BEST PERFORMANCE SO FAR")
    print("-" * 40)
    print(f"Best mAP50: {best_map50_value:.4f} (epoch {best_map50_epoch})")
    print(f"Best mAP50-95: {best_map_value:.4f} (epoch {best_map_epoch})")
    print()
    
    # 训练状态评估
    print("TRAINING STATUS ASSESSMENT")
    print("-" * 40)
    
    current_map50 = last_row['metrics/mAP50(B)']
    current_map = last_row['metrics/mAP50-95(B)']
    
    if current_map50 >= 0.99:
        print("[OK] mAP50: EXCELLENT (> 0.99)")
    elif current_map50 >= 0.95:
        print("[OK] mAP50: VERY GOOD (0.95-0.99)")
    elif current_map50 >= 0.90:
        print("[WARN] mAP50: GOOD (0.90-0.95)")
    elif current_map50 >= 0.80:
        print("[WARN] mAP50: FAIR (0.80-0.90)")
    else:
        print("[ERROR] mAP50: NEEDS IMPROVEMENT (< 0.80)")
    
    if current_map >= 0.90:
        print("[OK] mAP50-95: EXCELLENT (> 0.90)")
    elif current_map >= 0.80:
        print("[OK] mAP50-95: VERY GOOD (0.80-0.90)")
    elif current_map >= 0.70:
        print("[WARN] mAP50-95: GOOD (0.70-0.80)")
    elif current_map >= 0.60:
        print("[WARN] mAP50-95: FAIR (0.60-0.70)")
    else:
        print("[ERROR] mAP50-95: NEEDS IMPROVEMENT (< 0.60)")
    
    print()
    print("GENERATED CHARTS")
    print("-" * 40)
    print("1. detection_accuracy_chart.png - 识别率折线图")
    print("2. training_metrics_summary.png - 综合训练指标")
    print("3. training_report.html - HTML训练报告")
    print()
    
    print("RECOMMENDATIONS")
    print("-" * 40)
    if current_map50 >= 0.99 and current_map >= 0.90:
        print("[OK] Model is performing excellently!")
        print("   - Continue training to reach 150 epochs")
        print("   - Use apple_cpu_trained.pt for inference")
        print("   - Set confidence threshold: 0.15-0.25")
    elif current_map50 >= 0.95:
        print("[OK] Model is performing well")
        print("   - Continue training to improve mAP50-95")
        print("   - Consider adding more training data")
    else:
        print("[WARN] Model needs more training")
        print("   - Continue to 150 epochs")
        print("   - Check dataset quality")
        print("   - Consider data augmentation")
    
except Exception as e:
    print(f"Error reading results: {e}")

print()
print("=" * 70)
print("To view charts, open:")
print("1. detection_accuracy_chart.png")
print("2. training_metrics_summary.png")
print("3. training_report.html (in browser)")
print("=" * 70)