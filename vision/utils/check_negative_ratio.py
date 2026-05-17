#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查负样本比例
"""

import os

def main():
    print("=" * 80)
    print("检查负样本比例")
    print("=" * 80)
    
    # 训练标签目录
    train_labels_dir = "apple_dataset/labels/train"
    if not os.path.exists(train_labels_dir):
        print(f"错误: 找不到训练标签目录 {train_labels_dir}")
        return
    
    # 验证标签目录
    val_labels_dir = "apple_dataset/labels/val"
    
    # 统计训练集
    train_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    train_empty = 0
    for f in train_files:
        p = os.path.join(train_labels_dir, f)
        if os.path.getsize(p) == 0:
            train_empty += 1
    
    train_total = len(train_files)
    train_ratio = train_empty / train_total * 100 if train_total > 0 else 0
    
    print(f"\n训练集标签统计:")
    print(f"  总标签文件: {train_total}")
    print(f"  空标签（负样本）: {train_empty}")
    print(f"  负样本比例: {train_ratio:.1f}%")
    
    # 评估
    if train_ratio < 10:
        print(f"  ⚠️  警告: 负样本比例很低，模型容易误识别")
        print(f"  建议增加到20%以上")
    elif train_ratio < 20:
        print(f"  ⚠️  注意: 负样本比例适中，可以考虑进一步提高")
    else:
        print(f"  ✅ 良好: 负样本比例合适")
    
    # 检查验证集
    if os.path.exists(val_labels_dir):
        val_files = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
        val_empty = 0
        for f in val_files:
            p = os.path.join(val_labels_dir, f)
            if os.path.getsize(p) == 0:
                val_empty += 1
        
        val_total = len(val_files)
        val_ratio = val_empty / val_total * 100 if val_total > 0 else 0
        
        print(f"\n验证集标签统计:")
        print(f"  总标签文件: {val_total}")
        print(f"  空标签（负样本）: {val_empty}")
        print(f"  负样本比例: {val_ratio:.1f}%")
    
    # 建议操作
    print(f"\n建议:")
    if train_ratio < 20:
        print(f"1. 运行 python collect_negative_samples.py 收集更多负样本")
    print(f"2. 运行 python train_apple_ultra_precision.py 开始超精度训练")
    print(f"3. 运行 python test_misdetection.py 测试当前模型误识别情况")
    
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    main()