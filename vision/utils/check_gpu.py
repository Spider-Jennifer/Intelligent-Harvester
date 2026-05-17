#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查GPU和摄像头延迟问题
"""

import torch
import cv2
import time

print("=== 系统环境检查 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("警告: CUDA不可用，将使用CPU运行，这可能导致延迟较高")

print("\n=== 摄像头测试 ===")
# 测试摄像头延迟
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误: 无法打开摄像头")
else:
    print("摄像头已打开")
    
    # 测试读取帧的延迟
    start_time = time.time()
    success, frame = cap.read()
    capture_time = time.time() - start_time
    
    if success:
        print(f"摄像头分辨率: {frame.shape[1]}x{frame.shape[0]}")
        print(f"单帧捕获时间: {capture_time:.3f}秒")
        
        # 测试连续读取的延迟
        print("\n测试连续读取5帧...")
        total_time = 0
        for i in range(5):
            start_time = time.time()
            success, frame = cap.read()
            if success:
                frame_time = time.time() - start_time
                total_time += frame_time
                print(f"  第{i+1}帧: {frame_time:.3f}秒")
            else:
                print(f"  第{i+1}帧: 读取失败")
        
        avg_time = total_time / 5 if total_time > 0 else 0
        print(f"平均每帧读取时间: {avg_time:.3f}秒")
        print(f"理论FPS: {1/avg_time:.1f}" if avg_time > 0 else "无法计算FPS")
    else:
        print("错误: 无法读取摄像头帧")
    
    cap.release()

print("\n=== 建议 ===")
if not torch.cuda.is_available():
    print("1. 安装CUDA版本的PyTorch以启用GPU加速")
    print("2. 降低摄像头分辨率以减少处理时间")
    print("3. 使用更小的YOLOv8模型（如yolov8n.pt）")
else:
    print("1. 确保YOLOv8模型使用GPU进行推理")
    print("2. 降低摄像头分辨率（如640x480）")
    print("3. 使用Streamlit的缓存优化")