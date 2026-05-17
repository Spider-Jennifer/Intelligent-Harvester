#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用改进的苹果检测模型检测苹果照片并生成视频
基于 apple_5epoch_test.pt 模型，优化参数减少误识别
"""

import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def main():
    print("=" * 80)
    print("改进的苹果检测与视频生成")
    print("使用 apple_5epoch_test.pt 模型")
    print("优化参数以减少误识别")
    print("=" * 80)
    
    # 模型配置
    model_path = "apple_5epoch_test.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行训练脚本 train_apple_5epoch.py")
        return
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 检测参数（针对减少误识别优化）
    conf_threshold = 0.3        # 较高置信度阈值，减少误识别
    iou_threshold = 0.5         # IoU阈值
    imgsz = 320                 # 推理尺寸
    
    print(f"检测参数:")
    print(f"  置信度阈值: {conf_threshold} (较高，减少误识别)")
    print(f"  IoU阈值: {iou_threshold}")
    print(f"  推理尺寸: {imgsz}")
    
    # 目标目录
    target_dir = "C:\\Users\\李晨鑫\\Desktop\\苹果照片检测素材"
    if not os.path.exists(target_dir):
        print(f"错误: 目标目录不存在 {target_dir}")
        return
    
    # 获取所有图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(target_dir, ext)))
    
    if not image_files:
        print(f"错误: 目标目录中没有图片 {target_dir}")
        return
    
    # 按文件名排序
    image_files.sort()
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取视频尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"错误: 无法读取第一张图片 {image_files[0]}")
        return
    
    height, width = first_img.shape[:2]
    video_size = (width, height)
    
    # 视频输出配置
    video_output = "apple_detection_improved.avi"
    fps = 10  # 帧率
    video_duration = 8.0  # 目标视频时长（秒）
    
    # 计算每张图片的帧数
    total_frames_needed = int(video_duration * fps)
    frames_per_image = max(1, total_frames_needed // len(image_files))
    actual_total_frames = frames_per_image * len(image_files)
    actual_duration = actual_total_frames / fps
    
    print(f"\n视频配置:")
    print(f"  输出文件: {video_output}")
    print(f"  尺寸: {video_size}")
    print(f"  帧率: {fps}")
    print(f"  目标时长: {video_duration}秒")
    print(f"  实际时长: {actual_duration:.1f}秒")
    print(f"  每张图片帧数: {frames_per_image}")
    print(f"  总帧数: {actual_total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output, fourcc, fps, video_size)
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {video_output}")
        return
    
    print("\n开始处理图片...")
    
    total_apples_detected = 0
    processed_images = 0
    
    for i, img_path in enumerate(image_files):
        print(f"处理 [{i+1}/{len(image_files)}]: {os.path.basename(img_path)}")
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图片，跳过")
            continue
        
        # 调整尺寸以保持一致性
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, video_size)
        
        # 检测苹果
        results = model(img, conf=conf_threshold, iou=iou_threshold, 
                       imgsz=imgsz, verbose=False)
        
        apples_in_image = 0
        img_with_boxes = img.copy()
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                apples_in_image = len(boxes)
                
                # 绘制检测框
                for j in range(len(boxes)):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    conf = boxes.conf[j].cpu().numpy()
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 绘制绿色边界框
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 显示置信度
                    label = f"apple {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    cv2.rectangle(img_with_boxes, 
                                 (x1, y1 - label_height - baseline - 5),
                                 (x1 + label_width, y1),
                                 (0, 255, 0), -1)
                    
                    cv2.putText(img_with_boxes, label,
                               (x1, y1 - baseline - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 显示统计信息
        cv2.putText(img_with_boxes, f"Apples: {apples_in_image}",
                   (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(img_with_boxes, f"Conf: {conf_threshold}",
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 将当前帧写入视频（多帧以延长显示时间）
        for _ in range(frames_per_image):
            video_writer.write(img_with_boxes)
        
        total_apples_detected += apples_in_image
        processed_images += 1
        
        print(f"  检测到 {apples_in_image} 个苹果")
    
    # 释放视频写入器
    video_writer.release()
    
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"  处理的图片: {processed_images}/{len(image_files)}")
    print(f"  检测到的苹果总数: {total_apples_detected}")
    if processed_images > 0:
        print(f"  平均每张图片苹果数: {total_apples_detected/processed_images:.2f}")
    
    print(f"\n视频已生成: {video_output}")
    print(f"  文件大小: {os.path.getsize(video_output)/(1024*1024):.1f} MB")
    print(f"  时长: {actual_duration:.1f} 秒")
    print(f"  帧率: {fps} FPS")
    
    # 验证视频
    print("\n验证视频文件...")
    cap = cv2.VideoCapture(video_output)
    if cap.isOpened():
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  视频帧数: {video_frames}")
        print(f"  视频FPS: {video_fps:.1f}")
        print(f"  视频尺寸: {video_width}x{video_height}")
        
        cap.release()
        
        if video_frames == actual_total_frames:
            print("  ✅ 视频文件验证通过")
        else:
            print(f"  ⚠️  视频帧数不匹配: 预期 {actual_total_frames}, 实际 {video_frames}")
    else:
        print("  ❌ 无法打开视频文件")
    
    print("\n" + "=" * 80)
    print("改进说明:")
    print("1. 使用重新标注数据集训练的模型 (apple_5epoch_test.pt)")
    print("2. 置信度阈值设置为 0.3，减少误识别")
    print("3. 视频中每张图片显示时间延长，便于观察")
    print("4. 检测框显示置信度，便于评估")
    print("\n下一步:")
    print("1. 播放视频查看检测效果: apple_detection_improved.avi")
    print("2. 如需进一步减少误识别，可提高置信度阈值")
    print("3. 如需更高精度，运行 train_apple_final_improved.py")
    print("=" * 80)

if __name__ == "__main__":
    main()