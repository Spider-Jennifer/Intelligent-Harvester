#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量检测苹果照片并生成视频文件（修复版）
解决帧尺寸不一致和编解码器问题
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob
import time

# ============= 可调参数区 =============
# 模型选择：可选 "apple_sensitive.pt", "apple_best.pt", "apple_quick.pt", "yolov8n.pt"
# 优先使用修复后的模型
MODEL_CHOICE = "apple_no_fp.pt"
# 如果上面模型不存在，则回退到以下模型
BACKUP_MODELS = ["apple_sensitive.pt", "apple_best.pt", "apple_improved.pt", "apple_5epoch_test.pt"]
# 置信度阈值 (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.1
# 最大处理的图片数量 (设为0表示处理所有图片)
MAX_IMAGES = 0
# 输出视频的帧率 (每秒帧数)
VIDEO_FPS = 10
# 检测推理尺寸 (保持与训练一致)
INFERENCE_SIZE = 320
# 输出视频文件名
OUTPUT_VIDEO = "apple_detection_video_fixed.avi"
# 图片目录（优先使用修复流程添加的新照片目录）
NEW_PHOTOS_DIR = r"C:\Users\李晨鑫\Desktop\apple（photo）"
APPLE_PHOTOS_DIR = r"C:\Users\李晨鑫\Desktop\yolov8\apple_photos"
MATERIAL_DIR = r"C:\Users\李晨鑫\Desktop\苹果照片检测素材"

# 选择存在的图片目录
if os.path.exists(NEW_PHOTOS_DIR):
    IMAGE_DIR = NEW_PHOTOS_DIR
    print(f"使用新苹果照片目录: {IMAGE_DIR}")
elif os.path.exists(MATERIAL_DIR):
    IMAGE_DIR = MATERIAL_DIR
    print(f"使用苹果照片检测素材目录: {IMAGE_DIR}")
elif os.path.exists(APPLE_PHOTOS_DIR):
    IMAGE_DIR = APPLE_PHOTOS_DIR
    print(f"使用苹果照片目录: {IMAGE_DIR}")
else:
    IMAGE_DIR = "."  # 当前目录
    print(f"警告: 未找到图片目录，使用当前目录: {IMAGE_DIR}")
# 视频编码器 (使用XVID for AVI)
VIDEO_CODEC = 'XVID'
# 统一视频尺寸 (设为None则使用第一张图片的尺寸)
TARGET_SIZE = None  # (宽度, 高度) 例如 (640, 480)
# =====================================

def load_and_sort_images(image_dir):
    """加载并排序图片文件"""
    image_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    # 按文件名排序
    image_paths.sort()
    if MAX_IMAGES > 0:
        image_paths = image_paths[:MAX_IMAGES]
    print(f"找到 {len(image_paths)} 张图片")
    return image_paths

def detect_apples_in_image(model, image_path, conf_threshold):
    """对单张图片进行检测，返回原始图片和检测结果"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None, []
    original_h, original_w = img.shape[:2]
    
    # 转换为RGB用于推理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 推理
    resized = cv2.resize(img_rgb, (INFERENCE_SIZE, INFERENCE_SIZE))
    results = model(resized, conf=conf_threshold, verbose=False)
    
    boxes = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            scale_x = original_w / INFERENCE_SIZE
            scale_y = original_h / INFERENCE_SIZE
            
            for i, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = box
                conf = confs[i]
                # 缩放回原始尺寸
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                boxes.append((x1, y1, x2, y2, conf))
    
    return img, (original_w, original_h), boxes

def draw_detection_frame(img, boxes, target_size=None):
    """
    绘制检测框到图片上，并调整到目标尺寸
    返回带检测框的BGR图片
    """
    if target_size is not None:
        target_w, target_h = target_size
        img_h, img_w = img.shape[:2]
        # 调整图片大小
        img_resized = cv2.resize(img, (target_w, target_h))
        # 调整检测框坐标
        scale_x = target_w / img_w
        scale_y = target_h / img_h
        scaled_boxes = []
        for (x1, y1, x2, y2, conf) in boxes:
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            scaled_boxes.append((x1, y1, x2, y2, conf))
        boxes = scaled_boxes
        frame = img_resized
    else:
        frame = img.copy()
    
    for (x1, y1, x2, y2, conf) in boxes:
        # 绘制边界框
        color = (0, 255, 0)  # 绿色 (BGR)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 标签：苹果 + 置信度
        label = f"apple {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        label_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
        
        # 标签背景
        label_bg_top = max(y1 - label_h - 5, 0)
        label_bg_bottom = y1
        label_bg_left = x1
        label_bg_right = x1 + label_w
        cv2.rectangle(frame, (label_bg_left, label_bg_top), 
                     (label_bg_right, label_bg_bottom), color, -1)
        # 标签文字
        cv2.putText(frame, label, (x1, y1 - 5), 
                   font, font_scale, (0, 0, 0), label_thickness)
    
    return frame

def create_detection_video(image_paths, model, output_video, fps=10, target_size=None):
    """生成检测视频"""
    if not image_paths:
        print("没有图片可以处理")
        return
    
    # 确定视频尺寸
    if target_size is None:
        # 使用第一张图片的尺寸
        first_img, first_size, _ = detect_apples_in_image(model, image_paths[0], CONFIDENCE_THRESHOLD)
        if first_img is None:
            print(f"无法读取第一张图片: {image_paths[0]}")
            return
        target_w, target_h = first_size
    else:
        target_w, target_h = target_size
    
    print(f"视频尺寸: {target_w} x {target_h}, 帧率: {fps} FPS")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (target_w, target_h))
    
    if not video_writer.isOpened():
        print(f"无法创建视频文件: {output_video}")
        return
    
    total_images = len(image_paths)
    processed_count = 0
    
    for idx, img_path in enumerate(image_paths):
        print(f"处理图片 {idx+1}/{total_images}: {os.path.basename(img_path)}")
        img, size, boxes = detect_apples_in_image(model, img_path, CONFIDENCE_THRESHOLD)
        if img is None:
            continue
        
        # 绘制检测框并调整尺寸
        frame = draw_detection_frame(img, boxes, (target_w, target_h))
        
        # 写入视频帧
        video_writer.write(frame)
        processed_count += 1
    
    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存: {output_video}")
    print(f"成功处理 {processed_count}/{total_images} 张图片")

def main():
    print("=" * 60)
    print("批量检测苹果照片并生成视频（修复版）")
    print("=" * 60)
    print(f"模型: {MODEL_CHOICE}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"最大图片数: {'全部' if MAX_IMAGES == 0 else MAX_IMAGES}")
    print(f"视频帧率: {VIDEO_FPS} FPS")
    print(f"视频编码器: {VIDEO_CODEC}")
    print(f"目标尺寸: {TARGET_SIZE or '自动'}")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    print("=" * 60)
    
    # 动态选择模型
    model_candidates = [MODEL_CHOICE] + BACKUP_MODELS
    selected_model = None
    
    for model_candidate in model_candidates:
        if os.path.exists(model_candidate):
            selected_model = model_candidate
            print(f"✅ 找到模型: {model_candidate}")
            break
    
    if selected_model is None:
        print("❌ 错误: 没有找到任何可用的模型文件")
        print(f"   尝试的模型列表: {model_candidates}")
        return
    
    # 更新模型选择
    MODEL_CHOICE = selected_model
    
    # 检查图片目录
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 图片目录 '{IMAGE_DIR}' 不存在")
        return
    
    # 加载模型
    print("加载YOLOv8模型...")
    model = YOLO(MODEL_CHOICE)
    print("模型加载完成")
    
    # 获取图片列表
    image_paths = load_and_sort_images(IMAGE_DIR)
    if not image_paths:
        print("未找到图片")
        return
    
    # 生成视频
    start_time = time.time()
    create_detection_video(image_paths, model, OUTPUT_VIDEO, VIDEO_FPS, TARGET_SIZE)
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    print("完成！")

if __name__ == "__main__":
    main()