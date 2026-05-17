#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量检测苹果图片并生成GIF动画
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import imageio
import glob
import time

# ============= 可调参数区 =============
# 模型选择：可选 "apple_sensitive.pt", "apple_best.pt", "apple_quick.pt", "yolov8n.pt"
MODEL_CHOICE = "apple_sensitive.pt"
# 置信度阈值 (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.1
# 最大处理的图片数量 (设为0表示处理所有图片)
MAX_IMAGES = 30
# 输出GIF的帧率 (每秒帧数)
GIF_FPS = 10
# 每张图片的动画步数 (检测框渐进绘制步数)
ANIMATION_STEPS = 5
# 检测推理尺寸 (保持与训练一致)
INFERENCE_SIZE = 320
# 输出GIF文件名
OUTPUT_GIF = "apple_detection_animation.gif"
# 图片目录
IMAGE_DIR = r"C:\Users\李晨鑫\Desktop\yolov8\apple_dataset\images\train"
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
        return None, []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    
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
    
    return img_rgb, boxes

def draw_detection_frame(img_rgb, boxes, progress):
    """
    绘制检测框，progress 范围 0.0~1.0 表示动画进度
    返回 PIL Image 对象
    """
    # 创建副本
    frame = img_rgb.copy()
    h, w = frame.shape[:2]
    
    for (x1, y1, x2, y2, conf) in boxes:
        # 计算当前进度下的框位置（从中心向外扩展）
        if progress < 1.0:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            cur_width = int(width * progress)
            cur_height = int(height * progress)
            cur_x1 = center_x - cur_width // 2
            cur_y1 = center_y - cur_height // 2
            cur_x2 = center_x + cur_width // 2
            cur_y2 = center_y + cur_height // 2
        else:
            cur_x1, cur_y1, cur_x2, cur_y2 = x1, y1, x2, y2
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(frame, (cur_x1, cur_y1), (cur_x2, cur_y2), color, thickness)
        
        # 标签：苹果 + 置信度
        label = f"apple {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        label_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
        
        # 标签背景
        label_bg_top = max(cur_y1 - label_h - 5, 0)
        label_bg_bottom = cur_y1
        label_bg_left = cur_x1
        label_bg_right = cur_x1 + label_w
        cv2.rectangle(frame, (label_bg_left, label_bg_top), 
                     (label_bg_right, label_bg_bottom), color, -1)
        # 标签文字
        cv2.putText(frame, label, (cur_x1, cur_y1 - 5), 
                   font, font_scale, (0, 0, 0), label_thickness)
    
    # 转换为 PIL Image
    return Image.fromarray(frame)

def create_animated_gif(image_paths, model, output_gif, fps=10, animation_steps=5):
    """生成GIF动画"""
    frames = []
    total_images = len(image_paths)
    
    for idx, img_path in enumerate(image_paths):
        print(f"处理图片 {idx+1}/{total_images}: {os.path.basename(img_path)}")
        img_rgb, boxes = detect_apples_in_image(model, img_path, CONFIDENCE_THRESHOLD)
        if img_rgb is None:
            continue
        
        # 生成动画序列
        if boxes:
            # 有检测框：渐进绘制
            for step in range(animation_steps + 1):
                progress = step / animation_steps
                frame_img = draw_detection_frame(img_rgb, boxes, progress)
                frames.append(frame_img)
        else:
            # 无检测框：仍添加一张静态图
            frame_img = Image.fromarray(img_rgb)
            frames.append(frame_img)
            # 可选：添加一个"无检测"提示帧
            no_detection = img_rgb.copy()
            cv2.putText(no_detection, "No detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frames.append(Image.fromarray(no_detection))
    
    # 保存GIF
    if frames:
        print(f"生成GIF，共 {len(frames)} 帧，帧率 {fps} FPS")
        # 计算每帧持续时间（毫秒）
        duration = int(1000 / fps)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], 
                      duration=duration, loop=0)
        print(f"GIF已保存: {output_gif}")
    else:
        print("没有生成任何帧")

def main():
    print("=" * 60)
    print("批量检测苹果图片并生成GIF动画")
    print("=" * 60)
    print(f"模型: {MODEL_CHOICE}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"最大图片数: {MAX_IMAGES}")
    print(f"GIF帧率: {GIF_FPS} FPS")
    print(f"动画步数: {ANIMATION_STEPS}")
    print(f"图片目录: {IMAGE_DIR}")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(MODEL_CHOICE):
        print(f"错误: 模型文件 '{MODEL_CHOICE}' 不存在")
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
    
    # 生成GIF
    start_time = time.time()
    create_animated_gif(image_paths, model, OUTPUT_GIF, GIF_FPS, ANIMATION_STEPS)
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    print("完成！")

if __name__ == "__main__":
    main()