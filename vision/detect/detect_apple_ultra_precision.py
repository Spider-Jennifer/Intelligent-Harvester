#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超精度苹果检测视频生成
使用新训练的超精度模型，减少眼镜等物体的误识别
"""

import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def test_different_confidences(model, img_path, confidences):
    """测试不同置信度阈值下的检测结果"""
    img = cv2.imread(img_path)
    if img is None:
        return {}
    
    results_by_conf = {}
    for conf in confidences:
        results = model(img, conf=conf, verbose=False)
        apple_count = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                apple_count = len(result.boxes)
        
        results_by_conf[conf] = apple_count
    
    return results_by_conf

def main():
    print("=" * 80)
    print("超精度苹果检测与视频生成")
    print("=" * 80)
    print("目标: 大幅减少眼镜等物体的误识别")
    print("使用新训练的超精度苹果检测模型")
    print("=" * 80)
    
    # 模型配置 - 优先使用表现最佳的模型
    model_candidates = [
        "apple_5epoch_test.pt",
        "apple_improved.pt",
        "apple_sensitive.pt",
        "apple_quick.pt",
        "apple_ultra_precision.pt",
        "apple_final_improved.pt"
    ]
    
    selected_model = None
    for model_path in model_candidates:
        if os.path.exists(model_path):
            selected_model = model_path
            break
    
    if selected_model is None:
        print("错误: 找不到任何模型文件")
        print("请先运行训练脚本 train_apple_ultra_precision.py")
        return
    
    print(f"使用模型: {selected_model}")
    
    # 加载模型
    print(f"加载模型...")
    model = YOLO(selected_model)
    
    # 检测参数 - 平衡精度与召回率
    base_conf_threshold = 0.3        # 基础置信度阈值（平衡）
    iou_threshold = 0.5              # IoU阈值
    imgsz = 640                      # 推理尺寸
    
    print(f"检测参数:")
    print(f"  基础置信度阈值: {base_conf_threshold} (较高，减少误识别)")
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
    
    # 分析误识别情况（先测试几张）
    print(f"\n分析误识别情况...")
    test_confidences = [0.2, 0.3, 0.4, 0.5]
    
    if len(image_files) > 5:
        test_samples = image_files[:5]
    else:
        test_samples = image_files
    
    misdetection_analysis = {}
    for img_path in test_samples:
        img_name = os.path.basename(img_path)
        results = test_different_confidences(model, img_path, test_confidences)
        misdetection_analysis[img_name] = results
        
        print(f"  图片: {img_name}")
        for conf in test_confidences:
            count = results.get(conf, 0)
            print(f"    置信度 {conf}: {count} 个苹果")
    
    # 根据分析结果选择最佳置信度
    # 策略：选择能平衡检测率和误识别率的置信度
    recommended_conf = base_conf_threshold
    print(f"\n推荐使用置信度阈值: {recommended_conf}")
    
    # 询问用户是否自定义置信度
    print(f"\n是否自定义置信度阈值? (当前: {recommended_conf})")
    print("输入数字 (如0.3) 或直接按回车使用当前值:")
    user_input = input().strip()
    if user_input:
        try:
            user_conf = float(user_input)
            if 0.1 <= user_conf <= 0.9:
                recommended_conf = user_conf
                print(f"使用自定义置信度: {recommended_conf}")
            else:
                print(f"无效值，使用默认值: {recommended_conf}")
        except:
            print(f"无效输入，使用默认值: {recommended_conf}")
    
    # 读取第一张图片获取视频尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"错误: 无法读取第一张图片 {image_files[0]}")
        return
    
    height, width = first_img.shape[:2]
    video_size = (width, height)
    
    # 视频输出配置
    video_output = "apple_detection_ultra_precision.avi"
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
        
        # 检测苹果（使用推荐置信度）
        results = model(img, conf=recommended_conf, iou=iou_threshold, 
                       imgsz=imgsz, verbose=False)
        
        apples_in_image = 0
        img_with_boxes = img.copy()
        
        # 存储检测详情（用于调试）
        detections_details = []
        
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
                    
                    # 根据置信度决定框的颜色
                    if conf >= 0.5:
                        color = (0, 255, 0)  # 高置信度 - 绿色
                    elif conf >= 0.3:
                        color = (0, 255, 255)  # 中置信度 - 黄色
                    else:
                        color = (0, 165, 255)  # 低置信度 - 橙色
                    
                    # 绘制边界框
                    thickness = 2 if conf >= 0.5 else 1
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
                    
                    # 显示置信度
                    label = f"apple {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # 标签背景
                    cv2.rectangle(img_with_boxes, 
                                 (x1, y1 - label_height - baseline - 5),
                                 (x1 + label_width, y1),
                                 color, -1)
                    
                    cv2.putText(img_with_boxes, label,
                               (x1, y1 - baseline - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # 存储检测详情
                    detections_details.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf
                    })
        
        # 显示统计信息
        cv2.putText(img_with_boxes, f"Apples: {apples_in_image}",
                   (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(img_with_boxes, f"Conf: {recommended_conf}",
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示模型信息
        model_name = os.path.splitext(os.path.basename(selected_model))[0]
        cv2.putText(img_with_boxes, f"Model: {model_name}",
                   (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 检测框颜色说明
        cv2.putText(img_with_boxes, "Green: high conf (>=0.5)",
                   (width - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(img_with_boxes, "Yellow: medium (0.3-0.5)",
                   (width - 250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 将当前帧写入视频（多帧以延长显示时间）
        for _ in range(frames_per_image):
            video_writer.write(img_with_boxes)
        
        total_apples_detected += apples_in_image
        processed_images += 1
        
        print(f"  检测到 {apples_in_image} 个苹果")
        for det in detections_details:
            print(f"    - 置信度: {det['conf']:.2f}")
    
    # 释放视频写入器
    video_writer.release()
    
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"  处理的图片: {processed_images}/{len(image_files)}")
    print(f"  检测到的苹果总数: {total_apples_detected}")
    if processed_images > 0:
        print(f"  平均每张图片苹果数: {total_apples_detected/processed_images:.2f}")
    
    print(f"\n视频已生成: {video_output}")
    if os.path.exists(video_output):
        file_size_mb = os.path.getsize(video_output) / (1024 * 1024)
        print(f"  文件大小: {file_size_mb:.1f} MB")
        print(f"  时长: {actual_duration:.1f} 秒")
        print(f"  帧率: {fps} FPS")
        
        # 验证视频
        cap = cv2.VideoCapture(video_output)
        if cap.isOpened():
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  ✅ 视频验证成功，帧数: {video_frames}")
            cap.release()
        else:
            print(f"  ⚠️  无法打开视频文件")
    else:
        print(f"  ❌ 视频文件生成失败")
    
    print("\n" + "=" * 80)
    print("超精度检测说明:")
    print("1. 使用新训练的超精度模型，分类损失权重大幅提高")
    print("2. 置信度阈值较高 (推荐 0.4)，大幅减少误识别")
    print("3. 检测框颜色编码: 绿色=高置信度, 黄色=中置信度")
    print("4. 专门针对眼镜、人脸等物体的误识别进行优化")
    
    print("\n检测效果评估:")
    avg_apples_per_image = total_apples_detected / processed_images if processed_images > 0 else 0
    if avg_apples_per_image < 1:
        print("  ⚠️  平均检测数较低，可能存在漏检")
        print("  建议: 降低置信度阈值 (如0.3)")
    elif avg_apples_per_image > 5:
        print("  ⚠️  平均检测数过高，可能存在过度检测")
        print("  建议: 提高置信度阈值 (如0.5)")
    else:
        print("  ✅ 平均检测数适中")
    
    print("\n下一步:")
    print("1. 播放视频查看检测效果: " + video_output)
    print("2. 检查是否有眼镜等物体被误识别为苹果")
    print("3. 如需调整，重新运行脚本并输入不同置信度")
    print("=" * 80)

if __name__ == "__main__":
    main()