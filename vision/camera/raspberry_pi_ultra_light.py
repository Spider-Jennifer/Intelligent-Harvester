#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派4B苹果检测 - 极致精简版
绝对最小内存占用，最高性能
"""

import cv2
import time
import numpy as np
import os

# 全局配置 - 硬编码优化参数
CONFIG = {
    'width': 320,      # 绝对最小分辨率
    'height': 240,
    'inference_width': 160,  # 推理尺寸减半
    'inference_height': 120,
    'confidence': 0.1,  # 降低置信度阈值以提高检测灵敏度
    'skip_frames': 2,  # 默认跳帧
    'target_fps': 15
}

def load_model():
    """加载模型 - 极致简化"""
    try:
        from ultralytics import YOLO
        
        # 只查找最小模型
        model_files = ["yolov8n.pt", "best.pt"]
        for model_file in model_files:
            if os.path.exists(model_file):
                return YOLO(model_file)
    except:
        pass
    return None

def detect_objects(model, frame, confidence):
    """检测对象 - 极致优化"""
    if model is None:
        return []
    
    try:
        # 最小推理尺寸
        resized = cv2.resize(frame, (CONFIG['inference_width'], CONFIG['inference_height']))
        
        # 单次推理
        results = model(resized, conf=confidence, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if result.boxes is None:
            return []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        # 缩放参数
        scale_x = frame.shape[1] / CONFIG['inference_width']
        scale_y = frame.shape[0] / CONFIG['inference_height']
        
        detections = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            detections.append([x1, y1, x2, y2, conf])
        
        return detections
    except:
        return []

def main():
    """主函数 - 极致精简"""
    # 加载模型
    model = load_model()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    # 设置最小参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['height'])
    
    # 性能跟踪
    frame_counter = 0
    skip_counter = 0
    last_fps_time = time.time()
    fps = 0
    
    print("树莓派苹果检测 - 极致精简版")
    print("控制: Q=退出, C=置信度, S=跳帧")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        skip_counter += 1
        
        # 跳帧逻辑
        if skip_counter % CONFIG['skip_frames'] == 0:
            # 执行检测
            start_time = time.time()
            detections = detect_objects(model, frame, CONFIG['confidence'])
            inference_time = time.time() - start_time
            
            # 绘制结果
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"{conf:.1f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 显示FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_counter / (current_time - last_fps_time)
                frame_counter = 0
                last_fps_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {inference_time*1000:.0f}ms", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示
        cv2.imshow('Apple Detection', frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break
        elif key == ord('c'):
            # 循环切换置信度
            conf_values = [0.3, 0.5, 0.7]
            current_idx = conf_values.index(CONFIG['confidence']) if CONFIG['confidence'] in conf_values else 0
            CONFIG['confidence'] = conf_values[(current_idx + 1) % len(conf_values)]
        elif key == ord('s'):
            # 循环切换跳帧
            skip_values = [1, 2, 3, 4]
            current_idx = skip_values.index(CONFIG['skip_frames']) if CONFIG['skip_frames'] in skip_values else 0
            CONFIG['skip_frames'] = skip_values[(current_idx + 1) % len(skip_values)]
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()