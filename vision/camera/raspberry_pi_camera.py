#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派4B专用苹果摄像头检测 - 精简优化版
最低内存占用，最高性能
"""

import cv2
import time
import numpy as np
import os

# 尝试导入YOLO，如果失败则使用模拟模式
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class RaspberryPiCamera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.running = False
        
        # 树莓派优化参数
        self.width = 320    # 最低分辨率，减少计算量
        self.height = 240
        self.target_fps = 15
        
        # 性能统计（最小化）
        self.frame_count = 0
        self.total_inference_time = 0
        self.fps = 0
        
        # 检测参数
        self.confidence = 0.1  # 降低置信度阈值以提高检测灵敏度
        self.skip_frames = 2  # 每2帧处理1次
        self.frame_counter = 0
        
        # 加载模型
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载YOLO模型 - 精简版"""
        if not YOLO_AVAILABLE:
            return
        
        # 查找模型文件（优先使用yolov8n，最小）
        model_candidates = [
            "yolov8n.pt",  # 最小模型
            "best.pt",
            "apple_best.pt"
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path:
            try:
                self.model = YOLO(model_path)
            except:
                self.model = None
    
    def detect(self, frame):
        """检测对象 - 最简化版本"""
        if self.model is None:
            return []
        
        try:
            # 使用最小推理尺寸
            inference_size = (160, 120)  # 非常小的推理尺寸
            resized = cv2.resize(frame, inference_size)
            
            # 执行推理
            results = self.model(resized, conf=self.confidence, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # 缩放框到原始尺寸
                    scale_x = frame.shape[1] / inference_size[0]
                    scale_y = frame.shape[0] / inference_size[1]
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
            
            return detections
            
        except:
            return []
    
    def draw_detections(self, frame, detections):
        """绘制检测结果 - 最简化"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # 只绘制边界框（最简）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # 简化的标签
            label = f"{conf:.1f}"
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def draw_stats(self, frame, inference_time):
        """绘制性能统计 - 最简化"""
        # 只显示FPS和推理时间
        fps_text = f"FPS: {self.fps:.1f}"
        time_text = f"Time: {inference_time*1000:.0f}ms"
        
        cv2.putText(frame, fps_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, time_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """运行摄像头检测 - 主循环"""
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 设置摄像头参数（最小化）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.running = True
        last_time = time.time()
        fps_update_interval = 1.0  # 1秒更新一次FPS
        
        print("摄像头检测已启动")
        print("按 Q 或 ESC 退出")
        
        while self.running:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.frame_counter += 1
            
            # 跳帧处理
            if self.frame_counter % self.skip_frames != 0:
                # 显示原始帧（不检测）
                display_frame = frame.copy()
            else:
                # 执行检测
                start_time = time.time()
                detections = self.detect(frame)
                inference_time = time.time() - start_time
                
                self.total_inference_time += inference_time
                
                # 绘制检测结果
                display_frame = self.draw_detections(frame.copy(), detections)
                
                # 绘制统计信息
                display_frame = self.draw_stats(display_frame, inference_time)
            
            # 计算FPS
            current_time = time.time()
            if current_time - last_time >= fps_update_interval:
                self.fps = self.frame_count / (current_time - last_time)
                self.frame_count = 0
                last_time = current_time
            
            # 显示帧
            cv2.imshow('Apple Detection - Raspberry Pi', display_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:  # ESC或Q
                break
            elif key == ord('c'):  # 切换置信度
                self.confidence = 0.3 if self.confidence >= 0.7 else self.confidence + 0.2
            elif key == ord('s'):  # 切换跳帧
                self.skip_frames = 1 if self.skip_frames >= 4 else self.skip_frames + 1
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印最终统计
        if self.total_inference_time > 0:
            avg_inference_time = self.total_inference_time / (self.frame_count / self.skip_frames)
            print(f"平均推理时间: {avg_inference_time*1000:.1f}ms")
            print(f"最终FPS: {self.fps:.1f}")

def main():
    """主函数"""
    detector = RaspberryPiCamera(camera_id=0)
    detector.run()

if __name__ == "__main__":
    main()