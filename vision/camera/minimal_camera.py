#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小延迟摄像头苹果检测 - 纯OpenCV版本
不依赖Streamlit，获得最低延迟
"""

import cv2
import time
import numpy as np
import os
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics未安装，将使用模拟检测模式")
    print("安装命令: pip install ultralytics")

class MinimalCameraDetector:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.running = False
        
        # 摄像头参数
        self.width = 640   # 降低分辨率减少延迟
        self.height = 480
        self.target_fps = 20
        
        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_frame_time = 0
        self.fps_history = []
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 检测参数
        self.confidence_threshold = 0.1  # 降低置信度阈值以提高检测灵敏度
        self.skip_frames = 2  # 每2帧处理1次
        self.frame_counter = 0
        
    def load_model(self):
        """加载YOLO模型"""
        if not YOLO_AVAILABLE:
            print("使用模拟检测模式")
            return
        
        # 查找模型文件
        model_candidates = [
            "best.pt",
            "yolov8n.pt", 
            "yolov8s.pt",
            "apple_best.pt"
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                print(f"找到模型: {model_path}")
                break
        
        if not model_path:
            # 在子目录中查找
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".pt"):
                        model_path = os.path.join(root, file)
                        print(f"找到模型: {model_path}")
                        break
                if model_path:
                    break
        
        if model_path:
            try:
                print(f"加载模型: {model_path}")
                self.model = YOLO(model_path)
                print("✅ 模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None
        else:
            print("⚠️ 未找到模型文件，使用模拟模式")
    
    def detect_objects(self, frame):
        """检测对象"""
        if self.model is None:
            # 模拟检测模式
            return []
        
        try:
            # 进一步缩小推理尺寸
            inference_size = (320, 240)
            resized_frame = cv2.resize(frame, inference_size)
            
            # 执行推理
            results = self.model(resized_frame, 
                                conf=self.confidence_threshold,
                                verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # 缩放框到原始尺寸
                    scale_x = frame.shape[1] / inference_size[0]
                    scale_y = frame.shape[0] / inference_size[1]
                    
                    for box, conf, cls in zip(boxes, confs, classes):
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': int(cls)
                        })
            
            return detections
            
        except Exception as e:
            print(f"检测错误: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Apple: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 在框中心绘制点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return frame
    
    def draw_stats(self, frame, inference_time, fps):
        """绘制性能统计"""
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
        stats = [
            f"帧数: {self.frame_count}",
            f"推理时间: {inference_time*1000:.1f}ms",
            f"FPS: {fps:.1f}",
            f"置信度: {self.confidence_threshold}",
            f"跳帧: 1/{self.skip_frames}",
            f"分辨率: {self.width}x{self.height}"
        ]
        
        for i, stat in enumerate(stats):
            y_pos = 40 + i * 15
            cv2.putText(frame, stat, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """运行摄像头检测"""
        print("=" * 50)
        print("最小延迟摄像头苹果检测")
        print("=" * 50)
        print("控制说明:")
        print("  [Q] 或 [ESC] - 退出程序")
        print("  [C] - 切换置信度阈值 (0.3/0.5/0.7)")
        print("  [S] - 切换跳帧设置 (1/2/3/4)")
        print("  [R] - 重置性能统计")
        print("  [+] - 增加置信度")
        print("  [-] - 降低置信度")
        print("=" * 50)
        
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.running = True
        last_time = time.time()
        
        print("摄像头启动成功，开始检测...")
        print("按 Q 或 ESC 键退出")
        
        try:
            while self.running:
                # 记录帧开始时间
                frame_start = time.time()
                
                # 读取帧
                success, frame = cap.read()
                if not success:
                    print("摄像头读取失败")
                    break
                
                self.frame_counter += 1
                
                # 跳帧处理
                if self.frame_counter % self.skip_frames == 0:
                    # 执行检测
                    inference_start = time.time()
                    detections = self.detect_objects(frame)
                    inference_time = time.time() - inference_start
                    
                    # 更新统计
                    self.total_inference_time += inference_time
                    
                    # 绘制检测结果
                    if detections:
                        frame = self.draw_detections(frame, detections)
                        print(f"检测到 {len(detections)} 个苹果", end='\r')
                
                # 计算FPS
                frame_time = time.time() - frame_start
                self.total_frame_time += frame_time
                self.frame_count += 1
                
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                avg_inference = self.total_inference_time / max(self.frame_count, 1)
                
                # 绘制性能统计
                frame = self.draw_stats(frame, avg_inference, avg_fps)
                
                # 显示帧
                cv2.imshow('Minimal Apple Detection - Low Latency', frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q 或 ESC
                    break
                elif key == ord('c'):  # 切换置信度
                    if self.confidence_threshold == 0.3:
                        self.confidence_threshold = 0.5
                    elif self.confidence_threshold == 0.5:
                        self.confidence_threshold = 0.7
                    else:
                        self.confidence_threshold = 0.3
                    print(f"置信度阈值: {self.confidence_threshold}")
                elif key == ord('s'):  # 切换跳帧
                    self.skip_frames = (self.skip_frames % 4) + 1
                    print(f"跳帧设置: 每 {self.skip_frames} 帧处理1次")
                elif key == ord('r'):  # 重置统计
                    self.frame_count = 0
                    self.total_inference_time = 0
                    self.total_frame_time = 0
                    self.fps_history = []
                    print("性能统计已重置")
                elif key == ord('+'):  # 增加置信度
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                    print(f"置信度阈值: {self.confidence_threshold:.2f}")
                elif key == ord('-'):  # 降低置信度
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"置信度阈值: {self.confidence_threshold:.2f}")
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup(cap)
    
    def cleanup(self, cap):
        """清理资源"""
        self.running = False
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        if self.frame_count > 0:
            print("\n" + "=" * 50)
            print("最终性能统计:")
            print("=" * 50)
            print(f"总处理帧数: {self.frame_count}")
            print(f"平均推理时间: {self.total_inference_time/self.frame_count*1000:.1f}ms")
            print(f"平均帧处理时间: {self.total_frame_time/self.frame_count*1000:.1f}ms")
            print(f"平均FPS: {self.frame_count/self.total_frame_time:.1f}")
            
            if self.fps_history:
                print(f"最近FPS: {np.mean(self.fps_history[-10:]):.1f}")
            
            print("=" * 50)

def main():
    """主函数"""
    print("正在启动最小延迟摄像头检测...")
    
    # 检查OpenCV
    try:
        cv2_version = cv2.__version__
        print(f"OpenCV版本: {cv2_version}")
    except:
        print("错误: OpenCV未安装")
        print("安装命令: pip install opencv-python")
        return
    
    # 创建检测器
    detector = MinimalCameraDetector(camera_id=0)
    
    # 运行检测
    detector.run()

if __name__ == "__main__":
    main()