#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版摄像头苹果检测 - 解决延迟问题
"""

import cv2
import time
import threading
from queue import Queue
from ultralytics import YOLO
import numpy as np

class OptimizedCameraDetector:
    def __init__(self, model_path="best.pt", camera_id=0, target_fps=15):
        """
        初始化优化摄像头检测器
        
        参数:
            model_path: YOLOv8模型路径
            camera_id: 摄像头ID (0为默认摄像头)
            target_fps: 目标帧率
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 设置摄像头分辨率（降低分辨率以提高速度）
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 降低高度
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)   # 设置摄像头FPS
        
        # 多线程队列
        self.frame_queue = Queue(maxsize=2)  # 限制队列大小避免内存堆积
        self.running = False
        
        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_frame_time = 0
        
    def capture_frames(self):
        """摄像头帧捕获线程"""
        print("启动摄像头捕获线程...")
        while self.running:
            start_time = time.time()
            
            # 读取摄像头帧
            success, frame = self.cap.read()
            if not success:
                print("摄像头读取失败")
                break
            
            # 如果队列未满，放入帧
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # 控制帧率
            elapsed = time.time() - start_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
    
    def process_frames(self):
        """帧处理线程"""
        print("启动帧处理线程...")
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # 记录开始时间
                frame_start = time.time()
                
                # 执行推理（使用较小的图像尺寸）
                resized_frame = cv2.resize(frame, (320, 240))  # 进一步缩小推理尺寸
                
                inference_start = time.time()
                results = self.model(resized_frame, conf=0.1, verbose=False)  # 降低置信度阈值以提高检测灵敏度
                inference_time = time.time() - inference_start
                
                # 处理结果
                if results and len(results) > 0:
                    result = results[0]
                    
                    # 将检测框映射回原始尺寸
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        # 缩放框到原始尺寸
                        scale_x = frame.shape[1] / resized_frame.shape[1]
                        scale_y = frame.shape[0] / resized_frame.shape[0]
                        
                        for box, conf, cls in zip(boxes, confs, classes):
                            # 缩放框坐标
                            x1, y1, x2, y2 = box
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            # 在原始帧上绘制框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Apple: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('Optimized Apple Detection', frame)
                
                # 性能统计
                self.frame_count += 1
                self.total_inference_time += inference_time
                self.total_frame_time += (time.time() - frame_start)
                
                # 每10帧显示一次性能信息
                if self.frame_count % 10 == 0:
                    avg_inference = self.total_inference_time / self.frame_count
                    avg_frame = self.total_frame_time / self.frame_count
                    fps = 1.0 / avg_frame if avg_frame > 0 else 0
                    
                    print(f"帧数: {self.frame_count}, "
                          f"推理时间: {avg_inference:.3f}s, "
                          f"总帧时间: {avg_frame:.3f}s, "
                          f"FPS: {fps:.1f}")
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
    
    def run(self):
        """运行优化检测器"""
        print("启动优化摄像头检测...")
        print("按 'q' 键退出")
        
        self.running = True
        
        # 启动线程
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        
        capture_thread.start()
        process_thread.start()
        
        try:
            # 等待线程结束
            capture_thread.join()
            process_thread.join()
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终性能统计
        if self.frame_count > 0:
            print("\n=== 性能统计 ===")
            print(f"总处理帧数: {self.frame_count}")
            print(f"平均推理时间: {self.total_inference_time/self.frame_count:.3f}秒")
            print(f"平均帧处理时间: {self.total_frame_time/self.frame_count:.3f}秒")
            print(f"平均FPS: {self.frame_count/self.total_frame_time:.1f}")

def main():
    """主函数"""
    print("=== 优化版摄像头苹果检测 ===")
    print("解决延迟问题的方案:")
    print("1. 降低摄像头分辨率 (640x480)")
    print("2. 降低推理分辨率 (320x240)")
    print("3. 多线程处理 (捕获和处理分离)")
    print("4. 控制目标帧率 (15 FPS)")
    print("5. 使用队列限制内存使用")
    print("=" * 40)
    
    # 查找模型文件
    import os
    model_candidates = [
        "best.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "apple_best.pt",
        "apple_model.pt"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            print(f"找到模型文件: {model_path}")
            break
    
    if not model_path:
        # 尝试在子目录中查找
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pt") and ("best" in file or "yolov8" in file):
                    model_path = os.path.join(root, file)
                    print(f"找到模型文件: {model_path}")
                    break
            if model_path:
                break
    
    if not model_path:
        print("警告: 未找到模型文件，将尝试下载默认模型")
        model_path = "yolov8n.pt"
    
    # 创建并运行检测器
    detector = OptimizedCameraDetector(
        model_path=model_path,
        camera_id=0,
        target_fps=15
    )
    
    try:
        detector.run()
    except Exception as e:
        print(f"运行出错: {e}")
        detector.cleanup()

if __name__ == "__main__":
    main()