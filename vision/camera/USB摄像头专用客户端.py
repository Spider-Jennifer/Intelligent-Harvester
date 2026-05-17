#!/usr/bin/env python3
"""
USB摄像头专用YOLOv8客户端 - 简化版
无需CSI摄像头配置，直接使用USB摄像头
"""

import cv2
import time
import requests
import numpy as np
import threading
import queue
import sys

class SimpleUSBCameraClient:
    def __init__(self, server_url="http://192.168.45.1:5000"):
        self.server_url = server_url
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # USB摄像头参数
        self.width = 640
        self.height = 480
        self.fps = 15
        
        # 自动检测的摄像头索引
        self.camera_index = None
        
    def auto_detect_usb_camera(self):
        """自动检测可用的USB摄像头"""
        print("自动检测USB摄像头...")
        
        # USB摄像头通常从索引0开始
        camera_indices = [0, 1, 2, 3]
        
        for idx in camera_indices:
            print(f"尝试摄像头索引 {idx}...")
            
            cap = cv2.VideoCapture(idx)
            
            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 等待初始化
            time.sleep(1)
            
            if cap.isOpened():
                # 测试读取
                ret, frame = cap.read()
                if ret:
                    print(f"✓ 找到USB摄像头: 索引 {idx}")
                    print(f"  分辨率: {frame.shape[1]}x{frame.shape[0]}")
                    
                    # 显示摄像头信息
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  实际参数: {actual_width}x{actual_height} @ {actual_fps}FPS")
                    
                    cap.release()
                    return idx
                else:
                    print(f"✗ 摄像头 {idx} 无法读取帧")
            else:
                print(f"✗ 摄像头 {idx} 无法打开")
            
            cap.release()
        
        return None
    
    def check_server_connection(self):
        """检查服务器连接"""
        print(f"检查服务器连接: {self.server_url}")
        
        try:
            response = requests.get(f"{self.server_url}/", timeout=3)
            if response.status_code == 200:
                print("✓ 服务器连接正常")
                return True
            else:
                print(f"✗ 服务器响应异常: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("✗ 无法连接到服务器")
            print("请确保服务器正在运行:")
            print("1. 在Windows上运行: python remote_inference_server.py")
            print("2. 或运行: start_remote_inference.bat")
            return False
            
        except Exception as e:
            print(f"✗ 连接错误: {e}")
            return False
    
    def camera_capture_thread(self):
        """摄像头采集线程"""
        print("启动摄像头采集...")
        
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_index)
        
        # 设置参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not cap.isOpened():
            print("错误: 摄像头打开失败")
            return
        
        print("摄像头已打开，开始采集...")
        
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败，重试...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # 显示FPS信息
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"摄像头FPS: {fps:.1f}")
                
                # 放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                else:
                    # 队列满时丢弃旧帧
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                print(f"摄像头错误: {e}")
                time.sleep(0.1)
        
        cap.release()
        print("摄像头采集结束")
    
    def detection_thread(self):
        """检测线程"""
        print("启动检测线程...")
        
        while self.running:
            try:
                # 从队列获取帧
                frame = self.frame_queue.get(timeout=1)
                
                # 编码为JPEG
                _, img_encoded = cv2.imencode('.jpg', frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # 发送到服务器
                try:
                    response = requests.post(
                        f"{self.server_url}/detect",
                        files={'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')},
                        timeout=3
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'detections' in result:
                            detections = result['detections']
                            
                            if detections:
                                print(f"检测到 {len(detections)} 个目标")
                                
                                # 显示检测结果
                                for det in detections:
                                    label = det.get('label', 'unknown')
                                    confidence = det.get('confidence', 0)
                                    bbox = det.get('bbox', [])
                                    
                                    if bbox:
                                        x1, y1, x2, y2 = bbox
                                        print(f"  {label}: {confidence:.2f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                            else:
                                print("未检测到目标")
                
                except requests.exceptions.RequestException as e:
                    print(f"网络错误: {e}")
                    
            except queue.Empty:
                # 队列空，继续等待
                pass
            except Exception as e:
                print(f"检测错误: {e}")
    
    def run(self):
        """主运行函数"""
        print("=" * 50)
        print("USB摄像头专用客户端")
        print("=" * 50)
        
        # 1. 自动检测USB摄像头
        self.camera_index = self.auto_detect_usb_camera()
        if self.camera_index is None:
            print("错误: 未找到可用的USB摄像头")
            print("请检查:")
            print("1. USB摄像头是否已连接")
            print("2. 摄像头是否正常工作")
            print("3. 尝试重新插拔USB摄像头")
            return
        
        # 2. 检查服务器连接
        if not self.check_server_connection():
            print("错误: 服务器连接失败")
            print(f"请确保服务器运行在: {self.server_url}")
            return
        
        print("\n准备启动客户端...")
        print("按 Ctrl+C 停止")
        
        self.running = True
        
        # 启动线程
        camera_thread = threading.Thread(target=self.camera_capture_thread)
        detection_thread = threading.Thread(target=self.detection_thread)
        
        camera_thread.start()
        detection_thread.start()
        
        try:
            # 主循环
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n收到停止信号...")
        
        finally:
            # 停止线程
            self.running = False
            camera_thread.join(timeout=2)
            detection_thread.join(timeout=2)
            print("客户端已停止")

def main():
    """主函数"""
    # 设置服务器URL（根据您的实际情况修改）
    server_url = "http://192.168.45.1:5000"  # 默认服务器地址
    
    # 如果有命令行参数，使用参数作为服务器地址
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    print(f"使用服务器: {server_url}")
    
    # 创建并运行客户端
    client = SimpleUSBCameraClient(server_url=server_url)
    client.run()

if __name__ == "__main__":
    main()