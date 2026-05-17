#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派客户端 - 极致精简版
最低资源占用，适合树莓派4B
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import time
import sys

class UltraLightClient:
    def __init__(self, server_ip='192.168.45.217', server_port=9999):
        """
        初始化极致精简客户端
        
        参数:
            server_ip: 电脑服务器IP地址
            server_port: 服务器端口
        """
        self.server_ip = server_ip
        self.server_port = server_port
        
        # 硬编码优化参数（树莓派专用）
        self.width = 320
        self.height = 240
        self.target_fps = 8   # 进一步降低FPS
        self.skip_frames = 3  # 增加跳帧数
        
        self.frame_counter = 0
        self.running = False
        
        # 性能统计（最小化）
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 连接服务器
        self.connect()
    
    def connect(self):
        """连接到服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((self.server_ip, self.server_port))
            print(f"✓ 连接到 {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False
    
    def send_receive(self, frame):
        """发送帧并接收结果（单次操作）"""
        try:
            # 序列化
            data = pickle.dumps(frame)
            size = struct.pack("L", len(data))
            
            # 发送
            self.sock.sendall(size + data)
            
            # 接收大小
            data = b""
            while len(data) < 4:
                data += self.sock.recv(4)
            result_size = struct.unpack("L", data[:4])[0]
            
            # 接收数据
            data = data[4:]
            while len(data) < result_size:
                data += self.sock.recv(4096)
            
            # 反序列化
            return pickle.loads(data[:result_size])
            
        except Exception as e:
            print(f"网络错误: {e}")
            return None
    
    def run(self):
        """主运行函数"""
        print("启动摄像头...")
        
        # 尝试多种摄像头初始化方法
        cap = None
        camera_indices = [0, 1, -1]  # -1表示使用GStreamer
        
        for camera_id in camera_indices:
            try:
                if camera_id == -1:
                    # 使用GStreamer管道（树莓派专用）
                    print("尝试GStreamer摄像头...")
                    gstreamer_pipeline = (
                        "nvarguscamerasrc ! "
                        "video/x-raw(memory:NVMM), width=320, height=240, format=NV12, framerate=15/1 ! "
                        "nvvidconv flip-method=0 ! "
                        "video/x-raw, width=320, height=240, format=BGRx ! "
                        "videoconvert ! "
                        "video/x-raw, format=BGR ! "
                        "appsink"
                    )
                    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
                else:
                    print(f"尝试摄像头索引 {camera_id}...")
                    cap = cv2.VideoCapture(camera_id)
                
                if cap is not None and cap.isOpened():
                    # 等待摄像头初始化
                    time.sleep(0.5)
                    
                    # 测试读取一帧
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"✓ 摄像头 {camera_id} 初始化成功")
                        break
                    else:
                        print(f"✗ 摄像头 {camera_id} 读取失败")
                        cap.release()
                        cap = None
                else:
                    print(f"✗ 摄像头 {camera_id} 打开失败")
                    if cap:
                        cap.release()
                        cap = None
                        
            except Exception as e:
                print(f"摄像头 {camera_id} 错误: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            print("摄像头打开失败，请检查:")
            print("1. 摄像头是否已启用: sudo raspi-config -> Interface -> Camera")
            print("2. 摄像头是否连接正常")
            print("3. 运行测试命令: libcamera-hello -t 0")
            return
        
        # 设置参数（强制设置）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # 验证参数设置
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps}FPS")
        
        print("开始远程推理...")
        print("按 'q' 退出")
        
        self.running = True
        detections = []
        
        while self.running:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 跳帧处理
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0:
                continue
            
            # 发送推理
            result = self.send_receive(frame)
            
            if result:
                detections = result.get('detections', [])
            
            # 绘制结果（简化）
            display = frame.copy()
            
            # 绘制检测框
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 简化标签
                label = f"{conf:.1f}"
                cv2.putText(display, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 计算FPS
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
            
            # 显示FPS
            cv2.putText(display, f"FPS: {self.fps:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示检测数量
            cv2.putText(display, f"检测: {len(detections)}", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示
            cv2.imshow('树莓派远程检测', display)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        self.sock.close()
        print("客户端已停止")

def main():
    """主函数"""
    print("=" * 50)
    print("树莓派远程推理客户端 - 极致精简版")
    print("=" * 50)
    
    # 使用电脑IP地址
    server_ip = "192.168.45.217"
    
    print(f"服务器: {server_ip}:9999")
    print("正在连接...")
    
    # 创建客户端
    client = UltraLightClient(server_ip=server_ip, server_port=9999)
    
    if client.sock:
        client.run()
    else:
        print("连接失败，请检查:")
        print("1. 服务器IP是否正确")
        print("2. 服务器是否正在运行")
        print("3. 网络是否连通")

if __name__ == "__main__":
    main()