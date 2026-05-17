#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派客户端 - 连接电脑服务器进行远程推理
树莓派：运行此客户端，捕获摄像头图像，发送到服务器，接收并显示结果
"""

import socket
import threading
import pickle
import struct
import cv2
import numpy as np
import time
import os

class RaspberryClient:
    def __init__(self, server_ip='192.168.1.100', server_port=9999, camera_id=0):
        """
        初始化树莓派客户端
        
        参数:
            server_ip: 电脑服务器IP地址
            server_port: 服务器端口
            camera_id: 树莓派摄像头ID
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.camera_id = camera_id
        
        self.client_socket = None
        self.running = False
        self.connected = False
        
        # 摄像头参数（树莓派优化）
        self.width = 320
        self.height = 240
        self.target_fps = 10  # 降低帧率减少网络负载
        self.skip_frames = 2   # 跳帧处理
        self.cap = None
        
        # 性能统计
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # 检测结果
        self.latest_detections = []
        self.latest_inference_time = 0
        
        # 连接服务器
        self.connect_to_server()
    
    def connect_to_server(self):
        """连接到远程服务器"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5)
            self.client_socket.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"成功连接到服务器 {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"连接服务器失败: {e}")
            print("请检查:")
            print(f"  1. 服务器IP地址是否正确: {self.server_ip}")
            print(f"  2. 服务器端口是否正确: {self.server_port}")
            print(f"  3. 服务器是否正在运行")
            print(f"  4. 网络是否连通")
            return False
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("无法打开摄像头，尝试使用默认摄像头...")
                self.cap = cv2.VideoCapture(0)
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            print(f"摄像头启动成功: {self.width}x{self.height} @ {self.target_fps}FPS")
            return True
        except Exception as e:
            print(f"启动摄像头失败: {e}")
            return False
    
    def send_frame_for_inference(self, frame):
        """发送帧到服务器进行推理"""
        if not self.connected:
            return None
        
        try:
            # 序列化图像
            frame_pickle = pickle.dumps(frame)
            frame_size = struct.pack("L", len(frame_pickle))
            
            # 发送图像数据
            network_start = time.time()
            self.client_socket.sendall(frame_size + frame_pickle)
            
            # 接收结果
            data = b""
            payload_size = struct.calcsize("L")
            
            # 接收结果大小
            while len(data) < payload_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    self.connected = False
                    return None
                data += packet
            
            packed_result_size = data[:payload_size]
            data = data[payload_size:]
            result_size = struct.unpack("L", packed_result_size)[0]
            
            # 接收结果数据
            while len(data) < result_size:
                data += self.client_socket.recv(4096)
            
            result_data = data[:result_size]
            network_time = time.time() - network_start
            
            # 反序列化结果
            result = pickle.loads(result_data)
            
            # 更新统计
            self.total_network_time += network_time
            
            return result
            
        except Exception as e:
            print(f"发送/接收数据错误: {e}")
            self.connected = False
            return None
    
    def draw_detections(self, frame, detections):
        """在帧上绘制检测结果（简化版）"""
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # 绘制边界框（简化）
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 简化标签
            label = f"{confidence:.1f}"
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run(self):
        """主运行循环"""
        if not self.connected:
            print("未连接到服务器，无法运行")
            return
        
        if not self.start_camera():
            print("摄像头启动失败，无法运行")
            return
        
        self.running = True
        print("\n开始远程推理...")
        print("按 'q' 键退出")
        print("=" * 60)
        
        # 启动接收线程
        receive_thread = threading.Thread(target=self._receive_loop)
        receive_thread.daemon = True
        receive_thread.start()
        
        # 主循环
        while self.running:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                time.sleep(0.1)
                continue
            
            # 发送帧进行推理
            result = self.send_frame_for_inference(frame)
            
            if result:
                self.latest_detections = result['detections']
                self.latest_inference_time = result['inference_time']
            
            # 绘制检测结果
            display_frame = frame.copy()
            if self.latest_detections:
                display_frame = self.draw_detections(display_frame, self.latest_detections)
            
            # 显示性能信息
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # 添加性能信息到帧
            info_text = f"FPS: {self.fps:.1f}"
            if self.latest_inference_time > 0:
                info_text += f" | 推理: {self.latest_inference_time:.3f}s"
            if self.latest_detections:
                info_text += f" | 检测到: {len(self.latest_detections)}"
            
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示连接状态
            status_color = (0, 255, 0) if self.connected else (0, 0, 255)
            status_text = "已连接" if self.connected else "断开连接"
            cv2.putText(display_frame, f"状态: {status_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # 显示帧
            cv2.imshow('树莓派远程苹果检测', display_frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 清理
        self.stop()
    
    def _receive_loop(self):
        """接收循环（备用）"""
        while self.running and self.connected:
            time.sleep(0.01)
    
    def stop(self):
        """停止客户端"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.client_socket:
            self.client_socket.close()
        
        cv2.destroyAllWindows()
        print("\n客户端已停止")

def get_server_ip():
    """获取服务器IP地址"""
    # 尝试从用户输入获取
    default_ip = "192.168.1.100"  # 默认服务器IP
    
    print("请输入电脑服务器的IP地址:")
    print(f"默认: {default_ip} (按Enter使用默认)")
    user_input = input("服务器IP: ").strip()
    
    if user_input:
        return user_input
    else:
        return default_ip

def main():
    """主函数"""
    print("=" * 60)
    print("树莓派远程推理客户端")
    print("=" * 60)
    print("此客户端运行在树莓派上，连接电脑服务器进行YOLO推理")
    print()
    
    # 获取服务器IP
    server_ip = get_server_ip()
    server_port = 9999
    
    print(f"\n客户端配置:")
    print(f"  服务器IP: {server_ip}")
    print(f"  服务器端口: {server_port}")
    print(f"  摄像头ID: 0 (默认)")
    print()
    
    # 创建并运行客户端
    client = RaspberryClient(
        server_ip=server_ip,
        server_port=server_port,
        camera_id=0
    )
    
    if client.connected:
        client.run()
    else:
        print("无法连接到服务器，请检查配置后重试")

if __name__ == "__main__":
    main()