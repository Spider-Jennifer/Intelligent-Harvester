#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派客户端 - 摄像头超时修复版
专门解决摄像头初始化超时问题
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import time
import sys

class CameraFixedClient:
    def __init__(self, server_ip='192.168.45.217', server_port=9999):
        """
        初始化摄像头修复版客户端
        
        参数:
            server_ip: 电脑服务器IP地址
            server_port: 服务器端口
        """
        self.server_ip = server_ip
        self.server_port = server_port
        
        # 优化参数（针对树莓派摄像头）
        self.width = 320
        self.height = 240
        self.target_fps = 10   # 降低FPS要求
        self.skip_frames = 3   # 增加跳帧
        
        self.frame_counter = 0
        self.running = False
        
        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 连接服务器
        self.connect()
    
    def connect(self):
        """连接到服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)  # 增加超时时间
            self.sock.connect((self.server_ip, self.server_port))
            print(f"✓ 连接到 {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            print("请检查:")
            print(f"  1. 服务器IP是否正确: {self.server_ip}")
            print(f"  2. 服务器端口是否开放: {self.server_port}")
            print(f"  3. 服务器是否正在运行")
            return False
    
    def init_camera_with_retry(self):
        """
        带重试的摄像头初始化
        解决摄像头超时问题
        """
        print("初始化摄像头（带重试机制）...")
        
        max_retries = 3
        camera_methods = [
            # 方法1: 标准摄像头
            lambda: cv2.VideoCapture(0),
            # 方法2: 备用摄像头
            lambda: cv2.VideoCapture(1),
            # 方法3: GStreamer（树莓派CSI摄像头）
            lambda: cv2.VideoCapture(
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=320, height=240, format=NV12, framerate=15/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, width=320, height=240, format=BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! "
                "appsink",
                cv2.CAP_GSTREAMER
            ),
            # 方法4: V4L2
            lambda: cv2.VideoCapture("/dev/video0"),
            # 方法5: V4L2备用
            lambda: cv2.VideoCapture("/dev/video1"),
        ]
        
        for retry in range(max_retries):
            print(f"\n尝试 {retry + 1}/{max_retries}...")
            
            for method_idx, camera_method in enumerate(camera_methods):
                method_name = ["标准", "备用", "GStreamer", "V4L2", "V4L2备用"][method_idx]
                print(f"  尝试 {method_name} 方法...")
                
                try:
                    cap = camera_method()
                    
                    if cap is not None and cap.isOpened():
                        # 等待摄像头稳定
                        time.sleep(0.5)
                        
                        # 测试读取
                        for test_attempt in range(3):
                            ret, test_frame = cap.read()
                            if ret:
                                print(f"    ✓ {method_name} 摄像头初始化成功")
                                
                                # 设置参数
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                                
                                # 验证参数
                                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                                
                                print(f"    摄像头参数: {actual_width}x{actual_height} @ {actual_fps}FPS")
                                return cap
                            else:
                                print(f"    测试读取 {test_attempt + 1}/3 失败")
                                time.sleep(0.1)
                        
                        # 如果测试读取都失败，释放摄像头
                        cap.release()
                    else:
                        print(f"    ✗ {method_name} 摄像头无法打开")
                        
                except Exception as e:
                    print(f"    ✗ {method_name} 错误: {e}")
            
            # 如果所有方法都失败，等待后重试
            if retry < max_retries - 1:
                print(f"等待 2 秒后重试...")
                time.sleep(2)
        
        # 所有重试都失败
        print("\n✗ 摄像头初始化失败，请检查:")
        print("  1. 摄像头是否已启用: sudo raspi-config -> Interface -> Camera")
        print("  2. 摄像头是否连接正常")
        print("  3. 运行测试命令: libcamera-hello -t 0")
        print("  4. 检查权限: ls -la /dev/video*")
        return None
    
    def send_receive(self, frame):
        """发送帧并接收结果"""
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
        # 初始化摄像头（带重试）
        cap = self.init_camera_with_retry()
        if cap is None:
            return
        
        print("\n开始远程推理...")
        print("按 'q' 或 ESC 退出")
        
        self.running = True
        detections = []
        
        try:
            while self.running:
                # 读取帧（带超时保护）
                start_read = time.time()
                ret, frame = cap.read()
                read_time = time.time() - start_read
                
                if read_time > 1.0:
                    print(f"警告: 帧读取时间过长 ({read_time:.2f}秒)")
                
                if not ret:
                    print("摄像头读取失败，尝试重新初始化...")
                    time.sleep(0.5)
                    continue
                
                # 跳帧处理
                self.frame_counter += 1
                if self.frame_counter % self.skip_frames != 0:
                    continue
                
                # 发送推理
                result = self.send_receive(frame)
                
                if result:
                    detections = result.get('detections', [])
                
                # 绘制结果
                display = frame.copy()
                
                # 绘制检测框
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
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
                
                # 显示FPS和状态
                cv2.putText(display, f"FPS: {self.fps:.1f}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"检测: {len(detections)}", (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"连接: {self.server_ip}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示
                cv2.imshow('树莓派远程检测 - 修复版', display)
                
                # 检查退出
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:  # ESC或Q
                    break
                
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            # 清理
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'sock'):
                self.sock.close()
            print("客户端已停止")

def main():
    """主函数"""
    print("=" * 60)
    print("树莓派远程推理客户端 - 摄像头超时修复版")
    print("=" * 60)
    
    # 使用电脑IP地址
    server_ip = "192.168.45.217"
    
    print(f"服务器: {server_ip}:9999")
    print("正在连接...")
    
    # 创建客户端
    client = CameraFixedClient(server_ip=server_ip, server_port=9999)
    
    if hasattr(client, 'sock') and client.sock:
        client.run()
    else:
        print("连接失败，请检查网络和服务器状态")

if __name__ == "__main__":
    main()