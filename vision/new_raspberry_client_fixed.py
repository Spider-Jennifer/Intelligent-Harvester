#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派新版本客户端 - 兼容Bookworm/Bullseye
专门解决新版本树莓派OS摄像头问题
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import time
import sys
import subprocess
import os

class NewRaspberryClient:
    def __init__(self, server_ip='192.168.45.217', server_port=9999):
        """
        初始化新版本树莓派客户端
        
        参数:
            server_ip: 电脑服务器IP地址
            server_port: 服务器端口
        """
        self.server_ip = server_ip
        self.server_port = server_port
        
        # 优化参数（针对新版本树莓派）
        self.width = 320
        self.height = 240
        self.target_fps = 10
        self.skip_frames = 3
        
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
            self.sock.settimeout(5)
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
    
    def check_camera_system(self):
        """检查摄像头系统状态"""
        print("检查摄像头系统...")
        
        # 检查摄像头接口是否启用
        try:
            result = subprocess.run(
                ["sudo", "raspi-config", "nonint", "get_camera"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                if status == "0":
                    print("✓ 摄像头接口已启用")
                else:
                    print("✗ 摄像头接口未启用 (状态: {status})")
                    print("  启用命令: sudo raspi-config nonint do_camera 0")
        except:
            print("⚠ 无法检查摄像头接口状态")
        
        # 检查摄像头设备
        print("检查摄像头设备文件...")
        try:
            result = subprocess.run(
                ["ls", "-la", "/dev/video*"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"摄像头设备:\n{result.stdout}")
            else:
                print("未找到摄像头设备文件")
        except:
            print("无法检查摄像头设备")
        
        # 检查libcamera
        print("检查libcamera...")
        try:
            result = subprocess.run(
                ["which", "libcamera-hello"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ libcamera已安装")
            else:
                print("✗ libcamera未安装")
                print("  安装命令: sudo apt install libcamera-apps")
        except:
            print("无法检查libcamera")
    
    def init_camera_new_version(self):
        """
        新版本树莓派摄像头初始化
        兼容Bookworm/Bullseye
        """
        print("新版本树莓派摄像头初始化...")
        
        camera_methods = [
            # 方法1: 标准摄像头（可能不工作）
            {"name": "标准摄像头", "func": lambda: cv2.VideoCapture(0)},
            
            # 方法2: 备用摄像头
            {"name": "备用摄像头", "func": lambda: cv2.VideoCapture(1)},
            
            # 方法3: V4L2设备
            {"name": "V4L2设备", "func": lambda: cv2.VideoCapture("/dev/video0")},
            
            # 方法4: V4L2备用
            {"name": "V4L2备用", "func": lambda: cv2.VideoCapture("/dev/video1")},
            
            # 方法5: 使用libcamera的GStreamer管道（新版本推荐）
            {"name": "libcamera-GStreamer", "func": lambda: cv2.VideoCapture(
                "libcamera src ! "
                "video/x-raw,width=320,height=240,framerate=10/1 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink",
                cv2.CAP_GSTREAMER
            )},
            
            # 方法6: 简化GStreamer管道
            {"name": "简化GStreamer", "func": lambda: cv2.VideoCapture(
                "v4l2src device=/dev/video0 ! "
                "video/x-raw,width=320,height=240 ! "
                "videoconvert ! "
                "appsink",
                cv2.CAP_GSTREAMER
            )},
        ]
        
        for method in camera_methods:
            print(f"  尝试 {method['name']}...")
            cap = None
            
            try:
                cap = method['func']()
                
                if cap is not None and cap.isOpened():
                    # 等待摄像头稳定
                    time.sleep(1)
                    
                    # 测试读取
                    for attempt in range(3):
                        ret, frame = cap.read()
                        if ret:
                            print(f"    ✓ {method['name']} 初始化成功")
                            
                            # 尝试设置参数（可能不支持）
                            try:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                            except:
                                print("    ⚠ 无法设置摄像头参数，使用默认值")
                            
                            return cap
                        else:
                            time.sleep(0.2)
                    
                    # 测试读取失败
                    print(f"    ✗ {method['name']} 读取失败")
                    cap.release()
                else:
                    print(f"    ✗ {method['name']} 无法打开")
                    
            except Exception as e:
                print(f"    ✗ {method['name']} 错误: {e}")
                if cap:
                    cap.release()
        
        # 所有方法都失败
        print("\n✗ 所有摄像头初始化方法都失败")
        print("\n请执行以下步骤：")
        print("1. 启用摄像头接口:")
        print("   sudo raspi-config nonint do_camera 0")
        print("2. 添加摄像头配置到 /boot/config.txt:")
        print("   start_x=1")
        print("   gpu_mem=128")
        print("3. 重启树莓派:")
        print("   sudo reboot")
        print("4. 安装libcamera（如果需要）:")
        print("   sudo apt install libcamera-apps")
        print("5. 安装OpenCV（如果需要）:")
        print("   pip3 install opencv-python")
        
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
        # 检查摄像头系统
        self.check_camera_system()
        
        # 初始化摄像头（新版本兼容）
        cap = self.init_camera_new_version()
        if cap is None:
            return
        
        print("\n开始远程推理...")
        print("按 'q' 或 ESC 退出")
        
        self.running = True
        detections = []
        
        try:
            while self.running:
                # 读取帧
                start_read = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    print("摄像头读取失败，跳过此帧...")
                    time.sleep(0.1)
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
                
                # 显示信息
                cv2.putText(display, f"FPS: {self.fps:.1f}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"检测: {len(detections)}", (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, "新版本兼容客户端", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示
                cv2.imshow('树莓派远程检测 - 新版本', display)
                
                # 检查退出
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
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
    print("=" * 70)
    print("树莓派远程推理客户端 - 新版本兼容版")
    print("适用于：Raspberry Pi OS Bookworm/Bullseye")
    print("=" * 70)
    
    # 使用电脑IP地址
    server_ip = "192.168.45.217"
    
    print(f"服务器: {server_ip}:9999")
    print("正在连接...")
    
    # 创建客户端
    client = NewRaspberryClient(server_ip=server_ip, server_port=9999)
    
    if hasattr(client, 'sock') and client.sock:
        client.run()
    else:
        print("连接失败，请检查网络和服务器状态")

if __name__ == "__main__":
    main()