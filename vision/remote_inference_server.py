#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
远程推理服务器 - 电脑端运行YOLO模型，树莓派显示结果
电脑端：运行此服务器，提供YOLO推理服务
树莓派：运行客户端，发送图像，接收检测结果
"""

import socket
import threading
import pickle
import struct
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

class RemoteInferenceServer:
    def __init__(self, host='0.0.0.0', port=9999, model_path='apple_sensitive.pt'):
        """
        初始化远程推理服务器
        
        参数:
            host: 服务器监听地址
            port: 服务器监听端口
            model_path: YOLO模型路径
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.server_socket = None
        self.running = False
        self.clients = []
        
        # 自动选择最佳模型
        self.model = self._load_best_model()
        
        # 性能统计
        self.total_requests = 0
        self.total_inference_time = 0
    
    def _load_best_model(self):
        """自动选择最佳可用模型"""
        # 模型优先级列表
        model_priority = [
            'apple_sensitive.pt',    # 最敏感的模型
            'apple_best.pt',         # 最佳模型
            'apple_improved.pt',     # 改进模型
            'apple_quick.pt',        # 快速模型
            'yolov8n.pt',            # 基础模型
            'best.pt'                # 通用最佳模型
        ]
        
        print("正在查找最佳模型...")
        for model_file in model_priority:
            if os.path.exists(model_file):
                print(f"找到模型: {model_file}")
                try:
                    model = YOLO(model_file)
                    print(f"✓ 模型加载成功: {model_file}")
                    return model
                except Exception as e:
                    print(f"✗ 模型加载失败 {model_file}: {e}")
                    continue
        
        raise FileNotFoundError("未找到任何可用的YOLO模型文件")
        
        # 性能统计
        self.total_requests = 0
        self.total_inference_time = 0
        
    def start(self):
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1)
        
        self.running = True
        print(f"远程推理服务器启动在 {self.host}:{self.port}")
        print("等待树莓派客户端连接...")
        
        # 启动接受连接的线程
        accept_thread = threading.Thread(target=self._accept_connections)
        accept_thread.daemon = True
        accept_thread.start()
        
        # 主循环
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n正在关闭服务器...")
            self.stop()
    
    def _accept_connections(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"客户端连接: {client_address}")
                
                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.clients.append(client_socket)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"接受连接错误: {e}")
    
    def _handle_client(self, client_socket, client_address):
        """处理单个客户端连接"""
        try:
            while self.running:
                # 接收图像数据
                data = b""
                payload_size = struct.calcsize("L")
                
                # 接收数据大小
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # 接收图像数据
                while len(data) < msg_size:
                    data += client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # 解码图像
                frame = pickle.loads(frame_data)
                
                # 执行推理（优化参数）
                inference_start = time.time()
                results = self.model(frame, conf=0.1, imgsz=320, verbose=False)
                inference_time = time.time() - inference_start
                
                # 准备返回结果（简化）
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            # 只返回必要信息，减少网络传输
                            detections.append({
                                'bbox': boxes[i].tolist(),
                                'confidence': float(confidences[i])
                            })
                
                # 发送结果
                result_data = {
                    'detections': detections,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
                
                result_pickle = pickle.dumps(result_data)
                result_size = struct.pack("L", len(result_pickle))
                
                client_socket.sendall(result_size + result_pickle)
                
                # 更新统计
                self.total_requests += 1
                self.total_inference_time += inference_time
                
                if self.total_requests % 10 == 0:
                    avg_time = self.total_inference_time / self.total_requests
                    print(f"已处理 {self.total_requests} 个请求，平均推理时间: {avg_time:.3f}s")
                    
        except (ConnectionResetError, BrokenPipeError):
            print(f"客户端断开连接: {client_address}")
        except Exception as e:
            print(f"处理客户端错误: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        print("服务器已停止")
        print(f"总计处理请求: {self.total_requests}")
        if self.total_requests > 0:
            print(f"平均推理时间: {self.total_inference_time/self.total_requests:.3f}s")

def main():
    """主函数"""
    print("=" * 60)
    print("远程YOLO推理服务器")
    print("=" * 60)
    print("此服务器运行在电脑上，提供YOLO模型推理服务")
    print("树莓派作为客户端，发送图像并接收检测结果")
    print()
    
    # 获取本机IP地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"本机IP地址: {local_ip}")
        print(f"树莓派需要连接到此IP地址")
    except:
        local_ip = "0.0.0.0"
        print("无法获取本机IP地址，使用0.0.0.0")
    
    # 启动服务器
    server = RemoteInferenceServer(
        host='0.0.0.0',  # 监听所有网络接口
        port=9999,
        model_path='apple_sensitive.pt'
    )
    
    print(f"\n服务器配置:")
    print(f"  监听地址: 0.0.0.0 (所有网络接口)")
    print(f"  监听端口: 9999")
    print(f"  模型文件: apple_sensitive.pt")
    print(f"\n树莓派客户端需要连接到: {local_ip}:9999")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        server.stop()
    except Exception as e:
        print(f"服务器启动失败: {e}")

if __name__ == "__main__":
    main()