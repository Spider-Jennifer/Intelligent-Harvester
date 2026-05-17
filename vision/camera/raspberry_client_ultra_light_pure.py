#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import pickle
import struct
import cv2
import numpy as np
import time
import sys

class UltraLightClient:
    def __init__(self, server_ip='192.168.45.217', server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        
        self.width = 320
        self.height = 240
        self.target_fps = 8
        self.skip_frames = 3
        
        self.frame_counter = 0
        self.running = False
        
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        self.connect()
    
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((self.server_ip, self.server_port))
            print(f"Connected to {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_receive(self, frame):
        try:
            data = pickle.dumps(frame)
            size = struct.pack("L", len(data))
            
            self.sock.sendall(size + data)
            
            data = b""
            while len(data) < 4:
                data += self.sock.recv(4)
            result_size = struct.unpack("L", data[:4])[0]
            
            data = data[4:]
            while len(data) < result_size:
                data += self.sock.recv(4096)
            
            return pickle.loads(data[:result_size])
            
        except Exception as e:
            print(f"Network error: {e}")
            return None
    
    def run(self):
        print("Starting camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("Camera open failed")
                return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        print("Starting remote inference...")
        print("Press 'q' to quit")
        
        self.running = True
        detections = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0:
                continue
            
            result = self.send_receive(frame)
            
            if result:
                detections = result.get('detections', [])
            
            display = frame.copy()
            
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{conf:.1f}"
                cv2.putText(display, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
            
            cv2.putText(display, f"FPS: {self.fps:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(display, f"Detect: {len(detections)}", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Raspberry Remote Detection', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.sock.close()
        print("Client stopped")

def main():
    print("=" * 50)
    print("Raspberry Remote Inference Client - Ultra Light")
    print("=" * 50)
    
    server_ip = "192.168.45.217"
    
    print(f"Server: {server_ip}:9999")
    print("Connecting...")
    
    client = UltraLightClient(server_ip=server_ip, server_port=9999)
    
    if client.sock:
        client.run()
    else:
        print("Connection failed, please check:")
        print("1. Server IP correct")
        print("2. Server running")
        print("3. Network connected")

if __name__ == "__main__":
    main()