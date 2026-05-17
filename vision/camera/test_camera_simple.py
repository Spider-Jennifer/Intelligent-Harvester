#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派摄像头测试脚本
用于测试摄像头是否正常工作
"""

import cv2
import time
import sys

def test_camera(camera_id=0):
    """测试摄像头"""
    print(f"测试摄像头索引 {camera_id}...")
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"摄像头 {camera_id} 无法打开")
        return False
    
    print(f"摄像头 {camera_id} 已打开")
    
    # 获取摄像头信息
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"摄像头信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # 尝试读取几帧
    print("尝试读取帧...")
    success_count = 0
    
    for i in range(10):
        ret, frame = cap.read()
        
        if ret:
            success_count += 1
            print(f"  第 {i+1} 帧: 成功, 尺寸: {frame.shape}")
        else:
            print(f"  第 {i+1} 帧: 失败")
        
        time.sleep(0.1)
    
    # 释放摄像头
    cap.release()
    
    if success_count > 0:
        print(f"✓ 摄像头 {camera_id} 测试成功 ({success_count}/10 帧)")
        return True
    else:
        print(f"✗ 摄像头 {camera_id} 测试失败")
        return False

def test_gstreamer():
    """测试GStreamer摄像头"""
    print("测试GStreamer摄像头...")
    
    try:
        # 树莓派CSI摄像头GStreamer管道
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
        
        if not cap.isOpened():
            print("GStreamer摄像头无法打开")
            return False
        
        print("GStreamer摄像头已打开")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"✓ GStreamer摄像头测试成功, 帧尺寸: {frame.shape}")
            return True
        else:
            print("✗ GStreamer摄像头读取失败")
            return False
            
    except Exception as e:
        print(f"GStreamer摄像头错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("树莓派摄像头测试工具")
    print("=" * 50)
    
    print("\n1. 测试标准摄像头...")
    success_standard = False
    
    # 测试多个摄像头索引
    for camera_id in [0, 1, 2]:
        if test_camera(camera_id):
            success_standard = True
            break
    
    print("\n2. 测试GStreamer摄像头...")
    success_gstreamer = test_gstreamer()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print("=" * 50)
    
    if success_standard:
        print("✓ 标准摄像头: 正常")
    else:
        print("✗ 标准摄像头: 异常")
    
    if success_gstreamer:
        print("✓ GStreamer摄像头: 正常")
    else:
        print("✗ GStreamer摄像头: 异常")
    
    print("\n故障排除建议:")
    print("1. 检查摄像头是否已启用:")
    print("   sudo raspi-config")
    print("   -> Interface Options")
    print("   -> Legacy Camera (或 Camera)")
    print("   -> Yes")
    print("   -> 重启树莓派")
    
    print("\n2. 测试摄像头命令:")
    print("   libcamera-hello -t 0")
    print("   或")
    print("   raspistill -o test.jpg")
    
    print("\n3. 检查摄像头连接:")
    print("   - 确保摄像头模块正确连接")
    print("   - 检查摄像头排线")
    
    print("\n4. 检查权限:")
    print("   - 确保用户有摄像头访问权限")
    print("   - 尝试: sudo usermod -a -G video $USER")
    
    return success_standard or success_gstreamer

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)