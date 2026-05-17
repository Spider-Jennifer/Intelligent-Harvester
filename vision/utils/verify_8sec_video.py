#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os

video_file = "apple_detection_8sec.avi"
if not os.path.exists(video_file):
    print(f"视频文件不存在: {video_file}")
    exit(1)

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("无法打开视频文件")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps if fps > 0 else 0

print(f"视频文件: {video_file}")
print(f"尺寸: {width} x {height}")
print(f"帧数: {frame_count}")
print(f"帧率: {fps:.2f}")
print(f"时长: {duration:.2f} 秒")
print(f"文件大小: {os.path.getsize(video_file) / (1024*1024):.2f} MB")

# 读取前5帧确保可以解码
success_count = 0
for i in range(min(5, frame_count)):
    ret, frame = cap.read()
    if ret:
        success_count += 1
    else:
        print(f"帧 {i} 读取失败")

cap.release()
if success_count == min(5, frame_count):
    print(f"视频验证通过，包含 {frame_count} 帧，{duration:.2f} 秒")
else:
    print(f"视频验证失败：成功读取 {success_count}/{min(5, frame_count)} 帧")