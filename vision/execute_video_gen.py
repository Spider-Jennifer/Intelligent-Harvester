#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用runpy直接运行视频生成脚本，避免路径编码问题
"""
import os
import sys
import runpy

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"工作目录: {os.getcwd()}")

# 检查必要文件
script_path = "detect_photos_to_video_fixed.py"
if not os.path.exists(script_path):
    print(f"❌ 找不到脚本: {script_path}")
    sys.exit(1)

print(f"找到脚本: {script_path}")

# 删除旧视频文件
old_video = "apple_detection_video_fixed.avi"
if os.path.exists(old_video):
    print(f"删除旧视频文件: {old_video}")
    os.remove(old_video)

print("\n" + "="*60)
print("开始生成苹果检测视频...")
print("="*60)

try:
    # 使用runpy运行脚本
    runpy.run_path(script_path, run_name="__main__")
    
    print("\n" + "="*60)
    print("脚本执行完成!")
    print("="*60)
    
except Exception as e:
    print(f"❌ 运行脚本时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查输出文件
print("\n检查生成的视频文件:")
if os.path.exists("apple_detection_video_fixed.avi"):
    size = os.path.getsize("apple_detection_video_fixed.avi") / (1024*1024)
    print(f"  ✅ apple_detection_video_fixed.avi ({size:.2f} MB)")
    print(f"    文件路径: {os.path.abspath('apple_detection_video_fixed.avi')}")
    
    # 验证视频可读性
    try:
        import cv2
        cap = cv2.VideoCapture("apple_detection_video_fixed.avi")
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"    视频信息: {width}x{height}, {fps:.1f} FPS, {frame_count}帧, {duration:.1f}秒")
        else:
            print("    警告: 无法打开视频文件进行验证")
    except ImportError:
        print("    注意: 未安装opencv，跳过视频验证")
    
else:
    print("  ❌ 未找到生成的视频文件")
    sys.exit(1)

print("\n✅ 任务完成! 检测视频已成功生成。")