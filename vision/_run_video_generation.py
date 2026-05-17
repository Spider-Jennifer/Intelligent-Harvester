#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接运行视频生成脚本
"""
import os
import sys
import traceback

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"工作目录: {os.getcwd()}")

# 删除可能存在的旧视频文件
old_video = "apple_detection_video_fixed.avi"
if os.path.exists(old_video):
    print(f"删除旧视频文件: {old_video}")
    os.remove(old_video)

# 检查模型文件
print("\n检查可用的模型文件:")
pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
for f in pt_files:
    print(f"  - {f}")

# 检查图片目录
print("\n检查图片目录:")
if os.path.exists("apple_photos"):
    print(f"  ✅ apple_photos 存在")
    import glob
    images = glob.glob("apple_photos/*.jpg") + glob.glob("apple_photos/*.jpeg") + glob.glob("apple_photos/*.png")
    print(f"    找到 {len(images)} 张图片")
else:
    print("  ❌ apple_photos 不存在")
    sys.exit(1)

# 导入并运行视频生成脚本
print("\n" + "="*60)
print("开始生成检测视频...")
print("="*60)

try:
    # 添加当前目录到Python路径
    sys.path.insert(0, os.getcwd())
    
    # 导入脚本
    import detect_photos_to_video_fixed as video_script
    
    # 执行main函数
    video_script.main()
    
    print("\n" + "="*60)
    print("视频生成完成!")
    print("="*60)
    
except Exception as e:
    print(f"❌ 运行视频生成脚本时出错: {e}")
    traceback.print_exc()
    sys.exit(1)

# 检查输出文件
print("\n检查生成的视频文件:")
if os.path.exists("apple_detection_video_fixed.avi"):
    size = os.path.getsize("apple_detection_video_fixed.avi") / (1024*1024)
    print(f"  ✅ apple_detection_video_fixed.avi ({size:.2f} MB)")
    print(f"    文件路径: {os.path.abspath('apple_detection_video_fixed.avi')}")
else:
    print("  ❌ 未找到生成的视频文件")
    sys.exit(1)

print("\n✅ 任务完成! 检测视频已成功生成。")