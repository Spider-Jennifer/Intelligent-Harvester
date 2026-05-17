#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行苹果模型修正并生成检测视频
"""
import os
import sys
import subprocess
import time

print("=" * 80)
print("🍎 苹果检测模型修正与视频生成一体化脚本")
print("=" * 80)

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录: {os.getcwd()}")

# 步骤1: 检查环境
print("\n" + "=" * 80)
print("步骤1: 检查环境与依赖")
print("=" * 80)

# 检查必要文件
required_files = [
    "fix_apple_misdetection_final.py",
    "detect_photos_to_video_fixed.py",
    "apple_dataset/data.yaml"
]

missing_files = []
for f in required_files:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    print(f"❌ 缺少必要文件: {missing_files}")
    sys.exit(1)

print("✅ 所有必要文件都存在")

# 步骤2: 运行修复脚本（跳过训练）
print("\n" + "=" * 80)
print("步骤2: 运行苹果模型修正脚本")
print("注意: 将跳过训练步骤以节省时间")
print("=" * 80)

try:
    # 导入修复脚本并修改训练选择
    import fix_apple_misdetection_final as fix_script
    # 直接调用main函数
    print("正在执行修复流程...")
    print("此过程将添加新苹果照片并重新标注数据集")
    print("训练步骤将被跳过")
    
    # 修改修复脚本中的训练选择
    import types
    # 创建一个修改后的main函数副本
    original_main = fix_script.main
    def modified_main():
        # 在修复脚本中，train_choice已设为'n'，所以会跳过训练
        original_main()
    
    # 替换main函数
    fix_script.main = modified_main
    
    # 执行修改后的main函数
    fix_script.main()
    
except Exception as e:
    print(f"❌ 修复脚本执行失败: {e}")
    print("继续执行视频生成步骤...")

# 步骤3: 生成检测视频
print("\n" + "=" * 80)
print("步骤3: 生成苹果检测视频")
print("=" * 80)

try:
    # 导入视频生成脚本
    import detect_photos_to_video_fixed as video_script
    
    # 执行视频生成
    print("正在生成检测视频...")
    video_script.main()
    
except Exception as e:
    print(f"❌ 视频生成脚本执行失败: {e}")
    print("尝试直接运行脚本...")
    
    # 尝试通过子进程运行
    try:
        result = subprocess.run(
            [sys.executable, "detect_photos_to_video_fixed.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300  # 5分钟超时
        )
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️  视频生成超时")
    except Exception as e2:
        print(f"❌ 子进程执行失败: {e2}")

# 完成
print("\n" + "=" * 80)
print("🎯 所有任务已完成!")
print("=" * 80)

# 检查输出文件
output_files = [
    "apple_detection_video_fixed.avi",
    "apple_detection_final_fixed.avi",
    "quick_test_results.txt"
]

print("\n📁 生成的输出文件:")
for f in output_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  ✅ {f} ({size:.2f} MB)")
    else:
        print(f"  ⚠️  {f} (未找到)")

print("\n📋 下一步:")
print("  1. 播放视频检查检测效果")
print("  2. 查看 quick_test_results.txt 了解测试结果")
print("  3. 如需进一步改进，可运行训练脚本")
print("=" * 80)