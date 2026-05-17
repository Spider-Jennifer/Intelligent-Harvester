#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动修正苹果检测模型误识别问题的最终脚本
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("正在启动苹果检测模型修正脚本...")
    import fix_apple_misdetection_final
    fix_apple_misdetection_final.main()
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保 fix_apple_misdetection_final.py 在同一目录下")
except Exception as e:
    print(f"运行过程中出错: {e}")
    import traceback
    traceback.print_exc()

input("\n按Enter键退出...")