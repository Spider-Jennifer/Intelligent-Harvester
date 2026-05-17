#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动脚本 - 运行苹果模型修正并生成视频
解决路径问题
"""
import os
import sys

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录: {os.getcwd()}")

# 检查必要文件
if not os.path.exists("fix_apple_misdetection_final.py"):
    print("错误: 找不到 fix_apple_misdetection_final.py")
    sys.exit(1)

# 导入并执行主脚本
print("正在启动苹果模型修正流程...")
print("=" * 80)

# 直接运行主脚本
exec(open("fix_apple_misdetection_final.py", encoding='utf-8').read())

print("=" * 80)
print("所有任务已完成！")