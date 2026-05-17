#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行延迟测试的包装脚本
"""

import os
import sys

# 设置工作目录
workspace = r"c:\Users\李晨鑫\Desktop\yolov8"
os.chdir(workspace)

print(f"工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")
print()

# 检查必要文件
if not os.path.exists("simple_latency_test.py"):
    print("错误: simple_latency_test.py 不存在")
    sys.exit(1)

# 导入并运行测试
try:
    # 动态导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("latency_test", "simple_latency_test.py")
    latency_test = importlib.util.module_from_spec(spec)
    
    # 重定向输出
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 创建输出文件
    output_file = open("latency_test_output.txt", "w", encoding='utf-8')
    sys.stdout = output_file
    sys.stderr = output_file
    
    print("=" * 60)
    print("模型推理延迟测试 - 开始")
    print("=" * 60)
    print()
    
    # 执行测试
    spec.loader.exec_module(latency_test)
    
    print()
    print("=" * 60)
    print("模型推理延迟测试 - 完成")
    print("=" * 60)
    
    output_file.close()
    
except Exception as e:
    print(f"运行测试时出错: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 恢复标准输出
    sys.stdout = original_stdout
    sys.stderr = original_stderr

print("测试已完成，输出已保存到 latency_test_output.txt")
print("检查 latency_results 目录查看图表和报告")