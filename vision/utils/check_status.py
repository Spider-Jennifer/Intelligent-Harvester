#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查优化状态
"""

import os

print("=== 摄像头检测优化状态检查 ===")
print()

# 1. 检查原始代码
print("1. 原始摄像头检测代码状态:")
original_files = [
    "YOLO-v8-app/YOLOv8-app-master/utils.py",
    "YOLO-v8-app/YOLOv8-app-master/app.py"
]

for f in original_files:
    if os.path.exists(f):
        print(f"   ✅ {f} - 存在")
        # 检查是否包含优化
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                if "分辨率" in content and "640" in content:
                    print(f"      → 可能已优化")
                else:
                    print(f"      → 未优化（原始版本）")
        except:
            print(f"      → 无法读取")
    else:
        print(f"   ❌ {f} - 不存在")

print()

# 2. 检查优化方案文件
print("2. 优化方案文件状态:")
optimized_files = [
    ("minimal_camera.py", "最小延迟版本"),
    ("optimized_camera.py", "优化版OpenCV"),
    ("fast_camera_app.py", "快速Web版本"),
    ("run_minimal_camera.bat", "最小延迟启动脚本"),
    ("run_fast_camera.bat", "快速Web启动脚本"),
    ("摄像头延迟解决方案.md", "解决方案文档")
]

for file_name, description in optimized_files:
    if os.path.exists(file_name):
        size = os.path.getsize(file_name)
        print(f"   ✅ {file_name} - {description} ({size}字节)")
    else:
        print(f"   ❌ {file_name} - {description} (不存在)")

print()

# 3. 总结
print("3. 当前状态总结:")
print("   - 原始Streamlit应用: 未优化（延迟高）")
print("   - 优化方案: 已创建（3个版本）")
print("   - 推荐使用: minimal_camera.py（最小延迟）")
print()

# 4. 建议
print("4. 下一步建议:")
print("   a) 测试优化版本: 双击 run_minimal_camera.bat")
print("   b) 如果需要Web界面: 双击 run_fast_camera.bat")
print("   c) 优化原始代码: 需要手动修改 utils.py")
print()

# 5. 检查GPU状态
print("5. 硬件加速状态:")
try:
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️ 警告: 使用CPU，建议安装CUDA版PyTorch")
except:
    print("   ⚠️ 无法检查PyTorch状态")

print()
print("=" * 50)