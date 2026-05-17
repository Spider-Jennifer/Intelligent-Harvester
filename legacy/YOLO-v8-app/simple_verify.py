#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 苹果检测应用 - 简化验证脚本
"""

import os
from pathlib import Path

def main():
    print("=" * 50)
    print("YOLOv8 苹果检测应用 - 项目验证")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    app_dir = project_root / "YOLOv8-app-master"
    
    # 检查关键文件
    files_to_check = [
        "app.py",
        "config.py", 
        "utils.py",
        "weights/best.pt"
    ]
    
    print("\n[1] 检查项目文件:")
    all_exist = True
    for file_path in files_to_check:
        full_path = app_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  OK: {file_path} ({size:,} bytes)")
        else:
            print(f"  ERROR: {file_path} - 文件不存在")
            all_exist = False
    
    # 检查代码修改
    print("\n[2] 检查代码修改:")
    
    # 检查 app.py
    app_py = app_dir / "app.py"
    if app_py.exists():
        content = app_py.read_text(encoding='utf-8')
        if "from PIL import Image" in content:
            print("  OK: app.py - PIL 导入已修复")
        else:
            print("  ERROR: app.py - PIL 导入未修复")
            all_exist = False
    
    # 检查 utils.py
    utils_py = app_dir / "utils.py"
    if utils_py.exists():
        content = utils_py.read_text(encoding='utf-8')
        if "weights_only=False" in content:
            print("  OK: utils.py - PyTorch 兼容性修复已保存")
        else:
            print("  ERROR: utils.py - PyTorch 兼容性修复未保存")
            all_exist = False
    
    # 检查备份文件
    print("\n[3] 检查备份文件:")
    backup_files = list(project_root.glob("YOLOv8_apple_detection_backup_*.zip"))
    if backup_files:
        latest_backup = max(backup_files, key=os.path.getctime)
        size = latest_backup.stat().st_size / 1024 / 1024
        print(f"  OK: 完整备份存在 ({size:.2f} MB)")
    else:
        print("  ERROR: 未找到备份文件")
        all_exist = False
    
    # 总结
    print("\n" + "=" * 50)
    if all_exist:
        print("SUCCESS: 所有检查通过！")
        print("项目完全保存，可以随时运行！")
        print("\n启动方法:")
        print("1. 双击: restore_and_run.bat")
        print("2. 访问: http://localhost:8501")
    else:
        print("ERROR: 部分检查未通过")
        print("请检查上述问题")
    
    print("=" * 50)
    return all_exist

if __name__ == "__main__":
    main()