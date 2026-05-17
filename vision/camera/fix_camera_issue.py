#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复树莓派摄像头问题
解决摄像头超时和无法打开的问题
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    if description:
        print(f"\n{description}")
        print("-" * 50)
    
    print(f"执行: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
        else:
            print(f"✗ 失败 (代码: {result.returncode})")
            if result.stderr.strip():
                print(f"错误: {result.stderr.strip()}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ 命令超时")
        return False
    except Exception as e:
        print(f"✗ 异常: {e}")
        return False

def check_camera_status():
    """检查摄像头状态"""
    print("检查摄像头状态...")
    
    # 检查摄像头模块是否加载
    success = run_command(
        "vcgencmd get_camera",
        "1. 检查摄像头模块状态"
    )
    
    if not success:
        return False
    
    # 检查摄像头设备文件
    success = run_command(
        "ls -la /dev/video*",
        "2. 检查摄像头设备文件"
    )
    
    return success

def enable_camera():
    """启用摄像头"""
    print("\n启用摄像头...")
    
    # 检查是否已启用摄像头接口
    success = run_command(
        "sudo raspi-config nonint get_camera",
        "1. 检查摄像头是否已启用"
    )
    
    # 如果返回0表示已启用，1表示未启用
    # 这里我们只显示状态，不自动启用
    
    print("\n要启用摄像头，请运行以下命令:")
    print("sudo raspi-config")
    print("然后选择:")
    print("  3 Interface Options")
    print("  I1 Legacy Camera (或 Camera)")
    print("  Yes")
    print("  Finish")
    print("  重启树莓派")
    
    return True

def test_camera_commands():
    """测试摄像头命令"""
    print("\n测试摄像头命令...")
    
    # 测试libcamera
    success1 = run_command(
        "libcamera-hello --list-cameras",
        "1. 列出可用摄像头"
    )
    
    # 测试快速预览
    success2 = run_command(
        "timeout 2 libcamera-hello -t 0",
        "2. 快速摄像头预览"
    )
    
    # 测试拍照
    success3 = run_command(
        "libcamera-jpeg -o test_camera.jpg -t 1",
        "3. 拍照测试"
    )
    
    if success3 and os.path.exists("test_camera.jpg"):
        print(f"✓ 照片已保存: test_camera.jpg")
        os.remove("test_camera.jpg")
    
    return success1 or success2 or success3

def install_opencv_dependencies():
    """安装OpenCV依赖"""
    print("\n安装OpenCV依赖...")
    
    dependencies = [
        "libopencv-dev",
        "python3-opencv",
        "libatlas-base-dev",
        "libjasper-dev",
        "libqtgui4",
        "libqt4-test"
    ]
    
    for dep in dependencies:
        run_command(
            f"sudo apt-get install -y {dep}",
            f"安装 {dep}"
        )
    
    return True

def create_camera_test_script():
    """创建摄像头测试脚本"""
    script_content = '''#!/usr/bin/env python3
import cv2
import time

print("摄像头测试脚本")
print("=" * 50)

# 尝试多个摄像头索引
for camera_id in [0, 1, 2]:
    print(f"\\n尝试摄像头索引 {camera_id}...")
    
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"  摄像头 {camera_id} 无法打开")
            continue
        
        print(f"  ✓ 摄像头 {camera_id} 已打开")
        
        # 获取信息
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"    分辨率: {width}x{height}")
        print(f"    FPS: {fps}")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        
        if ret:
            print(f"  ✓ 成功读取帧，尺寸: {frame.shape}")
        else:
            print(f"  ✗ 读取帧失败")
        
        cap.release()
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")

print("\\n" + "=" * 50)
print("测试完成")
'''

    with open("camera_test.py", "w") as f:
        f.write(script_content)
    
    run_command("chmod +x camera_test.py", "设置脚本权限")
    print("✓ 摄像头测试脚本已创建: camera_test.py")
    
    return True

def main():
    """主函数"""
    print("=" * 70)
    print("树莓派摄像头问题修复工具")
    print("=" * 70)
    
    print("\n注意: 部分操作需要sudo权限")
    print("请确保在树莓派上运行此脚本")
    
    # 检查系统
    print("\n检查系统信息...")
    run_command("uname -a", "系统信息")
    run_command("cat /etc/os-release | grep PRETTY_NAME", "操作系统")
    
    # 检查摄像头状态
    check_camera_status()
    
    # 启用摄像头指导
    enable_camera()
    
    # 测试摄像头命令
    test_camera_commands()
    
    # 安装依赖
    install_opencv_dependencies()
    
    # 创建测试脚本
    create_camera_test_script()
    
    print("\n" + "=" * 70)
    print("修复步骤完成")
    print("=" * 70)
    
    print("\n下一步操作:")
    print("1. 如果摄像头未启用，请运行: sudo raspi-config")
    print("2. 启用摄像头后重启树莓派: sudo reboot")
    print("3. 重启后测试摄像头: python3 camera_test.py")
    print("4. 测试远程推理: python3 raspberry_client_ultra_light.py")
    
    print("\n常见问题解决方案:")
    print("1. 摄像头超时: 增加摄像头初始化等待时间")
    print("2. 无法打开摄像头: 检查权限和摄像头接口")
    print("3. 黑屏: 检查摄像头连接和光照")
    print("4. 低帧率: 降低分辨率和FPS设置")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)