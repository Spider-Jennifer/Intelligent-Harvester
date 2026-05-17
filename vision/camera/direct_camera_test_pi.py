#!/usr/bin/env python3
"""
直接在树莓派上运行的摄像头测试脚本
无需传输文件，直接复制代码到树莓派SSH中运行
"""

import subprocess
import os
import sys

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令超时"
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 60)
    print("树莓派摄像头直接测试工具")
    print("=" * 60)
    print()
    
    # 检查是否以root运行
    if os.geteuid() != 0:
        print("⚠ 建议使用sudo运行此脚本以获得更好的测试结果")
        print()
    
    # 1. 检查系统信息
    print("1. 系统信息:")
    print("-" * 40)
    success, out, err = run_command("cat /etc/os-release | grep PRETTY_NAME")
    if success:
        print(f"  系统: {out.strip()}")
    
    success, out, err = run_command("uname -a")
    if success:
        print(f"  内核: {out.strip()[:50]}...")
    print()
    
    # 2. 检查摄像头硬件
    print("2. 摄像头硬件检查:")
    print("-" * 40)
    success, out, err = run_command("vcgencmd get_camera")
    if success:
        print(f"  摄像头模块: {out.strip()}")
    else:
        print("  摄像头模块: 未检测到")
    
    success, out, err = run_command("ls /dev/video* 2>/dev/null")
    if success:
        devices = out.strip().split()
        print(f"  摄像头设备: {len(devices)} 个设备")
        for dev in devices:
            print(f"    {dev}")
    else:
        print("  摄像头设备: 未找到")
    print()
    
    # 3. 检查摄像头配置
    print("3. 摄像头配置检查:")
    print("-" * 40)
    success, out, err = run_command("raspi-config nonint get_camera")
    if success:
        status = "已启用" if out.strip() == "0" else "未启用"
        print(f"  摄像头接口: {status} ({out.strip()})")
    
    success, out, err = run_command("grep -E 'start_x|gpu_mem' /boot/config.txt")
    if success:
        print("  /boot/config.txt 配置:")
        for line in out.strip().split('\n'):
            print(f"    {line}")
    else:
        print("  /boot/config.txt: 未找到相关配置")
    print()
    
    # 4. 测试libcamera
    print("4. libcamera测试:")
    print("-" * 40)
    print("  测试libcamera-hello (2秒)...")
    success, out, err = run_command("timeout 2 libcamera-hello -t 0")
    if success:
        print("  ✓ libcamera工作正常")
    else:
        print("  ✗ libcamera测试失败")
        print(f"    错误: {err[:50]}...")
    print()
    
    # 5. Python OpenCV测试
    print("5. Python OpenCV摄像头测试:")
    print("-" * 40)
    
    try:
        import cv2
        print("  ✓ OpenCV已安装")
        
        # 测试不同的摄像头索引
        for camera_id in [0, 1, 2]:
            print(f"  尝试摄像头索引 {camera_id}...")
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    print(f"    ✓ 摄像头 {camera_id} 已打开")
                    
                    import time
                    time.sleep(0.5)  # 等待初始化
                    
                    ret, frame = cap.read()
                    if ret:
                        print(f"    ✓ 读取成功: {frame.shape}")
                        cap.release()
                        print("\n  ✓ Python摄像头测试通过")
                        break
                    else:
                        print(f"    ✗ 读取失败")
                        cap.release()
                else:
                    print(f"    ✗ 无法打开")
            except Exception as e:
                print(f"    ✗ 错误: {str(e)[:50]}...")
        else:
            print("\n  ✗ 所有摄像头索引都失败")
            
    except ImportError:
        print("  ✗ OpenCV未安装")
        print("    安装命令: sudo apt install python3-opencv")
    print()
    
    # 6. 修复建议
    print("6. 修复建议:")
    print("-" * 40)
    
    # 检查摄像头状态
    success, out, err = run_command("raspi-config nonint get_camera")
    if success and out.strip() == "1":
        print("  ⚠ 摄像头接口未启用")
        print("    启用命令: sudo raspi-config nonint do_camera 0")
    
    # 检查配置
    success, out, err = run_command("grep 'start_x=1' /boot/config.txt")
    if not success:
        print("  ⚠ start_x未设置为1")
        print("    添加命令: echo 'start_x=1' | sudo tee -a /boot/config.txt")
    
    success, out, err = run_command("grep 'gpu_mem=' /boot/config.txt")
    if not success:
        print("  ⚠ gpu_mem未设置")
        print("    添加命令: echo 'gpu_mem=128' | sudo tee -a /boot/config.txt")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    
    print("\n下一步操作:")
    print("1. 如果摄像头未启用，运行修复命令")
    print("2. 重启树莓派: sudo reboot")
    print("3. 重启后重新测试")
    print("4. 使用修复版客户端: python3 raspberry_client_camera_fixed.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n测试错误: {e}")
        sys.exit(1)