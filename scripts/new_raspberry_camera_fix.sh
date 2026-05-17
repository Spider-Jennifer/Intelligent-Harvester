#!/bin/bash
# 树莓派新版本（Bookworm/Bullseye）摄像头修复脚本
# 针对Interface Options中没有Camera选项的情况

echo "树莓派新版本摄像头修复工具"
echo "=============================="
echo "适用于：Raspberry Pi OS Bookworm/Bullseye"
echo "问题：raspi-config中没有Camera选项"
echo "=============================="

# 检查系统版本
echo -e "\n1. 检查系统版本："
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"
uname -a

# 检查摄像头硬件状态
echo -e "\n2. 检查摄像头硬件："
vcgencmd get_camera

# 方法1：使用非交互模式启用摄像头
echo -e "\n3. 启用摄像头接口（方法1 - 非交互模式）："
echo "执行: sudo raspi-config nonint do_camera 0"
sudo raspi-config nonint do_camera 0

# 检查启用状态
camera_status=$(sudo raspi-config nonint get_camera)
if [ "$camera_status" -eq 0 ]; then
    echo "✓ 摄像头接口已启用"
else
    echo "✗ 摄像头接口未启用 (状态: $camera_status)"
fi

# 方法2：手动编辑配置文件
echo -e "\n4. 配置摄像头参数（方法2 - 手动配置）："
echo "编辑 /boot/config.txt 文件..."

# 备份原文件
sudo cp /boot/config.txt /boot/config.txt.backup.$(date +%Y%m%d_%H%M%S)

# 检查并添加必要的配置
if ! grep -q "start_x=" /boot/config.txt; then
    echo "start_x=1" | sudo tee -a /boot/config.txt
    echo "✓ 添加 start_x=1"
else
    echo "start_x 已配置"
fi

if ! grep -q "gpu_mem=" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    echo "✓ 添加 gpu_mem=128"
else
    echo "gpu_mem 已配置"
fi

# 方法3：检查摄像头设备
echo -e "\n5. 检查摄像头设备文件："
ls -la /dev/video* 2>/dev/null || echo "未找到摄像头设备文件"

# 测试摄像头功能
echo -e "\n6. 测试摄像头功能："

# 测试libcamera
echo -e "\n6.1 测试 libcamera："
if command -v libcamera-hello &> /dev/null; then
    echo "运行 libcamera-hello --list-cameras："
    libcamera-hello --list-cameras 2>/dev/null || echo "命令执行失败"
    
    echo -e "\n运行快速预览（3秒）："
    timeout 3 libcamera-hello -t 0 2>/dev/null && echo "✓ libcamera预览成功" || echo "✗ libcamera预览失败"
else
    echo "libcamera 未安装"
    echo "安装命令: sudo apt install libcamera-apps"
fi

# Python摄像头测试（新版本兼容）
echo -e "\n6.2 Python摄像头测试："
cat > /tmp/camera_test_new.py << 'EOF'
#!/usr/bin/env python3
import os
import subprocess
import time

print("新版本树莓派摄像头兼容性测试")
print("=" * 50)

# 方法1: 检查libcamera
print("1. 检查libcamera支持...")
try:
    result = subprocess.run(["libcamera-hello", "--list-cameras"], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ libcamera可用")
        print("   输出:", result.stdout.strip()[:100])
    else:
        print("   ✗ libcamera不可用")
except:
    print("   ✗ libcamera未安装或不可用")

# 方法2: 尝试OpenCV
print("\n2. 测试OpenCV摄像头访问...")
try:
    import cv2
    
    # 尝试不同的摄像头索引
    for cam_id in [0, 1, 2]:
        print(f"   尝试摄像头索引 {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        
        if cap.isOpened():
            print(f"     ✓ 摄像头 {cam_id} 已打开")
            # 等待初始化
            time.sleep(1)
            
            # 尝试读取
            ret, frame = cap.read()
            if ret:
                print(f"     ✓ 成功读取帧: {frame.shape}")
                cap.release()
                break
            else:
                print(f"     ✗ 读取帧失败")
                cap.release()
        else:
            print(f"     ✗ 摄像头 {cam_id} 无法打开")
            
except ImportError:
    print("   ✗ OpenCV未安装")
    print("   安装命令: pip3 install opencv-python")
except Exception as e:
    print(f"   ✗ OpenCV错误: {e}")

# 方法3: 检查CSI摄像头
print("\n3. 检查CSI摄像头配置...")
config_file = "/boot/config.txt"
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        content = f.read()
        if "start_x=1" in content:
            print("   ✓ start_x已启用")
        else:
            print("   ✗ start_x未启用")
        
        if "gpu_mem=" in content:
            print("   ✓ gpu_mem已配置")
        else:
            print("   ✗ gpu_mem未配置")
else:
    print("   ✗ 配置文件不存在")

print("\n" + "=" * 50)
print("测试完成")
EOF

python3 /tmp/camera_test_new.py

# 清理
rm -f /tmp/camera_test_new.py

# 重启建议
echo -e "\n7. 重启建议："
echo "摄像头配置更改后需要重启才能生效。"
echo "是否立即重启？ (y/n)"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "正在重启树莓派..."
    sudo reboot
else
    echo "请稍后手动重启: sudo reboot"
    echo -e "\n重启后测试摄像头:"
    echo "  libcamera-hello -t 0"
    echo "  或"
    echo "  python3 -c \"import cv2; cap = cv2.VideoCapture(0); print('摄像头状态:', cap.isOpened())\""
fi

echo -e "\n=============================="
echo "修复完成"
echo "如果仍有问题，请检查："
echo "1. 摄像头硬件连接"
echo "2. 摄像头模块是否损坏"
echo "3. 系统是否为最新版本"
echo "=============================="