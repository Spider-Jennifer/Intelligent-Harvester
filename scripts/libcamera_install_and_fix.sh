#!/bin/bash

# 树莓派libcamera安装和摄像头修复脚本
# 解决: 1) libcamera-hello命令不存在 2) supported=0 detected=0

echo "=========================================="
echo "树莓派libcamera安装和摄像头修复脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印带颜色的消息
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以root运行
if [ "$EUID" -ne 0 ]; then 
    print_warning "建议使用sudo运行此脚本"
    print_status "请使用: sudo bash $0"
    exit 1
fi

print_status "开始libcamera安装和摄像头修复..."

# 1. 更新系统包列表
print_status "1. 更新系统包列表..."
apt update

# 2. 安装libcamera-apps和相关依赖
print_status "2. 安装libcamera-apps和相关依赖..."
apt install -y libcamera-apps python3-opencv python3-picamera2 v4l-utils

# 3. 检查摄像头接口状态
print_status "3. 检查摄像头接口状态..."
CAMERA_ENABLED=$(raspi-config nonint get_camera)
if [ "$CAMERA_ENABLED" -eq 0 ]; then
    print_success "摄像头接口已启用"
else
    print_warning "摄像头接口未启用，正在启用..."
    raspi-config nonint do_camera 0
    print_success "摄像头接口已启用"
fi

# 4. 配置/boot/config.txt
print_status "4. 配置/boot/config.txt..."
CONFIG_FILE="/boot/config.txt"

# 备份原配置
if [ -f "$CONFIG_FILE" ]; then
    BACKUP_FILE="/boot/config.txt.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    print_success "配置文件已备份到: $BACKUP_FILE"
fi

# 确保配置存在
if ! grep -q "^start_x=" "$CONFIG_FILE" 2>/dev/null; then
    echo "start_x=1" >> "$CONFIG_FILE"
    print_success "已添加: start_x=1"
fi

if ! grep -q "^gpu_mem=" "$CONFIG_FILE" 2>/dev/null; then
    echo "gpu_mem=128" >> "$CONFIG_FILE"
    print_success "已添加: gpu_mem=128"
fi

# 添加摄像头覆盖层（根据摄像头型号）
CAMERA_MODEL=""
if [ -f /proc/device-tree/model ]; then
    CAMERA_MODEL=$(cat /proc/device-tree/model)
    print_status "设备型号: $CAMERA_MODEL"
fi

# 检查摄像头型号并添加相应的dtoverlay
if echo "$CAMERA_MODEL" | grep -qi "pi 5"; then
    print_status "检测到树莓派5，添加imx219覆盖层..."
    if ! grep -q "^dtoverlay=imx219" "$CONFIG_FILE" 2>/dev/null; then
        echo "dtoverlay=imx219" >> "$CONFIG_FILE"
        print_success "已添加: dtoverlay=imx219"
    fi
else
    print_status "添加通用摄像头覆盖层..."
    if ! grep -q "^dtoverlay=vc4-kms-v3d" "$CONFIG_FILE" 2>/dev/null; then
        echo "dtoverlay=vc4-kms-v3d" >> "$CONFIG_FILE"
        print_success "已添加: dtoverlay=vc4-kms-v3d"
    fi
fi

# 5. 检查摄像头硬件连接
print_status "5. 检查摄像头硬件连接..."
print_warning "请确保摄像头排线正确连接:"
print_warning "1. CSI接口（靠近以太网接口）"
print_warning "2. 蓝色面朝向以太网接口"
print_warning "3. 听到'咔哒'声表示已锁紧"
read -p "检查完成后按Enter继续..."

# 6. 重启服务
print_status "6. 重启相关服务..."
systemctl restart systemd-udevd 2>/dev/null || true

# 7. 测试libcamera安装
print_status "7. 测试libcamera安装..."
if command -v libcamera-hello &> /dev/null; then
    print_success "libcamera-hello 命令已安装"
else
    print_error "libcamera-hello 命令未找到，尝试其他安装方法..."
    apt install -y libcamera-tools
fi

# 8. 创建测试脚本
print_status "8. 创建摄像头测试脚本..."
cat > /home/pi/test_camera_all_methods.py << 'EOF'
#!/usr/bin/env python3
"""
摄像头测试脚本 - 测试所有可能的摄像头访问方法
"""

import os
import time
import subprocess
import cv2

def check_system():
    """检查系统信息"""
    print("=" * 50)
    print("系统信息检查")
    print("=" * 50)
    
    # 检查摄像头状态
    try:
        result = subprocess.run(['vcgencmd', 'get_camera'], 
                              capture_output=True, text=True)
        print(f"摄像头状态: {result.stdout.strip()}")
    except:
        print("无法获取摄像头状态")
    
    # 检查摄像头接口
    try:
        result = subprocess.run(['raspi-config', 'nonint', 'get_camera'], 
                              capture_output=True, text=True)
        status = "已启用" if result.stdout.strip() == "0" else "未启用"
        print(f"摄像头接口: {status}")
    except:
        print("无法获取摄像头接口状态")
    
    # 检查设备文件
    print("\n摄像头设备文件:")
    for dev in ['/dev/video0', '/dev/video1', '/dev/video10', '/dev/video11']:
        if os.path.exists(dev):
            print(f"  {dev}: ✓ 存在")
        else:
            print(f"  {dev}: ✗ 不存在")

def test_opencv_cameras():
    """测试OpenCV摄像头"""
    print("\n" + "=" * 50)
    print("测试OpenCV摄像头")
    print("=" * 50)
    
    camera_indices = [0, 1, 2, 10, 11, 12]
    
    for idx in camera_indices:
        print(f"\n测试摄像头索引 {idx}...")
        cap = cv2.VideoCapture(idx)
        
        # 等待摄像头初始化
        time.sleep(1)
        
        if cap.isOpened():
            print(f"  ✓ 摄像头 {idx} 已打开")
            
            # 尝试读取一帧
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ 成功读取帧: {frame.shape}")
                cap.release()
                return idx, frame.shape
            else:
                print(f"  ✗ 无法读取帧")
        else:
            print(f"  ✗ 摄像头 {idx} 无法打开")
        
        cap.release()
    
    return None, None

def test_libcamera():
    """测试libcamera"""
    print("\n" + "=" * 50)
    print("测试libcamera")
    print("=" * 50)
    
    try:
        # 测试libcamera-hello
        print("运行 libcamera-hello...")
        result = subprocess.run(['timeout', '3', 'libcamera-hello', '-t', '0'],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✓ libcamera 工作正常")
            return True
        else:
            print(f"  ✗ libcamera 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ libcamera 错误: {e}")
        return False

def main():
    """主函数"""
    print("摄像头全面测试开始...")
    
    # 检查系统
    check_system()
    
    # 测试libcamera
    libcamera_ok = test_libcamera()
    
    # 测试OpenCV摄像头
    camera_idx, frame_shape = test_opencv_cameras()
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    if libcamera_ok:
        print("✓ libcamera: 工作正常")
    else:
        print("✗ libcamera: 需要安装或配置")
    
    if camera_idx is not None:
        print(f"✓ OpenCV摄像头: 索引 {camera_idx}, 分辨率 {frame_shape}")
    else:
        print("✗ OpenCV摄像头: 未找到可用摄像头")
    
    print("\n建议:")
    if not libcamera_ok:
        print("1. 运行: sudo apt install libcamera-apps")
    
    if camera_idx is None:
        print("2. 检查摄像头硬件连接")
        print("3. 确保摄像头接口已启用: sudo raspi-config nonint do_camera 0")
        print("4. 重启系统: sudo reboot")
    else:
        print(f"2. 在代码中使用摄像头索引: {camera_idx}")

if __name__ == "__main__":
    main()
EOF

chmod +x /home/pi/test_camera_all_methods.py
print_success "测试脚本已创建: /home/pi/test_camera_all_methods.py"

# 9. 显示下一步操作
print_status "9. 显示下一步操作..."
print_success "安装和配置完成！"
print_status "请执行以下步骤:"
print_status "1. 重启系统: sudo reboot"
print_status "2. 重启后运行测试: python3 /home/pi/test_camera_all_methods.py"
print_status "3. 如果测试成功，运行您的YOLOv8客户端"

# 10. 可选：立即重启
read -p "是否立即重启系统？(y/n): " REBOOT_NOW
if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
    print_status "正在重启系统..."
    reboot
else
    print_status "请手动重启系统以使配置生效"
fi