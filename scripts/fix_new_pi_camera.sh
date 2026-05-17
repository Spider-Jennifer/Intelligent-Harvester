#!/bin/bash
# 树莓派新版本摄像头修复脚本
# 直接在树莓派SSH中运行

echo "=========================================="
echo "  树莓派新版本摄像头修复工具"
echo "  (适用于Raspberry Pi OS Bookworm/Bullseye)"
echo "=========================================="
echo

# 检查是否以root运行
if [ "$EUID" -ne 0 ]; then 
    echo "请使用sudo运行此脚本:"
    echo "  sudo bash fix_new_pi_camera.sh"
    exit 1
fi

# 1. 显示系统信息
echo "1. 系统信息:"
echo "------------------------------------------"
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"
uname -a
echo

# 2. 检查当前摄像头状态
echo "2. 摄像头状态检查:"
echo "------------------------------------------"
echo -n "摄像头模块: "
vcgencmd get_camera
echo -n "摄像头设备: "
ls /dev/video* 2>/dev/null || echo "未找到"
echo -n "raspi-config摄像头状态: "
raspi-config nonint get_camera
echo

# 3. 启用摄像头
echo "3. 启用摄像头接口:"
echo "------------------------------------------"
echo "启用摄像头..."
raspi-config nonint do_camera 0
echo "摄像头状态: $(raspi-config nonint get_camera) (0=已启用, 1=未启用)"
echo

# 4. 配置/boot/config.txt
echo "4. 配置/boot/config.txt:"
echo "------------------------------------------"
CONFIG_FILE="/boot/config.txt"

# 检查并设置start_x
if grep -q "^start_x=" "$CONFIG_FILE"; then
    echo "修改现有 start_x 设置..."
    sudo sed -i 's/^start_x=.*/start_x=1/' "$CONFIG_FILE"
else
    echo "添加 start_x 设置..."
    echo "start_x=1" | sudo tee -a "$CONFIG_FILE"
fi

# 检查并设置gpu_mem
if grep -q "^gpu_mem=" "$CONFIG_FILE"; then
    echo "修改现有 gpu_mem 设置..."
    sudo sed -i 's/^gpu_mem=.*/gpu_mem=128/' "$CONFIG_FILE"
else
    echo "添加 gpu_mem 设置..."
    echo "gpu_mem=128" | sudo tee -a "$CONFIG_FILE"
fi

echo "配置文件已更新"
echo

# 5. 安装必要软件
echo "5. 安装必要软件:"
echo "------------------------------------------"
echo "更新软件包列表..."
apt update

echo "安装libcamera-apps..."
apt install -y libcamera-apps

echo "安装Python OpenCV..."
apt install -y python3-opencv

echo "安装GStreamer支持..."
apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good \
               gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
echo

# 6. 测试摄像头
echo "6. 摄像头测试:"
echo "------------------------------------------"
echo "测试1: libcamera-hello (2秒预览)"
echo "按Ctrl+C停止测试"
timeout 3 libcamera-hello -t 0 2>/dev/null || echo "测试失败或用户中断"
echo

echo "测试2: 拍照测试"
libcamera-jpeg -o /tmp/camera_test_fix.jpg -t 1 -n 2>/dev/null
if [ -f "/tmp/camera_test_fix.jpg" ]; then
    echo "✓ 拍照成功: /tmp/camera_test_fix.jpg"
    ls -lh /tmp/camera_test_fix.jpg
    rm -f /tmp/camera_test_fix.jpg
else
    echo "✗ 拍照失败"
fi
echo

# 7. Python摄像头测试
echo "7. Python摄像头测试:"
echo "------------------------------------------"
python3 -c "
import cv2
import time
import sys

print('Python摄像头测试...')
print('尝试不同的初始化方法:')

# 方法1: libcamera GStreamer
print('1. 尝试libcamera GStreamer...')
try:
    pipeline = 'libcamera src ! video/x-raw,width=320,height=240 ! videoconvert ! appsink'
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print('  ✓ GStreamer摄像头已打开')
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print(f'  ✓ 读取成功: {frame.shape}')
            cap.release()
            print('\\n摄像头测试通过 ✓')
            sys.exit(0)
        else:
            print('  ✗ 读取失败')
            cap.release()
    else:
        print('  ✗ GStreamer无法打开')
except Exception as e:
    print(f'  ✗ GStreamer错误: {str(e)[:50]}...')

# 方法2: 标准摄像头索引
print('\\n2. 尝试标准摄像头索引...')
for cam_id in [0, 1, 2]:
    print(f'  尝试索引 {cam_id}...')
    try:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f'    ✓ 摄像头 {cam_id} 已打开')
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret:
                print(f'    ✓ 读取成功: {frame.shape}')
                cap.release()
                print('\\n摄像头测试通过 ✓')
                sys.exit(0)
            else:
                print(f'    ✗ 读取失败')
                cap.release()
        else:
            print(f'    ✗ 无法打开')
    except Exception as e:
        print(f'    ✗ 错误: {str(e)[:50]}...')

print('\\n✗ 所有摄像头初始化方法都失败')
print('请检查摄像头连接和配置')
sys.exit(1)
"
PYTHON_RESULT=$?
echo

# 8. 总结和建议
echo "8. 修复总结:"
echo "------------------------------------------"
if [ $PYTHON_RESULT -eq 0 ]; then
    echo "✓ 摄像头修复成功！"
    echo "摄像头现在应该可以正常工作。"
else
    echo "⚠ 摄像头测试失败，但配置已更新。"
    echo "可能需要重启树莓派。"
fi
echo

echo "9. 下一步操作:"
echo "------------------------------------------"
echo "1. 重启树莓派使配置生效:"
echo "   sudo reboot"
echo
echo "2. 重启后测试摄像头:"
echo "   libcamera-hello -t 0"
echo
echo "3. 运行修复版客户端:"
echo "   python3 raspberry_client_camera_fixed.py"
echo
echo "4. 如果仍有问题，检查:"
echo "   - 摄像头模块连接"
echo "   - 摄像头排线"
echo "   - 摄像头型号兼容性"
echo

echo "=========================================="
echo "修复脚本执行完成"
echo "=========================================="