#!/bin/bash
# 树莓派摄像头单行测试脚本
# 直接在树莓派SSH中运行此命令

echo "树莓派摄像头测试 - 单行命令"
echo "=============================="

# 测试1: 检查摄像头模块
echo "1. 检查摄像头模块状态:"
vcgencmd get_camera

# 测试2: 检查摄像头设备
echo -e "\n2. 检查摄像头设备文件:"
ls -la /dev/video* 2>/dev/null || echo "未找到摄像头设备"

# 测试3: 测试libcamera
echo -e "\n3. 测试libcamera (2秒预览):"
timeout 2 libcamera-hello -t 0 2>/dev/null && echo "✓ libcamera工作正常" || echo "✗ libcamera测试失败"

# 测试4: Python摄像头测试
echo -e "\n4. Python摄像头测试:"
python3 -c "
import cv2
import time

print('Python OpenCV摄像头测试...')
success = False

for camera_id in [0, 1]:
    print(f'尝试摄像头索引 {camera_id}...')
    try:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f'  ✓ 摄像头 {camera_id} 已打开')
            # 等待摄像头初始化
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret:
                print(f'  ✓ 成功读取帧: {frame.shape}')
                success = True
            else:
                print(f'  ✗ 读取帧失败')
            cap.release()
            if success:
                break
        else:
            print(f'  ✗ 摄像头 {camera_id} 无法打开')
    except Exception as e:
        print(f'  ✗ 错误: {e}')

if not success:
    print('\\n摄像头测试失败，请检查:')
    print('1. 摄像头是否已启用: sudo raspi-config')
    print('2. 摄像头是否连接正常')
    print('3. OpenCV是否安装: pip3 install opencv-python')
"

# 测试5: 快速拍照测试
echo -e "\n5. 快速拍照测试:"
libcamera-jpeg -o /tmp/camera_test.jpg -t 1 -n 2>/dev/null
if [ -f "/tmp/camera_test.jpg" ]; then
    echo "✓ 拍照成功: /tmp/camera_test.jpg"
    ls -lh /tmp/camera_test.jpg
    rm -f /tmp/camera_test.jpg
else
    echo "✗ 拍照失败"
fi

echo -e "\n=============================="
echo "测试完成"
echo "如果摄像头测试失败，请运行: sudo raspi-config"
echo "选择: Interface Options -> Camera -> Yes -> Finish -> 重启"