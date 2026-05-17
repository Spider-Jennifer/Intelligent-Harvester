#!/bin/bash
# 树莓派4B依赖安装脚本

echo "========================================"
echo "树莓派4B苹果检测 - 依赖安装"
echo "========================================"
echo

# 更新系统
echo "更新系统包..."
sudo apt-get update
sudo apt-get upgrade -y

# 安装Python依赖
echo "安装Python依赖..."
sudo apt-get install -y python3-pip python3-opencv

# 安装Python包（最小集合）
echo "安装Python包..."
pip3 install --no-cache-dir numpy ultralytics

# 创建启动脚本
echo "创建启动脚本..."
cat > start_detection.sh << 'EOF'
#!/bin/bash
# 启动苹果检测

echo "启动树莓派苹果检测..."
echo "按 Ctrl+C 停止"

# 设置摄像头参数（树莓派专用）
v4l2-ctl --set-fmt-video=width=320,height=240,pixelformat=YUYV
v4l2-ctl --set-ctrl=brightness=50
v4l2-ctl --set-ctrl=contrast=10

# 运行检测程序
python3 raspberry_pi_ultra_light.py
EOF

chmod +x start_detection.sh

echo
echo "安装完成！"
echo "运行以下命令启动检测："
echo "  ./start_detection.sh"
echo
echo "或双击运行："
echo "  run_ultra_light.bat (Windows)"
echo "  start_detection.sh (Linux/Raspberry Pi)"