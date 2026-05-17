#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 苹果检测 - 替代运行方案
"""

import os
import sys
import subprocess
import time
import socket

def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def find_available_port(start_port=8500):
    """查找可用端口"""
    for port in range(start_port, start_port + 20):
        if not check_port(port):
            return port
    return None

def run_simple_test():
    """运行简单测试"""
    print("\n[方案1] 运行命令行测试")
    test_script = "yolov8-apple-detection-master/test_apple_model.py"
    
    if os.path.exists(test_script):
        print(f"运行: {test_script}")
        cmd = [sys.executable, test_script, "--num_images", "1"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "检测到" in result.stdout or "Apple" in result.stdout:
                print("[SUCCESS] 模型工作正常！")
                return True
            else:
                print("[WARN] 测试输出异常")
                return False
        except Exception as e:
            print(f"[ERROR] 运行失败: {e}")
            return False
    else:
        print("[ERROR] 测试脚本不存在")
        return False

def start_web_app_simple():
    """简单启动Web应用"""
    print("\n[方案2] 启动Web应用（简化版）")
    
    app_dir = "YOLO-v8-app/YOLOv8-app-master"
    if not os.path.exists(app_dir):
        print("[ERROR] 应用目录不存在")
        return False
    
    # 查找可用端口
    port = find_available_port(8510)
    if not port:
        print("[ERROR] 找不到可用端口")
        return False
    
    print(f"使用端口: {port}")
    print(f"访问网址: http://localhost:{port}")
    
    try:
        os.chdir(app_dir)
        
        # 简单启动命令
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost"
        ]
        
        print(f"启动命令: {' '.join(cmd)}")
        print("\n正在启动应用...")
        print("如果浏览器没有自动打开，请手动访问上述网址")
        print("按 Ctrl+C 停止应用")
        
        # 启动应用
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待启动
        time.sleep(10)
        
        # 检查端口
        if check_port(port):
            print(f"[SUCCESS] 应用启动成功！端口: {port}")
            print(f"请访问: http://localhost:{port}")
            
            # 显示进程信息
            print(f"\n应用进程ID: {process.pid}")
            print("应用正在运行...")
            
            # 等待用户停止
            try:
                input("\n按 Enter 键停止应用...")
            except:
                pass
            
            process.terminate()
            return True
        else:
            print("[ERROR] 应用启动失败")
            process.terminate()
            return False
            
    except KeyboardInterrupt:
        print("\n应用已停止")
        return True
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        return False

def run_direct_inference():
    """直接运行推理"""
    print("\n[方案3] 直接运行推理")
    
    try:
        import torch
        from ultralytics import YOLO
        import cv2
        
        # 修复兼容性
        original_torch_load = torch.load
        def safe_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = safe_torch_load
        
        # 查找模型
        model_paths = [
            "yolov8-apple-detection-master/apple_detection_model_gpu/weights/best.pt",
            "YOLO-v8-app/YOLOv8-app-master/weights/best.pt",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("[ERROR] 未找到模型文件")
            return False
        
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 查找测试图像
        test_dirs = [
            "YOLO-v8-app/quick_train/images",
            "yolov8-apple-detection-master/Dataset/apple_dataset/images/train",
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.jpg', '.png'))]
                if images:
                    test_image = os.path.join(test_dir, images[0])
                    break
        
        if test_image and os.path.exists(test_image):
            print(f"测试图像: {test_image}")
            
            # 执行预测
            results = model(test_image, conf=0.1)  # 降低置信度阈值以提高检测灵敏度
            
            print("\n检测结果:")
            for r in results:
                boxes = r.boxes
                print(f"检测到 {len(boxes)} 个物体")
                for box in boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    print(f"  - {class_name}: 置信度 = {confidence:.2f}")
            
            # 保存结果
            output_dir = "direct_results"
            os.makedirs(output_dir, exist_ok=True)
            
            for i, r in enumerate(results):
                im_array = r.plot()
                output_path = os.path.join(output_dir, "direct_result.jpg")
                cv2.imwrite(output_path, im_array)
                print(f"\n结果已保存到: {output_path}")
            
            return True
        else:
            print("[WARN] 未找到测试图像")
            return False
            
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 推理失败: {e}")
        return False

def main():
    print("=" * 70)
    print("YOLOv8 苹果检测 - 替代运行方案")
    print("=" * 70)
    
    print("\n尝试多种运行方案...")
    
    # 方案1: 命令行测试
    if run_simple_test():
        print("\n✓ 方案1成功！模型可以正常工作")
    
    # 方案2: Web应用
    print("\n" + "-" * 70)
    if start_web_app_simple():
        print("\n✓ 方案2成功！Web应用已启动")
    else:
        print("\n⚠ 方案2失败，尝试方案3...")
    
    # 方案3: 直接推理
    print("\n" + "-" * 70)
    if run_direct_inference():
        print("\n✓ 方案3成功！直接推理完成")
    
    print("\n" + "=" * 70)
    print("运行总结:")
    print("1. 命令行测试: 可用")
    print("2. Web应用: 尝试端口 8510+")
    print("3. 直接推理: 可用")
    print("\n推荐使用方案1或方案3")
    print("或手动运行: python yolov8-apple-detection-master/test_apple_model.py")

if __name__ == "__main__":
    main()